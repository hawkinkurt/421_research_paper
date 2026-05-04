"""
cs_did_analysis.py
==================
Callaway-Sant'Anna (CS, 2021 J.Econometrics) group-time ATT estimator
as a robustness check for the TWFE event study (M9), running alongside
the BJS imputation estimator. Mirrors bjs_imputation_analysis.py in
configuration, sample selection, and output format for direct
comparability.

Motivation
----------
Like BJS, CS addresses bias from forbidden comparisons in staggered
TWFE. Whereas BJS imputes counterfactuals from a single FE model,
CS estimates a separate ATT(g, t) for every cohort g and time t,
using a *clean* control group (never-treated or not-yet-treated)
in each cell. Aggregation to event-time and pooled ATT is then a
weighted average over (g, t) cells.

Implementation choices (matching BJS for direct comparability)
--------------------------------------------------------------
- Control group: not-yet-treated (matches BJS untreated pool definition).
- Aggregation:   event-study + pooled post-treatment ATT.
- Std errors:    pair-clustered bootstrap, 500 reps (matches BJS).
- Covariates:    same control set as BJS, absorbed via residualization
                 of the outcome on controls within the untreated pool
                 before CS aggregation. This is a Frisch-Waugh-style
                 partialling that delivers covariate-adjusted ATTs.
- Pair FEs:      absorbed via within-pair demeaning of the (already
                 covariate-residualized) outcome.

References
----------
Callaway & Sant'Anna (2021), "Difference-in-Differences with Multiple
    Time Periods", J. Econometrics 225(2), 200-230.
Borusyak, Jaravel & Spiess (2024), "Revisiting Event Study Designs",
    Review of Economic Studies 91(6).
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIGURATION  (mirrors bjs_imputation_analysis.py)
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'

IMDB_ANALYSIS_FILE = (f'{PROJECT_DIR}\\data\\processed\\imdb\\'
                      f'imdb_gravity_analysis.csv')
BATIS_ANALYSIS_FILE = (f'{PROJECT_DIR}\\data\\processed\\'
                       f'batis_sk1_gravity_analysis.csv')
INCENTIVE_FILE = (f'{PROJECT_DIR}\\data\\raw\\base\\'
                  f'film_incentive_intro_dates.csv')

OUTPUT_DIR = f'{PROJECT_DIR}\\output\\cs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Event window — matches existing M9 / BJS
EVENT_WINDOW_MIN = -5
EVENT_WINDOW_MAX = 10

ANGLO = {'US', 'GB', 'CA', 'AU', 'NZ'}

# Bootstrap configuration — matches BJS
N_BOOTSTRAP = 500
RANDOM_SEED = 20251215

# Dataset configurations — matches BJS exactly
DATASETS = {
    'IMDB_count': {
        'file': IMDB_ANALYSIS_FILE,
        'outcome': 'log_num_films',
        'label': 'IMDb film count (log)',
        'controls': [
            'log_gdp_importer', 'log_gdp_exporter',
            'remoteness_importer', 'remoteness_exporter',
            'efw_exporter', 'efw_importer',
            'rta', 'incentive_importer',
        ],
    },
    'IMDB_budget': {
        'file': IMDB_ANALYSIS_FILE,
        'outcome': 'log_budget',
        'label': 'IMDb budget (log)',
        'controls': [
            'log_gdp_importer', 'log_gdp_exporter',
            'remoteness_importer', 'remoteness_exporter',
            'efw_exporter', 'efw_importer',
            'rta', 'incentive_importer',
        ],
    },
}

SAMPLE_VARIANTS = {
    'full':          {'drop_anglo_exp': False, 'drop_anglo_pair': False},
    'no_anglo_exp':  {'drop_anglo_exp': True,  'drop_anglo_pair': False},
    'no_anglo_pair': {'drop_anglo_exp': False, 'drop_anglo_pair': True},
}


# =============================================================================
# PANEL PREPARATION (parallel to BJS prepare_panel)
# =============================================================================

def prepare_panel(df_src, outcome, controls, incentive_dict,
                  drop_anglo_exp=False, drop_anglo_pair=False):
    """
    Clean and annotate the panel for CS estimation. Same logic as BJS
    prepare_panel. Adds:
      pair_id           : exporter_importer string ID
      intro_year_exp    : treatment year for exporter (NaN if never)
      treated_ever      : 1 if exporter ever treated in sample
      treated_now       : 1 if year >= intro_year_exp
      rel_year          : year - intro_year_exp (NaN for never-treated)
      in_untreated_pool : 1 if obs usable as a clean control
                          (not-yet-treated OR never-treated)
      in_event_window   : 1 if rel_year in [EVENT_WINDOW_MIN, MAX]
    """
    d = df_src.copy()

    needed = [outcome] + controls
    missing_before = len(d)
    d = d.dropna(subset=needed)
    print(f"  Dropped {missing_before - len(d):,} obs with missing "
          f"outcome/controls ({len(d):,} remain)")

    if drop_anglo_exp:
        d = d[~d['exporter'].isin(ANGLO)].copy()
        print(f"  After dropping Anglo exporters: {len(d):,} obs")
    if drop_anglo_pair:
        d = d[~(d['exporter'].isin(ANGLO) | d['importer'].isin(ANGLO))].copy()
        print(f"  After dropping any-Anglo pairs: {len(d):,} obs")

    d['pair_id'] = d['exporter'].astype(str) + '_' + d['importer'].astype(str)

    d['intro_year_exp'] = d['exporter'].map(incentive_dict)
    d['treated_ever'] = d['intro_year_exp'].notna().astype(int)
    d['treated_now'] = (
        d['intro_year_exp'].notna() & (d['year'] >= d['intro_year_exp'])
    ).astype(int)

    d['rel_year'] = np.where(
        d['intro_year_exp'].notna(),
        d['year'] - d['intro_year_exp'],
        np.nan
    )

    d['in_untreated_pool'] = (d['treated_now'] == 0).astype(int)

    d['in_event_window'] = (
        (d['treated_now'] == 1)
        & (d['rel_year'] >= EVENT_WINDOW_MIN)
        & (d['rel_year'] <= EVENT_WINDOW_MAX)
    ).astype(int)

    return d


# =============================================================================
# COVARIATE RESIDUALIZATION + PAIR DEMEANING
# =============================================================================

def residualize_outcome(d, outcome, controls):
    """
    Partial out controls from the outcome.

    Step 1: regress Y on controls in the untreated pool, store coeffs.
    Step 2: subtract X*beta_hat from Y for ALL obs (treated + untreated).
    Step 3: within-pair demean the residualized outcome.

    The result `_y_clean` is the outcome with covariates and pair FEs
    removed. CS then operates on `_y_clean` to recover ATTs that match
    a covariate-adjusted, pair-FE specification.

    Returns d with new column '_y_clean'.
    """
    d = d.copy()
    untreated = d[d['in_untreated_pool'] == 1]

    if len(untreated) < 100:
        raise ValueError(
            f"Only {len(untreated)} untreated obs — too few to fit "
            f"covariate model.")

    # Fit Y ~ controls + const on untreated pool
    X_u = sm.add_constant(untreated[controls].values, has_constant='add')
    y_u = untreated[outcome].values
    ols_u = sm.OLS(y_u, X_u).fit()

    # Apply same coefficients to ALL obs to get covariate residual
    X_all = sm.add_constant(d[controls].values, has_constant='add')
    y_pred = ols_u.predict(X_all)
    d['_y_resid'] = d[outcome].values - y_pred

    # Within-pair demean to absorb pair FEs
    pair_means = d.groupby('pair_id')['_y_resid'].transform('mean')
    d['_y_clean'] = d['_y_resid'] - pair_means

    return d


# =============================================================================
# CALLAWAY-SANT'ANNA CORE: GROUP-TIME ATT
# =============================================================================

def compute_att_gt(d, g, t, ycol='_y_clean'):
    """
    Compute ATT(g, t) using not-yet-treated controls.

        ATT(g,t) = [E(Y_t | G=g) - E(Y_{g-1} | G=g)]
                 - [E(Y_t | not yet treated by t) - E(Y_{g-1} | not yet treated by t)]

    Uses panel structure: requires same units observed at both t0=g-1
    and t. Not-yet-treated control = obs where treated_now==0 at time t,
    i.e. intro_year_exp is NaN OR intro_year_exp > t.

    Returns (att, n_treated_in_cell). If either group is empty or the
    baseline year is missing, returns (NaN, 0).
    """
    t0 = g - 1

    # Treated cohort: pairs whose exporter has intro_year_exp == g
    treated = d[d['intro_year_exp'] == g]
    treat_t0 = treated[treated['year'] == t0].set_index('pair_id')[ycol]
    treat_t  = treated[treated['year'] == t].set_index('pair_id')[ycol]
    common_treat = treat_t0.index.intersection(treat_t.index)
    if len(common_treat) == 0:
        return np.nan, 0
    delta_treat = (treat_t.loc[common_treat]
                   - treat_t0.loc[common_treat]).mean()

    # Not-yet-treated control at time t: never-treated OR intro > t
    is_nyt = d['intro_year_exp'].isna() | (d['intro_year_exp'] > t)
    control = d[is_nyt]
    ctrl_t0 = control[control['year'] == t0].set_index('pair_id')[ycol]
    ctrl_t  = control[control['year'] == t].set_index('pair_id')[ycol]
    common_ctrl = ctrl_t0.index.intersection(ctrl_t.index)
    if len(common_ctrl) == 0:
        return np.nan, 0
    delta_ctrl = (ctrl_t.loc[common_ctrl]
                  - ctrl_t0.loc[common_ctrl]).mean()

    att = delta_treat - delta_ctrl
    return att, len(common_treat)


def estimate_cs_event_study(d):
    """
    Compute all ATT(g, t) for cohorts and times in the panel, then
    aggregate to event-time and pooled.

    Returns:
        att_e:     dict {h: ATT_h}, h = relative event time
        n_e:       dict {h: n_treated_at_h}
        att_pooled: pooled post-treatment ATT (weighted avg over h>=0)
        n_pooled:  total treated obs underlying pooled ATT
    """
    cohorts = sorted(d.loc[d['intro_year_exp'].notna(),
                           'intro_year_exp'].unique())

    gt_results = []
    years_in_panel = sorted(d['year'].unique())

    for g in cohorts:
        g = int(g)
        if g - 1 < min(years_in_panel):
            # No baseline available
            continue
        for t in years_in_panel:
            h = int(t - g)
            if h < EVENT_WINDOW_MIN or h > EVENT_WINDOW_MAX:
                continue
            if h == -1:
                continue  # baseline by construction
            att, n = compute_att_gt(d, g, int(t))
            if not np.isnan(att) and n > 0:
                gt_results.append((g, int(t), h, att, n))

    if len(gt_results) == 0:
        return {}, {}, np.nan, 0

    gt_df = pd.DataFrame(gt_results,
                         columns=['g', 't', 'h', 'att', 'n'])

    # Event-time aggregation: weighted average over cohorts at each h
    att_e, n_e = {}, {}
    for h, sub in gt_df.groupby('h'):
        w = sub['n'].values.astype(float)
        att_e[int(h)] = float(np.average(sub['att'].values, weights=w))
        n_e[int(h)] = int(w.sum())

    # Pooled post-treatment ATT (h >= 0)
    post = gt_df[gt_df['h'] >= 0]
    if len(post) == 0:
        att_pooled, n_pooled = np.nan, 0
    else:
        w = post['n'].values.astype(float)
        att_pooled = float(np.average(post['att'].values, weights=w))
        n_pooled = int(w.sum())

    return att_e, n_e, att_pooled, n_pooled


# =============================================================================
# PAIR-CLUSTER BOOTSTRAP
# =============================================================================

def pair_cluster_bootstrap(d, outcome, controls, n_boot, seed):
    """
    Pair-cluster bootstrap. Each rep:
      1. Resample pair_id values with replacement (n_pairs draws).
      2. Stack obs from sampled pairs (with unique pair_id per draw to
         keep clusters distinct).
      3. Re-residualize outcome and re-estimate CS event study.
      4. Store ATT_h and pooled ATT for the rep.

    Returns boot_att_h dict and boot_pooled list.
    """
    rng = np.random.default_rng(seed)
    pairs = d['pair_id'].unique()
    n_pairs = len(pairs)

    # Pre-build pair -> rows lookup
    pair_to_obs = {p: d[d['pair_id'] == p] for p in pairs}

    boot_att_h = {h: [] for h in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1)
                  if h != -1}
    boot_pooled = []

    n_failed = 0
    t0 = time.time()

    for b in range(n_boot):
        if b == 0 or (b + 1) % 50 == 0 or b == n_boot - 1:
            elapsed = time.time() - t0
            if b > 0:
                rate = elapsed / (b + 1)
                remaining = rate * (n_boot - b - 1)
                print(f"    Bootstrap rep {b+1}/{n_boot} "
                      f"(elapsed: {elapsed:.0f}s, "
                      f"remaining: ~{remaining:.0f}s, "
                      f"failed: {n_failed})")
            else:
                print(f"    Bootstrap rep {b+1}/{n_boot} (elapsed: 0s)")

        sample_pairs = rng.choice(pairs, size=n_pairs, replace=True)

        # Build resampled df, give each draw a unique pair_id so that
        # demeaning treats them as separate clusters
        chunks = []
        for i, p in enumerate(sample_pairs):
            sub = pair_to_obs[p].copy()
            sub['pair_id'] = f"{p}__b{i}"
            chunks.append(sub)
        df_b = pd.concat(chunks, ignore_index=True)

        try:
            df_b_clean = residualize_outcome(df_b, outcome, controls)
            att_e_b, _, att_pooled_b, _ = estimate_cs_event_study(df_b_clean)
            for h in boot_att_h:
                boot_att_h[h].append(att_e_b.get(h, np.nan))
            boot_pooled.append(att_pooled_b)
        except Exception:
            n_failed += 1
            for h in boot_att_h:
                boot_att_h[h].append(np.nan)
            boot_pooled.append(np.nan)

    elapsed = time.time() - t0
    print(f"    Bootstrap complete: {n_boot} reps in {elapsed:.0f}s "
          f"({n_failed} failed)")

    return boot_att_h, boot_pooled


# =============================================================================
# OUTPUT (parallels BJS print_event_study)
# =============================================================================

def summarize_bootstrap(att_e, n_e, att_pooled, n_pooled,
                        boot_att_h, boot_pooled):
    """Compute SEs and 95% percentile CIs from bootstrap draws."""
    rows = []
    for h in sorted(boot_att_h.keys()):
        boot_vals = np.array([v for v in boot_att_h[h] if not np.isnan(v)])
        att_h = att_e.get(h, np.nan)
        n_h = n_e.get(h, 0)
        if len(boot_vals) < 50:
            rows.append({'rel_year': h, 'att_h': att_h,
                         'se_h': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                         'n_treated': n_h, 'n_boot': len(boot_vals)})
        else:
            rows.append({
                'rel_year': h, 'att_h': att_h,
                'se_h': float(np.std(boot_vals, ddof=1)),
                'ci_lo': float(np.percentile(boot_vals, 2.5)),
                'ci_hi': float(np.percentile(boot_vals, 97.5)),
                'n_treated': n_h, 'n_boot': len(boot_vals),
            })

    event_summary = pd.DataFrame(rows).set_index('rel_year')

    boot_p = np.array([v for v in boot_pooled if not np.isnan(v)])
    if len(boot_p) < 50:
        overall = {'att': att_pooled, 'se': np.nan,
                   'ci_lo': np.nan, 'ci_hi': np.nan,
                   'n_boot': len(boot_p)}
    else:
        overall = {
            'att': att_pooled,
            'se': float(np.std(boot_p, ddof=1)),
            'ci_lo': float(np.percentile(boot_p, 2.5)),
            'ci_hi': float(np.percentile(boot_p, 97.5)),
            'n_boot': len(boot_p),
        }

    return event_summary, overall


def print_event_study(event_summary, overall, n_treated_total, label):
    """Console output mirroring BJS format."""
    print(f"\n  {'Rel Year':>10} {'ATT(h)':>10} {'SE':>10} "
          f"{'95% CI':>22} {'Sig':>5} {'N':>6}")
    print(f"  {'-' * 68}")

    def stars(coef, se):
        if np.isnan(se) or se == 0:
            return ""
        z = abs(coef / se)
        if z > 2.58:
            return "***"
        elif z > 1.96:
            return "**"
        elif z > 1.645:
            return "*"
        return ""

    for h, row in event_summary.iterrows():
        if np.isnan(row['att_h']):
            print(f"  {'t' + format(h, '+d'):>10} {'--':>10}")
            continue
        s = stars(row['att_h'], row['se_h'])
        ci_str = (f"[{row['ci_lo']:>7.3f},{row['ci_hi']:>8.3f}]"
                  if not np.isnan(row['ci_lo']) else "[ bootstrap failed ]")
        print(f"  {'t' + format(h, '+d'):>10} {row['att_h']:>10.4f} "
              f"{row['se_h']:>10.4f} {ci_str} {s:>5} "
              f"{int(row['n_treated']):>6}")

    print(f"\n  {'-' * 68}")
    if not np.isnan(overall['se']):
        s = ("***" if abs(overall['att'] / overall['se']) > 2.58
             else "**" if abs(overall['att'] / overall['se']) > 1.96
             else "*" if abs(overall['att'] / overall['se']) > 1.645
             else "")
        print(f"  Pooled ATT: {overall['att']:.4f} "
              f"(SE = {overall['se']:.4f}) {s}")
        print(f"    95% CI: [{overall['ci_lo']:.4f}, "
              f"{overall['ci_hi']:.4f}]")
        print(f"    Based on {n_treated_total:,} treated obs, "
              f"{overall['n_boot']} successful bootstrap reps")
    else:
        print(f"  Pooled ATT: {overall['att']:.4f} "
              f"(bootstrap SE unavailable)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("CS DiD ESTIMATOR — ROBUSTNESS CHECK FOR M9 EVENT STUDY")
    print("=" * 70)

    # Load incentive intro dates
    incentives = pd.read_csv(INCENTIVE_FILE)
    incentive_dict = dict(zip(
        incentives['country_iso2'].astype(str),
        incentives['incentive_intro_year']
    ))
    print(f"\nLoaded {len([v for v in incentive_dict.values() if not pd.isna(v)])}"
          f" countries with incentive intro dates")

    all_results = {}

    for ds_key, ds_cfg in DATASETS.items():
        print(f"\n\n{'#' * 70}")
        print(f"# DATASET: {ds_key} — {ds_cfg['label']}")
        print(f"{'#' * 70}")

        try:
            df_raw = pd.read_csv(ds_cfg['file'])
        except FileNotFoundError:
            print(f"  FILE NOT FOUND: {ds_cfg['file']} — skipping")
            continue

        print(f"Loaded {len(df_raw):,} rows from "
              f"{os.path.basename(ds_cfg['file'])}")

        if ds_cfg['outcome'] not in df_raw.columns:
            print(f"  Outcome {ds_cfg['outcome']} not in data — skipping")
            continue

        controls_present = [c for c in ds_cfg['controls']
                            if c in df_raw.columns]
        missing_ctrls = set(ds_cfg['controls']) - set(controls_present)
        if missing_ctrls:
            print(f"  WARNING: dropping missing controls: {missing_ctrls}")

        for sv_key, sv_cfg in SAMPLE_VARIANTS.items():
            print(f"\n{'=' * 70}")
            print(f"  SAMPLE: {sv_key}")
            print(f"{'=' * 70}")

            d = prepare_panel(
                df_raw, ds_cfg['outcome'], controls_present,
                incentive_dict,
                drop_anglo_exp=sv_cfg['drop_anglo_exp'],
                drop_anglo_pair=sv_cfg['drop_anglo_pair'],
            )

            n_treated_window = d['in_event_window'].sum()
            if n_treated_window < 30:
                print(f"  Only {n_treated_window} treated obs in event "
                      f"window — skipping")
                continue

            print(f"  Untreated pool: {d['in_untreated_pool'].sum():,} obs")
            print(f"  Treated obs in event window: {n_treated_window:,}")

            try:
                d_clean = residualize_outcome(
                    d, ds_cfg['outcome'], controls_present)
            except ValueError as e:
                print(f"  Residualization failed: {e} — skipping")
                continue

            att_e, n_e, att_pooled, n_pooled = estimate_cs_event_study(d_clean)

            if not att_e:
                print(f"  No estimable ATT(g,t) cells — skipping")
                continue

            print(f"\n  Running pair-cluster bootstrap "
                  f"({N_BOOTSTRAP} reps)...")
            boot_att_h, boot_pooled = pair_cluster_bootstrap(
                d, ds_cfg['outcome'], controls_present,
                N_BOOTSTRAP, RANDOM_SEED)

            event_summary, overall = summarize_bootstrap(
                att_e, n_e, att_pooled, n_pooled,
                boot_att_h, boot_pooled)

            print_event_study(event_summary, overall, n_pooled,
                              f"{ds_key} / {sv_key}")

            # Export to CSV
            result_key = f"{ds_key}__{sv_key}"
            outpath = os.path.join(
                OUTPUT_DIR, f'cs_event_study_{result_key}.csv')
            export = event_summary.reset_index().copy()
            export['dataset'] = ds_key
            export['sample'] = sv_key
            export.to_csv(outpath, index=False)
            print(f"\n  Saved: {outpath}")

            all_results[result_key] = {
                'event_summary': event_summary,
                'overall': overall,
                'n_treated': n_pooled,
            }

    # Combined pooled-ATT summary
    if all_results:
        print(f"\n\n{'#' * 70}")
        print("# OVERALL ATT COMPARISON ACROSS DATASETS / SAMPLES")
        print(f"{'#' * 70}")
        print(f"\n  {'Dataset / Sample':<35} {'ATT':>10} {'SE':>10} "
              f"{'95% CI':>22}")
        print(f"  {'-' * 80}")
        summary_rows = []
        for key, res in all_results.items():
            o = res['overall']
            ci = (f"[{o['ci_lo']:>7.3f},{o['ci_hi']:>8.3f}]"
                  if not np.isnan(o['ci_lo']) else "[bootstrap failed ]")
            print(f"  {key:<35} {o['att']:>10.4f} {o['se']:>10.4f} {ci}")
            summary_rows.append({
                'spec': key, 'att': o['att'], 'se': o['se'],
                'ci_lo': o['ci_lo'], 'ci_hi': o['ci_hi'],
                'n_treated': res['n_treated'],
            })
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'cs_pooled_att_summary.csv'),
            index=False)
        print(f"\n  Saved: "
              f"{os.path.join(OUTPUT_DIR, 'cs_pooled_att_summary.csv')}")

    elapsed = time.time() - t_start
    print(f"\n\n{'=' * 70}")
    print(f"COMPLETE — total runtime: {elapsed / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()