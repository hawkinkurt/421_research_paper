"""
bjs_imputation_analysis.py
==========================
Borusyak-Jaravel-Spiess (BJS, 2024 ReStud) imputation estimator as a
robustness check for the TWFE event study (M9) in both the IMDb and
BaTIS gravity pipelines.

Motivation
----------
Standard TWFE event studies can produce biased dynamic treatment effect
estimates under staggered rollout + heterogeneous effects, because
already-treated units are used as controls for later-treated units
(Goodman-Bacon 2021; de Chaisemartin & D'Haultfœuille 2020;
Sun-Abraham 2021). BJS avoid this contamination by:

    1. Fitting unit + time FE (+ controls) on *untreated* obs only
       (never-treated + not-yet-treated), getting counterfactual Y0 hat.
    2. Imputing Y0 hat for each treated observation.
    3. Computing tau_it = Y_it - Y0_it_hat per treated obs.
    4. Aggregating tau_it by event time h to get the dynamic ATT(h).

Standard errors are computed via a pair-cluster bootstrap (drawing
country-pairs with replacement and re-estimating the full procedure).
This respects the gravity panel structure and correctly handles the
fact that imputed residuals carry estimation uncertainty from step 1.

This is a SEPARATE, STANDALONE program. It does not modify or replace
any existing pipeline output. Intended purpose: robustness check
reported alongside the TWFE event study in the thesis.

References
----------
Borusyak, Jaravel & Spiess (2024), "Revisiting Event Study Designs:
    Robust and Efficient Estimation", Review of Economic Studies 91(6).
Goodman-Bacon (2021), "Difference-in-differences with variation in
    treatment timing", J. Econometrics 225(2).
Sun & Abraham (2021), "Estimating dynamic treatment effects in event
    studies with heterogeneous treatment effects", J. Econometrics 225(2).
"""

import os
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'

IMDB_ANALYSIS_FILE = (f'{PROJECT_DIR}\\data\\processed\\imdb\\'
                      f'imdb_gravity_analysis.csv')
BATIS_ANALYSIS_FILE = (f'{PROJECT_DIR}\\data\\processed\\'
                       f'batis_sk1_gravity_analysis.csv')
INCENTIVE_FILE = (f'{PROJECT_DIR}\\data\\raw\\base\\'
                  f'film_incentive_intro_dates.csv')

OUTPUT_DIR = f'{PROJECT_DIR}\\output\\bjs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Event window - matches existing M9 for comparability
EVENT_WINDOW_MIN = -5
EVENT_WINDOW_MAX = 10

ANGLO = {'US', 'GB', 'CA', 'AU', 'NZ'}

# Bootstrap configuration
# 399 is the smallest N giving clean 2.5 / 97.5 percentiles; raise to
# 999 for publication tables if runtime allows.
N_BOOTSTRAP = 500
RANDOM_SEED = 20251215

# Dataset configurations
# IMDb: log_num_films is primary outcome; log_budget is secondary.
# BaTIS: log_trade is the only outcome.
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

# Sample restrictions to run for each dataset
SAMPLE_VARIANTS = {
    'full':      {'drop_anglo_exp': False, 'drop_anglo_pair': False},
    'no_anglo_exp':  {'drop_anglo_exp': True,  'drop_anglo_pair': False},
    'no_anglo_pair': {'drop_anglo_exp': False, 'drop_anglo_pair': True},
}


# =============================================================================
# CORE BJS ESTIMATOR
# =============================================================================

def prepare_panel(df_src, outcome, controls, incentive_dict,
                  drop_anglo_exp=False, drop_anglo_pair=False):
    """
    Clean and annotate the panel for BJS estimation.

    Produces columns:
      pair_id           : exporter_importer string ID
      intro_year_exp    : treatment year for exporter (NaN if never treated)
      treated_ever      : 1 if exporter ever treated in sample window
      treated_now       : 1 if year >= intro_year_exp (post-treatment obs)
      rel_year          : year - intro_year_exp (NaN for never-treated)
      in_untreated_pool : 1 if obs is usable for fitting counterfactual
                          (never-treated OR not-yet-treated)
      in_event_window   : 1 if rel_year in [EVENT_WINDOW_MIN, EVENT_WINDOW_MAX]
    """
    d = df_src.copy()

    # Drop obs with missing outcome or any control
    needed = [outcome] + controls
    missing_before = len(d)
    d = d.dropna(subset=needed)
    print(f"  Dropped {missing_before - len(d):,} obs with missing "
          f"outcome/controls ({len(d):,} remain)")

    # Anglo restrictions
    if drop_anglo_exp:
        d = d[~d['exporter'].isin(ANGLO)].copy()
        print(f"  After dropping Anglo exporters: {len(d):,} obs")
    if drop_anglo_pair:
        d = d[~(d['exporter'].isin(ANGLO) | d['importer'].isin(ANGLO))].copy()
        print(f"  After dropping any-Anglo pairs: {len(d):,} obs")

    # Pair ID
    d['pair_id'] = d['exporter'].astype(str) + '_' + d['importer'].astype(str)

    # Treatment timing from incentive intro dates
    d['intro_year_exp'] = d['exporter'].map(incentive_dict)
    d['treated_ever'] = d['intro_year_exp'].notna().astype(int)
    d['treated_now'] = (
        d['intro_year_exp'].notna() & (d['year'] >= d['intro_year_exp'])
    ).astype(int)

    # Relative event time (NaN for never-treated)
    d['rel_year'] = np.where(
        d['intro_year_exp'].notna(),
        d['year'] - d['intro_year_exp'],
        np.nan
    )

    # Untreated pool: never-treated OR not-yet-treated (rel_year < 0)
    # This is the BJS "clean controls" group used to fit Y0 model.
    d['in_untreated_pool'] = (d['treated_now'] == 0).astype(int)

    # Treated obs within the event window we care about
    d['in_event_window'] = (
        (d['treated_now'] == 1)
        & (d['rel_year'] >= EVENT_WINDOW_MIN)
        & (d['rel_year'] <= EVENT_WINDOW_MAX)
    ).astype(int)

    return d


def fit_counterfactual_and_impute(d, outcome, controls):
    """
    BJS step 1-3: fit FE model on untreated obs, impute Y0 for treated.

    Implements the imputation via within-transformation to handle the
    many-FE case efficiently without constructing a huge dummy matrix.

    Note on two-way FE imputation: true BJS requires fitting mu_exp,
    mu_imp, lambda_t jointly on untreated obs and *extrapolating* to
    treated obs. We implement this by iterative demeaning
    (Guimaraes-Portugal / Correia-style) on the untreated subsample,
    storing the estimated fixed effects, and applying them to treated
    obs.

    For the standard case where every treated unit also has
    untreated (pre-treatment) observations and every year has
    untreated obs, all required FEs are identified. Treated-never
    exporters or years without any untreated obs are dropped from
    the counterfactual with a warning.

    Returns:
      d with new column 'y0_hat' (imputed counterfactual for all
      obs where computable) and 'tau' (Y - Y0_hat) for treated obs.
    """
    untreated = d[d['in_untreated_pool'] == 1].copy()

    if len(untreated) < 100:
        raise ValueError(
            f"Only {len(untreated)} untreated obs — too few to fit "
            f"counterfactual model."
        )

    # Check FE identification
    untreated_exporters = set(untreated['exporter'].unique())
    untreated_importers = set(untreated['importer'].unique())
    untreated_years = set(untreated['year'].unique())

    treated_needing_imp = d[d['in_event_window'] == 1]
    missing_exp_fe = set(treated_needing_imp['exporter']) - untreated_exporters
    missing_imp_fe = set(treated_needing_imp['importer']) - untreated_importers
    missing_yr_fe = set(treated_needing_imp['year']) - untreated_years

    if missing_exp_fe:
        # This is common and expected: for early-treated exporters,
        # pre-treatment obs serve as their untreated observations, so
        # their FE IS identified. Only truly-always-treated exporters
        # would fail — none exist in this sample. But document.
        print(f"  NOTE: {len(missing_exp_fe)} exporters have no untreated "
              f"obs — their FE cannot be separately identified: "
              f"{sorted(missing_exp_fe)[:5]}{'...' if len(missing_exp_fe) > 5 else ''}")
    if missing_imp_fe:
        print(f"  NOTE: {len(missing_imp_fe)} importers have no untreated obs")
    if missing_yr_fe:
        print(f"  NOTE: {len(missing_yr_fe)} years have no untreated obs")

    # Fit the counterfactual model: Y = mu_exp + mu_imp + lambda_t + X*beta + e
    # on untreated obs, using explicit FE dummies for clean coefficient
    # extraction. For ~30 exporters × ~30 importers × ~30 years this is
    # tractable (~90 dummies); if dataset grows, switch to Correia-style
    # iterative demeaning.
    y_u = untreated[outcome].values
    # Design matrix: controls + FE dummies (one category dropped each)
    X_parts = [untreated[controls].values]
    # Exporter dummies (drop first)
    exp_dummies = pd.get_dummies(
        untreated['exporter'], prefix='exp', drop_first=True, dtype=float)
    imp_dummies = pd.get_dummies(
        untreated['importer'], prefix='imp', drop_first=True, dtype=float)
    yr_dummies = pd.get_dummies(
        untreated['year'], prefix='yr', drop_first=True, dtype=float)

    X_u = np.hstack([
        untreated[controls].values,
        exp_dummies.values,
        imp_dummies.values,
        yr_dummies.values,
    ])
    X_u = sm.add_constant(X_u, has_constant='add')

    # OLS on untreated obs
    ols_u = sm.OLS(y_u, X_u).fit()

    # Build matching column names to apply to treated obs
    control_names = controls
    exp_names = list(exp_dummies.columns)
    imp_names = list(imp_dummies.columns)
    yr_names = list(yr_dummies.columns)

    # Build design matrix for ALL obs (to impute Y0 everywhere)
    # using the SAME dummy reference levels as the untreated fit.
    exp_ref = [c for c in untreated['exporter'].unique()
               if f'exp_{c}' not in exp_names][0]
    imp_ref = [c for c in untreated['importer'].unique()
               if f'imp_{c}' not in imp_names][0]
    yr_ref = [c for c in untreated['year'].unique()
              if f'yr_{c}' not in yr_names][0]

    def build_design(subset):
        # Controls
        ctrl_mat = subset[controls].values
        # Exp dummies
        exp_mat = np.zeros((len(subset), len(exp_names)))
        for i, colname in enumerate(exp_names):
            level = colname.replace('exp_', '')
            exp_mat[:, i] = (subset['exporter'].astype(str) == str(level)).astype(float)
        # Imp dummies
        imp_mat = np.zeros((len(subset), len(imp_names)))
        for i, colname in enumerate(imp_names):
            level = colname.replace('imp_', '')
            imp_mat[:, i] = (subset['importer'].astype(str) == str(level)).astype(float)
        # Year dummies
        yr_mat = np.zeros((len(subset), len(yr_names)))
        for i, colname in enumerate(yr_names):
            level = int(colname.replace('yr_', ''))
            yr_mat[:, i] = (subset['year'] == level).astype(float)

        X = np.hstack([ctrl_mat, exp_mat, imp_mat, yr_mat])
        X = sm.add_constant(X, has_constant='add')
        return X

    # Impute Y0_hat for the whole dataset
    # Obs with exp/imp/year not seen in untreated sample will have
    # all-zero dummy cols → effectively mapped to reference level,
    # which is INCORRECT. Flag these.
    d = d.copy()
    d['_exp_in_sample'] = (
        d['exporter'].isin(untreated['exporter']) |
        (d['exporter'].astype(str) == str(exp_ref))
    )
    d['_imp_in_sample'] = (
        d['importer'].isin(untreated['importer']) |
        (d['importer'].astype(str) == str(imp_ref))
    )
    d['_yr_in_sample'] = (
        d['year'].isin(untreated['year']) |
        (d['year'] == yr_ref)
    )
    d['imp_valid'] = (
        d['_exp_in_sample'] & d['_imp_in_sample'] & d['_yr_in_sample']
    )

    X_all = build_design(d)
    d['y0_hat'] = ols_u.predict(X_all)
    d.loc[~d['imp_valid'], 'y0_hat'] = np.nan
    d['tau'] = d[outcome] - d['y0_hat']

    n_imputed = d['imp_valid'].sum()
    n_treated_imputed = ((d['in_event_window'] == 1) & d['imp_valid']).sum()
    print(f"  Counterfactual fit on {len(untreated):,} untreated obs "
          f"(R² = {ols_u.rsquared:.3f})")
    print(f"  Valid imputations: {n_imputed:,} obs "
          f"({n_treated_imputed:,} in event window)")

    return d


def aggregate_by_event_time(d):
    """
    BJS step 4: average tau_it by event time h.

    Returns DataFrame indexed by rel_year with columns:
      att_h    : point estimate for ATT(h)
      n_h      : number of treated obs at that horizon
    """
    treated_valid = d[(d['in_event_window'] == 1) & d['imp_valid']].copy()

    rows = []
    for h in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1):
        grp = treated_valid[treated_valid['rel_year'] == h]
        if len(grp) == 0:
            rows.append({'rel_year': h, 'att_h': np.nan, 'n_h': 0})
        else:
            rows.append({
                'rel_year': h,
                'att_h': grp['tau'].mean(),
                'n_h': len(grp),
            })
    return pd.DataFrame(rows).set_index('rel_year')


def overall_att(d):
    """Pooled ATT across all treated obs in event window (equal-weighted)."""
    treated_valid = d[(d['in_event_window'] == 1) & d['imp_valid']]
    if len(treated_valid) == 0:
        return np.nan, 0
    return treated_valid['tau'].mean(), len(treated_valid)


def pair_cluster_bootstrap(d, outcome, controls, n_boot, seed):
    """
    Pair-cluster bootstrap for BJS event study SEs.

    Draws pairs with replacement, stacks all year-obs for sampled pairs,
    and re-runs the full imputation procedure. Returns distributions
    of ATT(h) across bootstrap reps.
    """
    rng = np.random.default_rng(seed)
    pairs = d['pair_id'].unique()
    n_pairs = len(pairs)

    boot_att_h = {h: [] for h in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1)}
    boot_att_overall = []

    t0 = time.time()
    n_failed = 0

    for b in range(n_boot):
        if (b + 1) % 50 == 0 or b == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (b + 1) * (n_boot - b - 1) if b > 0 else 0
            print(f"    Bootstrap rep {b+1}/{n_boot} "
                  f"(elapsed: {elapsed:.0f}s, "
                  f"remaining: ~{remaining:.0f}s, "
                  f"failed: {n_failed})")

        # Draw pairs with replacement
        sample_pairs = rng.choice(pairs, size=n_pairs, replace=True)

        # Stack obs for sampled pairs (duplicates allowed)
        pair_to_obs = {p: d[d['pair_id'] == p] for p in pairs}
        boot_parts = [pair_to_obs[p] for p in sample_pairs]
        boot_df = pd.concat(boot_parts, ignore_index=True)

        try:
            boot_imputed = fit_counterfactual_and_impute(
                boot_df, outcome, controls)
            boot_event = aggregate_by_event_time(boot_imputed)
            for h in boot_att_h:
                boot_att_h[h].append(boot_event.loc[h, 'att_h'])
            att_o, _ = overall_att(boot_imputed)
            boot_att_overall.append(att_o)
        except Exception as e:
            n_failed += 1
            for h in boot_att_h:
                boot_att_h[h].append(np.nan)
            boot_att_overall.append(np.nan)

    elapsed = time.time() - t0
    print(f"    Bootstrap complete: {n_boot} reps in {elapsed:.0f}s "
          f"({n_failed} failed)")

    return boot_att_h, boot_att_overall


def summarize_bootstrap(point_event, point_overall,
                        boot_att_h, boot_att_overall):
    """Compute SEs and 95% percentile CIs from bootstrap draws."""
    rows = []
    for h in sorted(boot_att_h.keys()):
        boot_vals = np.array([v for v in boot_att_h[h] if not np.isnan(v)])
        if len(boot_vals) < 50:
            rows.append({
                'rel_year': h,
                'att_h': point_event.loc[h, 'att_h'],
                'se_h': np.nan,
                'ci_lo': np.nan,
                'ci_hi': np.nan,
                'n_treated': point_event.loc[h, 'n_h'],
                'n_boot': len(boot_vals),
            })
            continue
        rows.append({
            'rel_year': h,
            'att_h': point_event.loc[h, 'att_h'],
            'se_h': np.std(boot_vals, ddof=1),
            'ci_lo': np.percentile(boot_vals, 2.5),
            'ci_hi': np.percentile(boot_vals, 97.5),
            'n_treated': point_event.loc[h, 'n_h'],
            'n_boot': len(boot_vals),
        })

    event_summary = pd.DataFrame(rows).set_index('rel_year')

    boot_o = np.array([v for v in boot_att_overall if not np.isnan(v)])
    if len(boot_o) < 50:
        overall = {
            'att': point_overall,
            'se': np.nan,
            'ci_lo': np.nan,
            'ci_hi': np.nan,
            'n_boot': len(boot_o),
        }
    else:
        overall = {
            'att': point_overall,
            'se': np.std(boot_o, ddof=1),
            'ci_lo': np.percentile(boot_o, 2.5),
            'ci_hi': np.percentile(boot_o, 97.5),
            'n_boot': len(boot_o),
        }

    return event_summary, overall


def print_event_study(event_summary, overall, n_treated_total, label):
    """Formatted console output mirroring existing M9 style."""
    print(f"\n  {'Rel Year':>10} {'ATT(h)':>10} {'SE':>10} "
          f"{'95% CI':>22} {'Sig':>5} {'N':>6}")
    print(f"  {'-' * 68}")

    def stars_from_ci(coef, ci_lo, ci_hi, se):
        if np.isnan(se) or se == 0:
            return ""
        # Approximate z-test
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
        s = stars_from_ci(row['att_h'], row['ci_lo'],
                          row['ci_hi'], row['se_h'])
        ci_str = (f"[{row['ci_lo']:>7.3f},{row['ci_hi']:>8.3f}]"
                  if not np.isnan(row['ci_lo']) else "[ bootstrap failed ]")
        print(f"  {'t' + format(h, '+d'):>10} {row['att_h']:>10.4f} "
              f"{row['se_h']:>10.4f} {ci_str} {s:>5} {int(row['n_treated']):>6}")

    print(f"\n  {'-' * 68}")
    if not np.isnan(overall['se']):
        s = ("***" if abs(overall['att'] / overall['se']) > 2.58
             else "**" if abs(overall['att'] / overall['se']) > 1.96
             else "*" if abs(overall['att'] / overall['se']) > 1.645
             else "")
        print(f"  Pooled ATT: {overall['att']:.4f} "
              f"(SE = {overall['se']:.4f}) {s}")
        print(f"    95% CI: [{overall['ci_lo']:.4f}, {overall['ci_hi']:.4f}]")
        print(f"    Based on {n_treated_total:,} treated obs, "
              f"{overall['n_boot']} successful bootstrap reps")
    else:
        print(f"  Pooled ATT: {overall['att']:.4f} (bootstrap SE unavailable)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("BJS IMPUTATION ESTIMATOR — ROBUSTNESS CHECK FOR M9 EVENT STUDY")
    print("=" * 70)

    # Load incentive intro dates once
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

        print(f"Loaded {len(df_raw):,} rows from {os.path.basename(ds_cfg['file'])}")

        if ds_cfg['outcome'] not in df_raw.columns:
            print(f"  Outcome {ds_cfg['outcome']} not in data — skipping")
            continue

        # Drop any controls that aren't in this dataset (defensive)
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
                print(f"  Only {n_treated_window} treated obs in event window "
                      f"— skipping")
                continue

            print(f"  Untreated pool: {d['in_untreated_pool'].sum():,} obs")
            print(f"  Treated obs in event window: {n_treated_window:,}")

            try:
                d_imputed = fit_counterfactual_and_impute(
                    d, ds_cfg['outcome'], controls_present)
            except ValueError as e:
                print(f"  Counterfactual fit failed: {e} — skipping")
                continue

            point_event = aggregate_by_event_time(d_imputed)
            point_overall, n_treated_total = overall_att(d_imputed)

            print(f"\n  Running pair-cluster bootstrap "
                  f"({N_BOOTSTRAP} reps)...")
            boot_att_h, boot_att_overall = pair_cluster_bootstrap(
                d, ds_cfg['outcome'], controls_present,
                N_BOOTSTRAP, RANDOM_SEED)

            event_summary, overall = summarize_bootstrap(
                point_event,
                point_overall,
                boot_att_h,
                boot_att_overall,
            )

            print_event_study(
                event_summary, overall, n_treated_total,
                f"{ds_key} / {sv_key}")

            # Export to CSV for thesis
            result_key = f"{ds_key}__{sv_key}"
            outpath = os.path.join(
                OUTPUT_DIR, f'bjs_event_study_{result_key}.csv')
            export = event_summary.reset_index().copy()
            export['dataset'] = ds_key
            export['sample'] = sv_key
            export.to_csv(outpath, index=False)
            print(f"\n  Saved: {outpath}")

            all_results[result_key] = {
                'event_summary': event_summary,
                'overall': overall,
                'n_treated': n_treated_total,
            }

    # Combined summary table
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
                'spec': key,
                'att': o['att'],
                'se': o['se'],
                'ci_lo': o['ci_lo'],
                'ci_hi': o['ci_hi'],
                'n_treated': res['n_treated'],
            })
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'bjs_pooled_att_summary.csv'),
            index=False)
        print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'bjs_pooled_att_summary.csv')}")

    elapsed = time.time() - t_start
    print(f"\n\n{'=' * 70}")
    print(f"COMPLETE — total runtime: {elapsed / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()