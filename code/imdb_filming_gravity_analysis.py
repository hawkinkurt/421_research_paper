"""
imdb_filming_gravity_analysis.py
================================
Gravity model estimation using IMDb filming location data.
Loads pre-prepared datasets from data_fetch_clean.py Phase 8.

Models (all run for both film count and budget dep vars,
        AND for both raw EFW and income-adjusted EFWRESID):

  ─── PPML with Importer + Exporter + Year FE ───
  M1:  Baseline gravity (GDP, distance, remoteness)
  M2:  + cultural/historical proximity + RTA + EFW
  M3:  + incentive dummies (exporter & importer)
  M4:  + incentive type dummies (replacing exporter incentive)
  M5:  + generosity (exporter & importer, replacing incentive dummies)
  M6:  Cluster: M3 + years_active
  M7:  Cluster: early vs late adopters (replacing exporter incentive)

  ─── OLS with Country-Pair + Year FE (within-transformation) ───
  M8:  Incentive dummies (mirrors M3)
  M9:  Event study (relative-time dummies replacing exporter incentive)
  M10: Cluster: M8 + years_active
  M11: Incentive type dummies (mirrors M4, pair FE)

  Pair FE models (M8, M9, M10, M11) each run on:
    (a) Full sample
    (b) Excluding Anglo-to-Anglo pairs
    (c) Excluding all Anglo pairs
    (d) Without EFW controls

NOTE: Incentive type typology follows the Olsberg SPI Global Incentives
Index 2025, which classifies schemes into three categories: rebate,
tax_credit, and tax_shelter. The is_tax_credit category encompasses what
were previously separately coded as refundable, transferable, and
non-refundable (standard) credits. Since there is substantial collinearity
between the three-way split with only ~6 tax credit observations and
minimal within-sample variation across subtypes, the unified tax_credit
category is used for estimation. A tax_credit_subtype column in the
incentive dataset preserves the finer distinction for descriptive use.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'

ANALYSIS_FILE = f'{PROJECT_DIR}\\data\\processed\\imdb\\imdb_gravity_analysis.csv'
MERGED_FILE = f'{PROJECT_DIR}\\data\\processed\\imdb\\imdb_gravity_merged.csv'
INCENTIVE_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\film_incentive_intro_dates.csv'

ANGLO = {'US', 'GB', 'CA', 'AU', 'NZ'}

EVENT_WINDOW_MIN = -5
EVENT_WINDOW_MAX = 10
MAX_YEARS_ACTIVE = 20
EARLY_CUTOFF = 2005

# Dependent variable mappings: PPML uses levels, OLS pair FE uses logs
SPECS = {
    'count': {'ppml_dep': 'num_films', 'ols_dep': 'log_num_films', 'label': 'FILM COUNT'},
    'budget': {'ppml_dep': 'total_budget', 'ols_dep': 'log_budget', 'label': 'BUDGET'},
}


# =============================================================================
# PHASE 1: LOAD DATA
# =============================================================================

print("=" * 70)
print("PHASE 1: LOAD PRE-PREPARED DATA")
print("=" * 70)

df = pd.read_csv(ANALYSIS_FILE)
merged = pd.read_csv(MERGED_FILE)

print(f"Analysis dataset: {len(df):,} rows")
print(f"Merged dataset (for no-EFW specs): {len(merged):,} rows")
print(f"Exporters: {df['exporter'].nunique()}, Importers: {df['importer'].nunique()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")

incentives = pd.read_csv(INCENTIVE_FILE)
incentive_dict = dict(zip(
    incentives['country_iso2'].astype(str),
    incentives['incentive_intro_year']
))

has_generosity = 'generosity_exp' in df.columns and df['generosity_exp'].nunique() > 1
print(f"Generosity exporter data available: {has_generosity}")

has_generosity_imp = 'generosity_imp' in df.columns and df['generosity_imp'].nunique() > 1
print(f"Generosity importer data available: {has_generosity_imp}")

has_budget = 'total_budget' in df.columns and df['total_budget'].notna().sum() >= 50
print(f"Budget data available: {has_budget} "
      f"({df['total_budget'].notna().sum() if 'total_budget' in df.columns else 0} obs)")

has_efwresid = ('efwresid_exporter' in df.columns
                and df['efwresid_exporter'].notna().sum() > 50)
print(f"EFWRESID data available: {has_efwresid}")
if has_efwresid:
    print(f"  efwresid_exporter coverage: {df['efwresid_exporter'].notna().sum():,}")
    print(f"  efwresid_importer coverage: {df['efwresid_importer'].notna().sum():,}")

# Type dummy variables — Olsberg three-category typology (rebate, tax_credit, tax_shelter)
type_dummy_vars = [f'{d}_exp' for d in
                   ['is_rebate', 'is_tax_credit', 'is_tax_shelter']]
type_vars_available = [v for v in type_dummy_vars
                       if v in df.columns and df[v].nunique() > 1]
print(f"Incentive type dummies available: {type_vars_available}")

# Pre-compute derived variables on all dataframes
print("\nPre-computing derived variables...")
for d_label, d in [('Analysis', df), ('Merged', merged)]:
    d['intro_year_exp'] = d['exporter'].map(incentive_dict)
    d['years_active_exp'] = np.where(
        (d['incentive_exporter'] == 1) & d['intro_year_exp'].notna(),
        np.minimum(d['year'] - d['intro_year_exp'], MAX_YEARS_ACTIVE), 0
    )
    d['early_adopter_exp'] = d.apply(
        lambda row: (1 if (row['exporter'] in incentive_dict
                           and pd.notna(incentive_dict[row['exporter']])
                           and incentive_dict[row['exporter']] < EARLY_CUTOFF
                           and row['year'] >= incentive_dict[row['exporter']])
                     else 0), axis=1)
    d['late_adopter_exp'] = d.apply(
        lambda row: (1 if (row['exporter'] in incentive_dict
                           and pd.notna(incentive_dict[row['exporter']])
                           and incentive_dict[row['exporter']] >= EARLY_CUTOFF
                           and row['year'] >= incentive_dict[row['exporter']])
                     else 0), axis=1)
    print(f"  {d_label}: years_active>0: {(d['years_active_exp'] > 0).sum():,}, "
          f"early: {d['early_adopter_exp'].sum():,}, "
          f"late: {d['late_adopter_exp'].sum():,}")


# =============================================================================
# HELPERS
# =============================================================================

def fit_ppml(formula, data, label=""):
    """Fit PPML (GLM Poisson) with pair-clustered standard errors."""
    try:
        # Create pair identifier for clustering
        pair_ids = data['exporter'].astype(str) + '_' + data['importer'].astype(str)
        m = smf.glm(formula, data=data,
                     family=sm.families.Poisson()).fit(
            maxiter=100,
            cov_type='cluster',
            cov_kwds={'groups': pair_ids}
        )
        return m
    except Exception as e:
        print(f"  PPML failed{' (' + label + ')' if label else ''}: {e}")
        return None


def pseudo_r2(m):
    """McFadden pseudo R-squared."""
    if m is None:
        return np.nan
    return 1 - m.deviance / m.null_deviance


def print_model_results(m, var_list, label=""):
    """Print key coefficients from a fitted model."""
    if m is None:
        print(f"  {label}: model not estimated")
        return
    print(f"\n{'Variable':<30} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5} {'% effect':>10}")
    print("-" * 75)
    for var in var_list:
        # Check both raw and demeaned versions
        lookup = var if var in m.params else f'{var}_dm'
        if lookup in m.params:
            coef = m.params[lookup]
            se = m.bse[lookup]
            p = m.pvalues[lookup]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            pct = (np.exp(coef) - 1) * 100
            print(f"{var:<30} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5} {pct:>+9.1f}%")
        else:
            print(f"{var:<30} {'--':>10}")


def fmt_coef(model, var_name):
    """Format a coefficient for summary tables."""
    if model is None:
        return f"{'--':>14}"
    for lookup in [var_name, f'{var_name}_dm']:
        if lookup in model.params:
            coef = model.params[lookup]
            p = model.pvalues[lookup]
            stars = ("***" if p < 0.01 else "**" if p < 0.05
                     else "*" if p < 0.1 else "")
            pct = (np.exp(coef) - 1) * 100
            return f"{coef:>6.3f}{stars:<3}({pct:>+.0f}%)"
    return f"{'--':>14}"


def run_pair_fe(df_sub, dep_var, tv, label):
    """
    Run OLS pair FE with within-transformation.
    Returns the fitted model or None.
    """
    df_sub = df_sub.copy()
    df_sub['pair_id'] = df_sub['exporter'] + '_' + df_sub['importer']

    # Drop singleton pairs
    pc = df_sub.groupby('pair_id').size()
    df_panel = df_sub[df_sub['pair_id'].isin(pc[pc > 1].index)].copy()

    # Filter to valid dependent variable
    df_panel = df_panel[df_panel[dep_var].notna()].copy()

    print(f"\n{'='*70}")
    print(f"PAIR FE (OLS): {label}")
    print(f"{'='*70}")
    print(f"Observations: {len(df_panel):,}, Pairs: {df_panel['pair_id'].nunique()}")

    if len(df_panel) < 50:
        print(f"  SKIPPED - too few observations ({len(df_panel)})")
        return None

    # Report within-pair variation in incentive variables
    for inc_var in ['incentive_exporter', 'incentive_importer']:
        if inc_var in tv:
            iv = df_panel.groupby('pair_id')[inc_var].nunique()
            n_varies = (iv > 1).sum()
            print(f"  {inc_var} varies within pair: "
                  f"{n_varies} ({n_varies / len(iv) * 100:.1f}%)")

    # Within-transformation (demean by pair)
    demean_vars = [dep_var] + tv
    df_dm = df_panel.copy()
    pm = df_panel.groupby('pair_id')[demean_vars].transform('mean')
    for var in demean_vars:
        df_dm[f'{var}_dm'] = df_panel[var] - pm[var]

    # OLS on demeaned data with year dummies
    dm_formula = (f'{dep_var}_dm ~ '
                  + ' + '.join([f'{v}_dm' for v in tv])
                  + ' + C(year) - 1')
    m = smf.ols(dm_formula, data=df_dm).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_dm['pair_id']}
    )

    # Within R-squared
    ss_res = np.sum(m.resid ** 2)
    ss_tot = np.sum(df_dm[f'{dep_var}_dm'] ** 2)
    within_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"Within R²: {within_r2:.4f}, N: {int(m.nobs)}")
    print_model_results(m, tv, label)

    return m


def run_event_study(df_src, log_dep_var, tv_controls, label_str):
    """Run OLS pair FE event study with incentive_importer control."""
    print(f"\n{'='*70}")
    print(f"EVENT STUDY: {label_str}")
    print(f"{'='*70}")

    # Restrict to exporters with known incentive intro dates
    df_ev = df_src[df_src['exporter'].map(incentive_dict).notna()].copy()
    df_ev['intro_year_exp'] = df_ev['exporter'].map(incentive_dict)
    df_ev['rel_year'] = (df_ev['year'] - df_ev['intro_year_exp']).astype(int)
    df_ev = df_ev[(df_ev['rel_year'] >= EVENT_WINDOW_MIN) &
                  (df_ev['rel_year'] <= EVENT_WINDOW_MAX)].copy()

    if log_dep_var not in df_ev.columns:
        print(f"  SKIPPED - {log_dep_var} not in data")
        return None

    df_ev = df_ev[df_ev[log_dep_var].notna()].copy()
    print(f"Sample: {len(df_ev):,} obs, "
          f"{df_ev['exporter'].nunique()} exporters with incentive")

    if len(df_ev) < 50:
        print(f"  SKIPPED - too few observations")
        return None

    # Create relative-time dummies (t-1 is omitted reference)
    for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1):
        if t != -1:
            df_ev[f'rel_t{t}'] = (df_ev['rel_year'] == t).astype(int)

    # Pair FE setup — drop singletons
    df_ev['pair_id'] = df_ev['exporter'] + '_' + df_ev['importer']
    pc = df_ev.groupby('pair_id').size()
    df_ev = df_ev[df_ev['pair_id'].isin(pc[pc > 1].index)].copy()
    print(f"After dropping singletons: {len(df_ev):,} obs, "
          f"{df_ev['pair_id'].nunique()} pairs")

    if len(df_ev) < 50:
        print(f"  SKIPPED - too few observations after dropping singletons")
        return None

    # Controls: time-varying gravity vars + importer incentive
    # Remove exporter incentive (replaced by event dummies)
    controls = [v for v in tv_controls
                if v != 'incentive_exporter']
    # Ensure incentive_importer is included
    if 'incentive_importer' not in controls:
        controls.append('incentive_importer')

    rel_vars = [f'rel_t{t}'
                for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1)
                if t != -1]

    # Within-transformation
    all_ev_vars = [log_dep_var] + controls + rel_vars
    df_dm = df_ev.copy()
    pm = df_ev.groupby('pair_id')[all_ev_vars].transform('mean')
    for var in all_ev_vars:
        df_dm[f'{var}_dm'] = df_ev[var] - pm[var]

    # Year dummies
    year_dummies = pd.get_dummies(df_dm['year'], prefix='yr',
                                  drop_first=True, dtype=float)
    X = pd.concat([df_dm[[f'{v}_dm' for v in controls + rel_vars]],
                    year_dummies], axis=1)
    y = df_dm[f'{log_dep_var}_dm']

    m_ev = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_dm['pair_id']}
    )

    # Print event study coefficients
    print(f"\n{'Rel Year':>10} {'Coef':>10} {'SE':>10} "
          f"{'95% CI':>22} {'Sig':>5}")
    print("-" * 60)

    event_coefs = []
    for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1):
        if t == -1:
            event_coefs.append((t, 0, 0, 0, 0))
            print(f"{'t-1':>10} {'0.000':>10} {'(ref)':>10}")
            continue
        dm_var = f'rel_t{t}_dm'
        if dm_var in m_ev.params:
            coef = m_ev.params[dm_var]
            se = m_ev.bse[dm_var]
            p = m_ev.pvalues[dm_var]
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se
            stars = ("***" if p < 0.01 else "**" if p < 0.05
                     else "*" if p < 0.1 else "")
            event_coefs.append((t, coef, se, ci_lo, ci_hi))
            print(f"{'t' + format(t, '+d'):>10} {coef:>10.4f} "
                  f"{se:>10.4f} [{ci_lo:>8.4f}, {ci_hi:>8.4f}] "
                  f"{stars:>5}")

    # Print importer incentive coefficient
    imp_dm = 'incentive_importer_dm'
    if imp_dm in m_ev.params:
        coef = m_ev.params[imp_dm]
        se = m_ev.bse[imp_dm]
        p = m_ev.pvalues[imp_dm]
        stars = ("***" if p < 0.01 else "**" if p < 0.05
                 else "*" if p < 0.1 else "")
        print(f"\n  incentive_importer: {coef:.4f} (SE={se:.4f}, p={p:.4f}) {stars}")

    # Summary statistics
    pre_coefs = [c for t, c, s, lo, hi in event_coefs
                 if EVENT_WINDOW_MIN <= t < -1]
    post_coefs = [c for t, c, s, lo, hi in event_coefs if t >= 0]
    if pre_coefs and post_coefs:
        print(f"\n  Avg pre-treatment (t-5 to t-2): "
              f"{np.mean(pre_coefs):.4f}")
        print(f"  Avg post-treatment (t0 to t+10): "
              f"{np.mean(post_coefs):.4f}")

    return m_ev


# =============================================================================
# EFW VARIANT DEFINITIONS
# =============================================================================

EFW_VARIANTS = {
    'EFW': {
        'exp': 'efw_exporter',
        'imp': 'efw_importer',
        'label': 'Raw EFW',
    },
}

if has_efwresid:
    EFW_VARIANTS['EFWRESID'] = {
        'exp': 'efwresid_exporter',
        'imp': 'efwresid_importer',
        'label': 'Income-adjusted EFWRESID',
    }

print(f"\nEFW variants to run: {list(EFW_VARIANTS.keys())}")
print(f"Total estimation rounds: {len(EFW_VARIANTS)} EFW variants "
      f"x {len([s for s in SPECS if s != 'budget' or has_budget])} dep vars "
      f"= {len(EFW_VARIANTS) * len([s for s in SPECS if s != 'budget' or has_budget])} rounds")


# =============================================================================
# MAIN ESTIMATION LOOP
# =============================================================================

all_results = {}

for efw_key, efw_cfg in EFW_VARIANTS.items():
    efw_exp = efw_cfg['exp']
    efw_imp = efw_cfg['imp']
    efw_label = efw_cfg['label']

    # ─── Build formula components for this EFW variant ───
    FE_TERMS = 'C(importer) + C(exporter) + C(year)'

    BASELINE_VARS = ('log_gdp_importer + log_gdp_exporter + log_dist + '
                     'remoteness_importer + remoteness_exporter')

    CULTURAL_VARS = (f'contig + comlang_off + col45 + rta + '
                     f'{efw_exp} + {efw_imp}')

    FULL_GRAVITY = f'{BASELINE_VARS} + {CULTURAL_VARS}'

    TV_FULL = [
        'log_gdp_importer', 'log_gdp_exporter',
        'remoteness_importer', 'remoteness_exporter',
        efw_exp, efw_imp,
        'rta', 'incentive_exporter', 'incentive_importer'
    ]

    TV_NO_EFW = [v for v in TV_FULL if v not in [efw_exp, efw_imp]]

    # ─── Loop over dependent variables ───
    for spec_key, spec in SPECS.items():
        ppml_dep = spec['ppml_dep']
        ols_dep = spec['ols_dep']
        label = spec['label']

        if spec_key == 'budget' and not has_budget:
            print(f"\n{'#'*70}")
            print(f"# SKIPPING {label} / {efw_label} - insufficient budget data")
            print(f"{'#'*70}")
            continue

        result_key = (spec_key, efw_key)
        tag = f"{label} [{efw_label}]"

        print(f"\n\n{'#'*70}")
        print(f"# {'='*66} #")
        print(f"#   DEP VAR: {label:>20}  |  EFW: {efw_label:<24} #")
        print(f"# {'='*66} #")
        print(f"{'#'*70}")

        spec_results = {}

        # ─────────────────────────────────────────────────────────────
        # MODEL 1: Baseline gravity (PPML, Imp+Exp+Year FE)
        # ─────────────────────────────────────────────────────────────

        print(f"\n{'='*70}")
        print(f"MODEL 1: Baseline Gravity (PPML) - {tag}")
        print(f"{'='*70}")

        f1 = f'{ppml_dep} ~ {BASELINE_VARS} + {FE_TERMS}'
        m1 = fit_ppml(f1, df, "M1")
        if m1:
            print(f"Pseudo-R²: {pseudo_r2(m1):.4f}, N: {int(m1.nobs)}")
            print_model_results(m1, [
                'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
                'remoteness_importer', 'remoteness_exporter'
            ], "M1")
        spec_results['M1'] = m1

        # ─────────────────────────────────────────────────────────────
        # MODEL 2: + Cultural/historical + RTA + EFW (PPML)
        # ─────────────────────────────────────────────────────────────

        print(f"\n{'='*70}")
        print(f"MODEL 2: Full Gravity (PPML) - {tag}")
        print(f"{'='*70}")

        f2 = f'{ppml_dep} ~ {FULL_GRAVITY} + {FE_TERMS}'
        m2 = fit_ppml(f2, df, "M2")
        if m2:
            print(f"Pseudo-R²: {pseudo_r2(m2):.4f}, N: {int(m2.nobs)}")
            print_model_results(m2, [
                'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
                'remoteness_importer', 'remoteness_exporter',
                'contig', 'comlang_off', 'col45', 'rta',
                efw_exp, efw_imp
            ], "M2")
        spec_results['M2'] = m2

        # ─────────────────────────────────────────────────────────────
        # MODEL 3: + Incentive dummies (PPML)
        # ─────────────────────────────────────────────────────────────

        print(f"\n{'='*70}")
        print(f"MODEL 3: Incentive Dummies (PPML) - {tag}")
        print(f"{'='*70}")

        f3 = (f'{ppml_dep} ~ {FULL_GRAVITY} + '
              f'incentive_exporter + incentive_importer + {FE_TERMS}')
        m3 = fit_ppml(f3, df, "M3")
        if m3:
            print(f"Pseudo-R²: {pseudo_r2(m3):.4f}, N: {int(m3.nobs)}")
            print_model_results(m3, [
                'log_dist', 'contig', 'comlang_off', 'col45', 'rta',
                efw_exp, efw_imp,
                'incentive_exporter', 'incentive_importer'
            ], "M3")
        spec_results['M3'] = m3

        # ─────────────────────────────────────────────────────────────
        # MODEL 4: Incentive type dummies (PPML)
        # Olsberg typology: rebate, tax_credit, tax_shelter
        # Note: to avoid the dummy variable trap, one category must be
        # omitted as the reference. We omit is_rebate_exp since rebates
        # are the modal category (~48 of 57 active-incentive countries).
        # The tax_credit and tax_shelter coefficients therefore represent
        # the effect relative to the rebate baseline.
        # ─────────────────────────────────────────────────────────────

        print(f"\n{'='*70}")
        print(f"MODEL 4: Incentive Types (PPML) - {tag}")
        print(f"{'='*70}")

        m4 = None
        if type_vars_available:
            # Omit is_rebate_exp as reference category
            type_vars_m4 = [v for v in type_vars_available
                            if v != 'is_rebate_exp']
            if type_vars_m4:
                type_str = ' + '.join(type_vars_m4)
                f4 = (f'{ppml_dep} ~ {FULL_GRAVITY} + '
                      f'incentive_exporter + {type_str} + incentive_importer + '
                      f'{FE_TERMS}')
                m4 = fit_ppml(f4, df, "M4")
                if m4:
                    print(f"Pseudo-R²: {pseudo_r2(m4):.4f}, N: {int(m4.nobs)}")
                    print(f"Reference category: rebate (is_rebate_exp omitted)")
                    print_model_results(m4, [
                        'log_dist', 'comlang_off',
                        'incentive_exporter', 'incentive_importer'
                    ] + type_vars_m4, "M4")
            else:
                print("  SKIPPED - no non-reference type variables with variation")
        else:
            print("  SKIPPED - no incentive type variables with variation")
        spec_results['M4'] = m4

        # ─────────────────────────────────────────────────────────────
        # MODEL 5: Generosity (PPML)
        # ─────────────────────────────────────────────────────────────

        print(f"\n{'='*70}")
        print(f"MODEL 5: Generosity (PPML) - {tag}")
        print(f"{'='*70}")

        m5 = None
        if has_generosity:
            gen_imp_term = (' + generosity_imp' if has_generosity_imp
                            else ' + incentive_importer')
            f5 = (f'{ppml_dep} ~ {FULL_GRAVITY} + '
                  f'generosity_exp{gen_imp_term} + {FE_TERMS}')
            m5 = fit_ppml(f5, df, "M5")
            if m5:
                print(f"Pseudo-R²: {pseudo_r2(m5):.4f}, N: {int(m5.nobs)}")
                gen_vars = ['generosity_exp']
                gen_vars.append('generosity_imp' if has_generosity_imp
                                else 'incentive_importer')
                print_model_results(m5, gen_vars, "M5")

                # Diagnostic: generosity ranges
                active_gen = df[df['generosity_exp'] > 0]['generosity_exp']
                if len(active_gen) > 0:
                    print(f"\n  Generosity exp (active): mean={active_gen.mean()*100:.1f}%, "
                          f"min={active_gen.min()*100:.1f}%, max={active_gen.max()*100:.1f}%")
        else:
            print("  SKIPPED - no generosity data available")
        spec_results['M5'] = m5

        # ─────────────────────────────────────────────────────────────
        # MODEL 6: Cluster - years active (PPML)
        # ─────────────────────────────────────────────────────────────

        print(f"\n{'='*70}")
        print(f"MODEL 6: Cluster - Years Active (PPML) - {tag}")
        print(f"{'='*70}")

        # Diagnostic: years_active distribution
        active = df[df['years_active_exp'] > 0]['years_active_exp']
        if len(active) > 0:
            print(f"Years active (when >0): mean={active.mean():.1f}, "
                  f"median={active.median():.1f}, max={active.max():.0f}")

        f6 = (f'{ppml_dep} ~ {FULL_GRAVITY} + '
              f'incentive_exporter + years_active_exp + incentive_importer + '
              f'{FE_TERMS}')
        m6 = fit_ppml(f6, df, "M6")
        if m6:
            print(f"Pseudo-R²: {pseudo_r2(m6):.4f}, N: {int(m6.nobs)}")
            print_model_results(m6, [
                'incentive_exporter', 'years_active_exp', 'incentive_importer'
            ], "M6")
        spec_results['M6'] = m6

        # ─────────────────────────────────────────────────────────────
        # MODEL 7: Cluster - early vs late adopters (PPML)
        # ─────────────────────────────────────────────────────────────

        print(f"\n{'='*70}")
        print(f"MODEL 7: Cluster - Early vs Late (PPML) - {tag}")
        print(f"{'='*70}")

        print(f"Early adopter obs: {df['early_adopter_exp'].sum():,}")
        print(f"Late adopter obs:  {df['late_adopter_exp'].sum():,}")

        f7 = (f'{ppml_dep} ~ {FULL_GRAVITY} + '
              f'early_adopter_exp + late_adopter_exp + incentive_importer + '
              f'{FE_TERMS}')
        m7 = fit_ppml(f7, df, "M7")
        if m7:
            print(f"Pseudo-R²: {pseudo_r2(m7):.4f}, N: {int(m7.nobs)}")
            print_model_results(m7, [
                'early_adopter_exp', 'late_adopter_exp', 'incentive_importer'
            ], "M7")
        spec_results['M7'] = m7

        # ─────────────────────────────────────────────────────────────
        # PREPARE SUBSAMPLES FOR PAIR FE MODELS (M8, M9, M10, M11)
        # ─────────────────────────────────────────────────────────────

        print(f"\n--- Preparing subsamples for pair FE models ---")

        # (a) Full sample = df
        # (b) Excluding Anglo-to-Anglo pairs
        df_excl_aa = df[~((df['exporter'].isin(ANGLO)) &
                           (df['importer'].isin(ANGLO)))].copy()
        # (c) Excluding all Anglo pairs
        df_excl_anglo = df[~((df['exporter'].isin(ANGLO)) |
                              (df['importer'].isin(ANGLO)))].copy()
        # (d) Without EFW — use merged dataset with broader coverage
        est_vars_no_efw = [ols_dep, 'log_gdp_importer', 'log_gdp_exporter',
                           'log_dist', 'contig', 'comlang_off', 'col45',
                           'rta', 'remoteness_importer', 'remoteness_exporter',
                           'incentive_exporter', 'incentive_importer']
        df_no_efw = merged.dropna(subset=est_vars_no_efw).copy()

        print(f"  Full sample:         {len(df):,}")
        print(f"  Excl Anglo-to-Anglo: {len(df_excl_aa):,}")
        print(f"  Excl all Anglo:      {len(df_excl_anglo):,}")
        print(f"  No EFW (merged):     {len(df_no_efw):,}")

        subsamples = [
            ('Full sample', df, TV_FULL),
            ('Excl Anglo-to-Anglo', df_excl_aa, TV_FULL),
            ('Excl all Anglo', df_excl_anglo, TV_FULL),
            ('No EFW', df_no_efw, TV_NO_EFW),
        ]

        # ─────────────────────────────────────────────────────────────
        # MODEL 8: Incentive dummies, pair FE (OLS) — all subsamples
        # ─────────────────────────────────────────────────────────────

        print(f"\n\n{'#'*70}")
        print(f"# MODEL 8: Incentive Dummies - Pair FE (OLS) - {tag}")
        print(f"{'#'*70}")

        m8_results = {}
        for ss_name, ss_df, ss_tv in subsamples:
            m8_results[ss_name] = run_pair_fe(
                ss_df, ols_dep, ss_tv,
                f'M8 {tag} - {ss_name}')
        spec_results['M8'] = m8_results

        # ─────────────────────────────────────────────────────────────
        # MODEL 9: Event study, pair FE (OLS) — all subsamples
        # ─────────────────────────────────────────────────────────────

        print(f"\n\n{'#'*70}")
        print(f"# MODEL 9: Event Study - Pair FE (OLS) - {tag}")
        print(f"{'#'*70}")

        m9_results = {}
        for ss_name, ss_df, ss_tv in subsamples:
            m9_results[ss_name] = run_event_study(
                ss_df, ols_dep, ss_tv,
                f'M9 {tag} - {ss_name}')
        spec_results['M9'] = m9_results

        # ─────────────────────────────────────────────────────────────
        # MODEL 10: Cluster - years active, pair FE (OLS) — all subsamples
        # ─────────────────────────────────────────────────────────────

        print(f"\n\n{'#'*70}")
        print(f"# MODEL 10: Cluster Years Active - Pair FE (OLS) - {tag}")
        print(f"{'#'*70}")

        TV_FULL_CLUSTER = TV_FULL + ['years_active_exp']
        TV_NO_EFW_CLUSTER = TV_NO_EFW + ['years_active_exp']

        subsamples_m10 = [
            ('Full sample', df, TV_FULL_CLUSTER),
            ('Excl Anglo-to-Anglo', df_excl_aa, TV_FULL_CLUSTER),
            ('Excl all Anglo', df_excl_anglo, TV_FULL_CLUSTER),
            ('No EFW', df_no_efw, TV_NO_EFW_CLUSTER),
        ]

        m10_results = {}
        for ss_name, ss_df, ss_tv in subsamples_m10:
            m10_results[ss_name] = run_pair_fe(
                ss_df, ols_dep, ss_tv,
                f'M10 {tag} - {ss_name}')
        spec_results['M10'] = m10_results

        # ─────────────────────────────────────────────────────────────
        # MODEL 11: Incentive Types, pair FE (OLS) — all subsamples
        # Rebate is the omitted reference category (see M4 note).
        # ─────────────────────────────────────────────────────────────

        print(f"\n\n{'#'*70}")
        print(f"# MODEL 11: Incentive Types - Pair FE (OLS) - {tag}")
        print(f"{'#'*70}")

        m11_results = {}
        if type_vars_available:
            # Omit is_rebate_exp as reference (same as M4)
            type_vars_m11 = [v for v in type_vars_available
                             if v != 'is_rebate_exp']

            if type_vars_m11:
                # Keep incentive_exporter + tax_credit + tax_shelter type dummies
                # (rebate is the reference when incentive_exporter=1 and both
                # non-rebate type dummies = 0)
                TV_FULL_TYPES = TV_FULL + type_vars_m11
                TV_NO_EFW_TYPES = TV_NO_EFW + type_vars_m11

                subsamples_m11 = [
                    ('Full sample', df, TV_FULL_TYPES),
                    ('Excl Anglo-to-Anglo', df_excl_aa, TV_FULL_TYPES),
                    ('Excl all Anglo', df_excl_anglo, TV_FULL_TYPES),
                    ('No EFW', df_no_efw, TV_NO_EFW_TYPES),
                ]

                for ss_name, ss_df, ss_tv in subsamples_m11:
                    m11_results[ss_name] = run_pair_fe(
                        ss_df, ols_dep, ss_tv,
                        f'M11 {tag} - {ss_name}')
            else:
                print("  SKIPPED - no non-reference type variables with variation")
        else:
            print("  SKIPPED - no incentive type variables with variation")
        spec_results['M11'] = m11_results

        # ─────────────────────────────────────────────────────────────
        # SUMMARY TABLE FOR THIS SPEC+EFW COMBINATION
        # ─────────────────────────────────────────────────────────────

        print(f"\n\n{'='*70}")
        print(f"SUMMARY: INCENTIVE EFFECTS - {tag}")
        print(f"{'='*70}")

        # --- Overall incentive effects ---
        print(f"\n--- OVERALL INCENTIVE ---")
        print(f"\n{'Specification':<40} {'inc_exp':>15} {'inc_imp':>15}")
        print("-" * 72)

        for mname in ['M3', 'M6']:
            m = spec_results.get(mname)
            if m:
                print(f"{mname + ' (PPML)':<40}"
                      f"{fmt_coef(m, 'incentive_exporter'):>15}"
                      f"{fmt_coef(m, 'incentive_importer'):>15}")

        for mname in ['M8', 'M10']:
            mr = spec_results.get(mname, {})
            for ss_name, model in mr.items():
                if model is None:
                    continue
                lab = f"{mname} Pair FE ({ss_name})"[:40]
                print(f"{lab:<40}"
                      f"{fmt_coef(model, 'incentive_exporter'):>15}"
                      f"{fmt_coef(model, 'incentive_importer'):>15}")

        # --- Incentive types (Olsberg three-category typology) ---
        print(f"\n--- INCENTIVE TYPES (reference: rebate) ---")
        type_display = ['incentive_exporter',
                        'is_tax_credit_exp',
                        'is_tax_shelter_exp']
        print(f"\n{'Specification':<25}", end="")
        for tv in type_display:
            short = tv.replace('_exp', '').replace('is_', '').replace('_', ' ')
            print(f"{short:>20}", end="")
        print()
        print("-" * (25 + 20 * len(type_display)))

        m4_model = spec_results.get('M4')
        if m4_model:
            row = f"{'M4 (PPML)':<25}"
            for tv in type_display:
                row += fmt_coef(m4_model, tv).rjust(20)
            print(row)

        # M11 Pair FE type results
        m11_mr = spec_results.get('M11', {})
        for ss_name, model in m11_mr.items():
            if model is None:
                continue
            lab = f"M11 PFE ({ss_name})"[:25]
            row = f"{lab:<25}"
            for tv in type_display:
                row += fmt_coef(model, tv).rjust(20)
            print(row)

        # --- Early vs late ---
        print(f"\n--- EARLY vs LATE ADOPTERS ---")
        print(f"{'Specification':<25} {'early':>18} {'late':>18} {'imp':>18}")
        print("-" * 79)
        m7_model = spec_results.get('M7')
        if m7_model:
            print(f"{'M7 (PPML)':<25}"
                  f"{fmt_coef(m7_model, 'early_adopter_exp'):>18}"
                  f"{fmt_coef(m7_model, 'late_adopter_exp'):>18}"
                  f"{fmt_coef(m7_model, 'incentive_importer'):>18}")

        # --- Generosity ---
        print(f"\n--- GENEROSITY ---")
        m5_model = spec_results.get('M5')
        if m5_model:
            gen_imp_var = ('generosity_imp' if has_generosity_imp
                           else 'incentive_importer')
            print(f"{'M5 (PPML)':<25}"
                  f"{fmt_coef(m5_model, 'generosity_exp'):>18}"
                  f"{fmt_coef(m5_model, gen_imp_var):>18}")

        print(f"\n* p<0.1, ** p<0.05, *** p<0.01")
        print(f"PPML: estimated on levels, coefficients = semi-elasticities")
        print(f"OLS pair FE: within-transformation on log dep var")

        all_results[result_key] = spec_results


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")

for (spec_key, efw_key), sr in all_results.items():
    label = SPECS[spec_key]['label']
    efw_label = EFW_VARIANTS[efw_key]['label']
    n_estimated = sum(1 for k, v in sr.items()
                      if v is not None and not isinstance(v, dict))
    n_pair_fe = sum(1 for k, v in sr.items()
                    if isinstance(v, dict)
                    for _, m in v.items() if m is not None)
    print(f"  {label} [{efw_label}]: {n_estimated} PPML models, "
          f"{n_pair_fe} pair FE specifications estimated")