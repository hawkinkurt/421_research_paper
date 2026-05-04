"""
batis_gravity_analysis.py
=========================
Gravity model estimation using BaTIS SK1 (audiovisual services) data.
Loads pre-prepared datasets from data_fetch_clean.py Phase 7.

Performance-optimised version:
  - pyfixest for FE absorption (fepois / feols) instead of statsmodels
    dummy-variable expansion → orders-of-magnitude faster on PPML
  - Vectorised derived-variable construction (no .apply())
  - Pair IDs computed once and reused
  - Year dummies pre-built where needed

Models (run for both raw EFW and income-adjusted EFWRESID):

  ─── PPML with Importer + Exporter + Year FE (absorbed) ───
  M1:  Baseline gravity (GDP, distance, remoteness)
  M2:  + cultural/historical proximity + RTA + EFW
  M3:  + incentive dummies (exporter & importer)
  M4:  + incentive type dummies (replacing exporter incentive)
  M5:  + generosity (exporter & importer, replacing incentive dummies)
  M6:  Cluster: M3 + years_active
  M7:  Cluster: early vs late adopters (replacing exporter incentive)

  ─── OLS with Country-Pair + Year FE (absorbed) ───
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
tax_credit, and tax_shelter. Rebate is used as the omitted reference
category in type-heterogeneity specifications (M4, M11) since it is the
modal category and allows tax_credit and tax_shelter effects to be
interpreted relative to the rebate baseline.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'

ANALYSIS_FILE = f'{PROJECT_DIR}\\data\\processed\\batis_sk1_gravity_analysis.csv'
MERGED_FILE = f'{PROJECT_DIR}\\data\\processed\\batis_sk1_gravity_merged.csv'
INCENTIVE_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\film_incentive_intro_dates.csv'

ANGLO = {'US', 'GB', 'CA', 'AU', 'NZ'}

EVENT_WINDOW_MIN = -5
EVENT_WINDOW_MAX = 10
MAX_YEARS_ACTIVE = 20
EARLY_CUTOFF = 2005

# BaTIS dependent variable: PPML uses level, OLS pair FE uses log
PPML_DEP = 'trade_value'
OLS_DEP = 'log_trade'

t_start = time.time()

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
print(f"Exporters: {df['exporter'].nunique()}, "
      f"Importers: {df['importer'].nunique()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")

incentives = pd.read_csv(INCENTIVE_FILE)
incentive_dict = dict(zip(
    incentives['country_iso2'].astype(str),
    incentives['incentive_intro_year']
))

has_generosity = ('generosity_exp' in df.columns
                  and df['generosity_exp'].nunique() > 1)
print(f"Generosity exporter data available: {has_generosity}")

has_generosity_imp = ('generosity_imp' in df.columns
                      and df['generosity_imp'].nunique() > 1)
print(f"Generosity importer data available: {has_generosity_imp}")

has_efwresid = ('efwresid_exporter' in df.columns
                and df['efwresid_exporter'].notna().sum() > 50)
print(f"EFWRESID data available: {has_efwresid}")
if has_efwresid:
    print(f"  efwresid_exporter coverage: "
          f"{df['efwresid_exporter'].notna().sum():,}")
    print(f"  efwresid_importer coverage: "
          f"{df['efwresid_importer'].notna().sum():,}")

# Type dummy variables — Olsberg three-category typology (rebate, tax_credit, tax_shelter)
type_dummy_vars = [f'{d}_exp' for d in
                   ['is_rebate', 'is_tax_credit', 'is_tax_shelter']]
type_vars_available = [v for v in type_dummy_vars
                       if v in df.columns and df[v].nunique() > 1]
print(f"Incentive type dummies available: {type_vars_available}")


# =============================================================================
# PHASE 1b: PRE-COMPUTE DERIVED VARIABLES (vectorised)
# =============================================================================

print("\nPre-computing derived variables (vectorised)...")

for d_label, d in [('Analysis', df), ('Merged', merged)]:
    # Pair ID — compute once, reuse everywhere
    d['pair_id'] = d['exporter'].astype(str) + '_' + d['importer'].astype(str)

    # Map incentive intro year
    d['intro_year_exp'] = d['exporter'].map(incentive_dict)

    # Vectorised years_active
    has_intro = d['intro_year_exp'].notna()
    d['years_active_exp'] = np.where(
        (d['incentive_exporter'] == 1) & has_intro,
        np.minimum(d['year'] - d['intro_year_exp'], MAX_YEARS_ACTIVE), 0
    )

    # Vectorised early/late adopter (replaces slow .apply() with lambdas)
    active = has_intro & (d['year'] >= d['intro_year_exp'])
    d['early_adopter_exp'] = (
        active & (d['intro_year_exp'] < EARLY_CUTOFF)
    ).astype(int)
    d['late_adopter_exp'] = (
        active & (d['intro_year_exp'] >= EARLY_CUTOFF)
    ).astype(int)

    print(f"  {d_label}: years_active>0: "
          f"{(d['years_active_exp'] > 0).sum():,}, "
          f"early: {d['early_adopter_exp'].sum():,}, "
          f"late: {d['late_adopter_exp'].sum():,}")


# =============================================================================
# HELPERS
# =============================================================================

def fit_ppml(fml_rhs, fe_rhs, data, label=""):
    """
    Fit PPML via pyfixest.fepois with absorbed FE and pair-clustered SE.
    fml_rhs: regressors string, e.g. 'log_gdp_importer + log_dist'
    fe_rhs:  FE string, e.g. 'importer + exporter + year'
    Returns the fitted Fepois model or None.
    """
    formula = f'{PPML_DEP} ~ {fml_rhs} | {fe_rhs}'
    try:
        m = pf.fepois(formula, data=data,
                       vcov={'CRV1': 'pair_id'})
        return m
    except Exception as e:
        print(f"  PPML failed{' (' + label + ')' if label else ''}: {e}")
        return None


def fit_pair_fe(fml_rhs, data, label=""):
    """
    Fit OLS with pair + year FE via pyfixest.feols.
    Returns the fitted Feols model or None.
    """
    formula = f'{OLS_DEP} ~ {fml_rhs} | pair_id + year'
    try:
        m = pf.feols(formula, data=data,
                      vcov={'CRV1': 'pair_id'})
        return m
    except Exception as e:
        print(f"  Pair FE failed{' (' + label + ')' if label else ''}: {e}")
        return None


def get_coef_info(m, var_name):
    """Extract (coef, se, pvalue) for a variable, or None."""
    if m is None:
        return None
    try:
        tidy = m.tidy()
        if var_name in tidy.index:
            row = tidy.loc[var_name]
            return row['Estimate'], row['Std. Error'], row['Pr(>|t|)']
    except Exception:
        pass
    return None


def stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


def print_model_results(m, var_list, label=""):
    """Print key coefficients from a fitted pyfixest model."""
    if m is None:
        print(f"  {label}: model not estimated")
        return
    print(f"\n{'Variable':<30} {'Coef':>10} {'SE':>10} "
          f"{'p':>8} {'Sig':>5} {'% effect':>10}")
    print("-" * 75)
    tidy = m.tidy()
    for var in var_list:
        if var in tidy.index:
            row = tidy.loc[var]
            coef = row['Estimate']
            se = row['Std. Error']
            p = row['Pr(>|t|)']
            s = stars(p)
            pct = (np.exp(coef) - 1) * 100
            print(f"{var:<30} {coef:>10.4f} {se:>10.4f} "
                  f"{p:>8.4f} {s:>5} {pct:>+9.1f}%")
        else:
            print(f"{var:<30} {'--':>10}")


def fmt_coef(model, var_name):
    """Format a coefficient for summary tables."""
    info = get_coef_info(model, var_name)
    if info is None:
        return f"{'--':>14}"
    coef, se, p = info
    s = stars(p)
    pct = (np.exp(coef) - 1) * 100
    return f"{coef:>6.3f}{s:<3}({pct:>+.0f}%)"


def pseudo_r2(m):
    """McFadden pseudo R² for pyfixest fepois model."""
    if m is None:
        return np.nan
    try:
        return 1 - m.deviance / m._null_deviance
    except Exception:
        # Fallback: pyfixest may not expose null_deviance easily
        return np.nan


def run_pair_fe_model(df_sub, tv_vars, label):
    """
    Run OLS pair + year FE via pyfixest.feols.
    Returns the fitted model or None.
    """
    df_sub = df_sub.copy()

    # Ensure pair_id exists
    if 'pair_id' not in df_sub.columns:
        df_sub['pair_id'] = (df_sub['exporter'].astype(str) + '_'
                             + df_sub['importer'].astype(str))

    # Drop singleton pairs (pyfixest handles this, but we report counts)
    pc = df_sub.groupby('pair_id').size()
    df_panel = df_sub[df_sub['pair_id'].isin(pc[pc > 1].index)].copy()

    # Filter to valid dependent variable
    df_panel = df_panel[df_panel[OLS_DEP].notna()].copy()

    print(f"\n{'='*70}")
    print(f"PAIR FE (OLS): {label}")
    print(f"{'='*70}")
    print(f"Observations: {len(df_panel):,}, "
          f"Pairs: {df_panel['pair_id'].nunique()}")

    if len(df_panel) < 50:
        print(f"  SKIPPED - too few observations ({len(df_panel)})")
        return None

    # Report within-pair variation in incentive variables
    for inc_var in ['incentive_exporter', 'incentive_importer']:
        if inc_var in tv_vars:
            iv = df_panel.groupby('pair_id')[inc_var].nunique()
            n_varies = (iv > 1).sum()
            print(f"  {inc_var} varies within pair: "
                  f"{n_varies} ({n_varies / len(iv) * 100:.1f}%)")

    rhs = ' + '.join(tv_vars)
    m = fit_pair_fe(rhs, df_panel, label)

    if m:
        print(f"Within R²: {m._r2_within:.4f}, N: {m._N}")
        print_model_results(m, tv_vars, label)

    return m


def run_event_study(df_src, tv_controls, label_str):
    """
    Run OLS pair + year FE event study via pyfixest.feols.
    incentive_importer kept as control; exporter incentive replaced by
    relative-time dummies.
    """
    print(f"\n{'='*70}")
    print(f"EVENT STUDY: {label_str}")
    print(f"{'='*70}")

    # Restrict to exporters with known incentive intro dates
    df_ev = df_src[df_src['exporter'].map(incentive_dict).notna()].copy()
    df_ev['intro_year_exp'] = df_ev['exporter'].map(incentive_dict)
    df_ev['rel_year'] = (df_ev['year'] - df_ev['intro_year_exp']).astype(int)
    df_ev = df_ev[
        (df_ev['rel_year'] >= EVENT_WINDOW_MIN)
        & (df_ev['rel_year'] <= EVENT_WINDOW_MAX)
    ].copy()

    df_ev = df_ev[df_ev[OLS_DEP].notna()].copy()
    print(f"Sample: {len(df_ev):,} obs, "
          f"{df_ev['exporter'].nunique()} exporters with incentive")

    if len(df_ev) < 50:
        print(f"  SKIPPED - too few observations")
        return None

    # Ensure pair_id
    if 'pair_id' not in df_ev.columns:
        df_ev['pair_id'] = (df_ev['exporter'].astype(str) + '_'
                            + df_ev['importer'].astype(str))

    # Drop singleton pairs
    pc = df_ev.groupby('pair_id').size()
    df_ev = df_ev[df_ev['pair_id'].isin(pc[pc > 1].index)].copy()
    print(f"After dropping singletons: {len(df_ev):,} obs, "
          f"{df_ev['pair_id'].nunique()} pairs")

    if len(df_ev) < 50:
        print(f"  SKIPPED - too few observations after dropping singletons")
        return None

    # Create relative-time dummies (t-1 is omitted reference)
    # Use clean column names for pyfixest compatibility
    rel_vars = []
    for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1):
        if t != -1:
            col = f'rel_t_pos{t}' if t >= 0 else f'rel_t_neg{abs(t)}'
            df_ev[col] = (df_ev['rel_year'] == t).astype(int)
            rel_vars.append(col)

    # Controls: time-varying gravity vars minus exporter incentive
    controls = [v for v in tv_controls if v != 'incentive_exporter']
    if 'incentive_importer' not in controls:
        controls.append('incentive_importer')

    rhs = ' + '.join(controls + rel_vars)
    formula = f'{OLS_DEP} ~ {rhs} | pair_id + year'

    try:
        m_ev = pf.feols(formula, data=df_ev,
                         vcov={'CRV1': 'pair_id'})
    except Exception as e:
        print(f"  Event study failed: {e}")
        return None

    # Print event study coefficients
    tidy = m_ev.tidy()
    print(f"\n{'Rel Year':>10} {'Coef':>10} {'SE':>10} "
          f"{'95% CI':>22} {'Sig':>5}")
    print("-" * 60)

    event_coefs = []
    for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1):
        if t == -1:
            event_coefs.append((t, 0, 0, 0, 0))
            print(f"{'t-1':>10} {'0.000':>10} {'(ref)':>10}")
            continue
        col = f'rel_t_pos{t}' if t >= 0 else f'rel_t_neg{abs(t)}'
        if col in tidy.index:
            row = tidy.loc[col]
            coef = row['Estimate']
            se = row['Std. Error']
            p = row['Pr(>|t|)']
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se
            s = stars(p)
            event_coefs.append((t, coef, se, ci_lo, ci_hi))
            print(f"{'t' + format(t, '+d'):>10} {coef:>10.4f} "
                  f"{se:>10.4f} [{ci_lo:>8.4f}, {ci_hi:>8.4f}] "
                  f"{s:>5}")

    # Print importer incentive coefficient
    if 'incentive_importer' in tidy.index:
        row = tidy.loc['incentive_importer']
        coef = row['Estimate']
        se = row['Std. Error']
        p = row['Pr(>|t|)']
        s = stars(p)
        print(f"\n  incentive_importer: {coef:.4f} "
              f"(SE={se:.4f}, p={p:.4f}) {s}")

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


# =============================================================================
# MAIN ESTIMATION LOOP
# =============================================================================

all_results = {}

# Fixed effects string for PPML models (absorbed by pyfixest)
PPML_FE = 'importer + exporter + year'

for efw_key, efw_cfg in EFW_VARIANTS.items():
    efw_exp = efw_cfg['exp']
    efw_imp = efw_cfg['imp']
    efw_label = efw_cfg['label']

    # ─── Build formula components for this EFW variant ───
    BASELINE_RHS = ('log_gdp_importer + log_gdp_exporter + log_dist + '
                    'remoteness_importer + remoteness_exporter')

    CULTURAL_RHS = (f'contig + comlang_off + col45 + rta + '
                    f'{efw_exp} + {efw_imp}')

    FULL_RHS = f'{BASELINE_RHS} + {CULTURAL_RHS}'

    # Time-varying vars for pair FE models
    TV_FULL = [
        'log_gdp_importer', 'log_gdp_exporter',
        'remoteness_importer', 'remoteness_exporter',
        efw_exp, efw_imp,
        'rta', 'incentive_exporter', 'incentive_importer'
    ]

    TV_NO_EFW = [v for v in TV_FULL if v not in [efw_exp, efw_imp]]

    tag = f"BaTIS [{efw_label}]"

    print(f"\n\n{'#'*70}")
    print(f"# {'='*66} #")
    print(f"#   EFW VARIANT: {efw_label:>48} #")
    print(f"# {'='*66} #")
    print(f"{'#'*70}")

    sr = {}

    # ─────────────────────────────────────────────────────────────────
    # MODEL 1: Baseline gravity (PPML)
    # ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"MODEL 1: Baseline Gravity (PPML) - {tag}")
    print(f"{'='*70}")

    m1 = fit_ppml(BASELINE_RHS, PPML_FE, df, "M1")
    if m1:
        print(f"Pseudo-R²: {pseudo_r2(m1):.4f}, N: {m1._N}")
        print_model_results(m1, [
            'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
            'remoteness_importer', 'remoteness_exporter'
        ], "M1")
    sr['M1'] = m1

    # ─────────────────────────────────────────────────────────────────
    # MODEL 2: Full gravity (PPML)
    # ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"MODEL 2: Full Gravity (PPML) - {tag}")
    print(f"{'='*70}")

    m2 = fit_ppml(FULL_RHS, PPML_FE, df, "M2")
    if m2:
        print(f"Pseudo-R²: {pseudo_r2(m2):.4f}, N: {m2._N}")
        print_model_results(m2, [
            'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
            'remoteness_importer', 'remoteness_exporter',
            'contig', 'comlang_off', 'col45', 'rta',
            efw_exp, efw_imp
        ], "M2")
    sr['M2'] = m2

    # ─────────────────────────────────────────────────────────────────
    # MODEL 3: Incentive dummies (PPML)
    # ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"MODEL 3: Incentive Dummies (PPML) - {tag}")
    print(f"{'='*70}")

    m3_rhs = f'{FULL_RHS} + incentive_exporter + incentive_importer'
    m3 = fit_ppml(m3_rhs, PPML_FE, df, "M3")
    if m3:
        print(f"Pseudo-R²: {pseudo_r2(m3):.4f}, N: {m3._N}")
        print_model_results(m3, [
            'log_dist', 'contig', 'comlang_off', 'col45', 'rta',
            efw_exp, efw_imp,
            'incentive_exporter', 'incentive_importer'
        ], "M3")
    sr['M3'] = m3

    # ─────────────────────────────────────────────────────────────────
    # MODEL 4: Incentive type dummies (PPML)
    # Olsberg typology: rebate, tax_credit, tax_shelter
    # Rebate is the omitted reference category. The incentive_exporter
    # dummy is retained to capture the rebate effect; tax_credit and
    # tax_shelter coefficients are interpreted as the additional effect
    # relative to rebate.
    # ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"MODEL 4: Incentive Types (PPML) - {tag}")
    print(f"{'='*70}")

    m4 = None
    if type_vars_available:
        type_vars_m4 = [v for v in type_vars_available
                        if v != 'is_rebate_exp']
        if type_vars_m4:
            type_str = ' + '.join(type_vars_m4)
            m4_rhs = (f'{FULL_RHS} + incentive_exporter + {type_str} + '
                      f'incentive_importer')
            m4 = fit_ppml(m4_rhs, PPML_FE, df, "M4")
            if m4:
                print(f"Pseudo-R²: {pseudo_r2(m4):.4f}, N: {m4._N}")
                print(f"Reference category: rebate (is_rebate_exp omitted)")
                print_model_results(m4, [
                    'log_dist', 'comlang_off',
                    'incentive_exporter', 'incentive_importer'
                ] + type_vars_m4, "M4")
        else:
            print("  SKIPPED - no non-reference type variables with variation")
    else:
        print("  SKIPPED - no incentive type variables with variation")
    sr['M4'] = m4

    # ─────────────────────────────────────────────────────────────────
    # MODEL 5: Generosity (PPML)
    # ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"MODEL 5: Generosity (PPML) - {tag}")
    print(f"{'='*70}")

    m5 = None
    if has_generosity:
        gen_imp_term = ('generosity_imp' if has_generosity_imp
                        else 'incentive_importer')
        m5_rhs = f'{FULL_RHS} + generosity_exp + {gen_imp_term}'
        m5 = fit_ppml(m5_rhs, PPML_FE, df, "M5")
        if m5:
            print(f"Pseudo-R²: {pseudo_r2(m5):.4f}, N: {m5._N}")
            gen_vars = ['generosity_exp']
            gen_vars.append('generosity_imp' if has_generosity_imp
                            else 'incentive_importer')
            print_model_results(m5, gen_vars, "M5")

            # Diagnostic: generosity ranges
            active_gen = df[df['generosity_exp'] > 0]['generosity_exp']
            if len(active_gen) > 0:
                print(f"\n  Generosity exp (active): "
                      f"mean={active_gen.mean()*100:.1f}%, "
                      f"min={active_gen.min()*100:.1f}%, "
                      f"max={active_gen.max()*100:.1f}%")
    else:
        print("  SKIPPED - no generosity data available")
    sr['M5'] = m5

    # ─────────────────────────────────────────────────────────────────
    # MODEL 6: Cluster - years active (PPML)
    # ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"MODEL 6: Cluster - Years Active (PPML) - {tag}")
    print(f"{'='*70}")

    active = df[df['years_active_exp'] > 0]['years_active_exp']
    if len(active) > 0:
        print(f"Years active (when >0): mean={active.mean():.1f}, "
              f"median={active.median():.1f}, max={active.max():.0f}")

    m6_rhs = (f'{FULL_RHS} + incentive_exporter + years_active_exp + '
              f'incentive_importer')
    m6 = fit_ppml(m6_rhs, PPML_FE, df, "M6")
    if m6:
        print(f"Pseudo-R²: {pseudo_r2(m6):.4f}, N: {m6._N}")
        print_model_results(m6, [
            'incentive_exporter', 'years_active_exp', 'incentive_importer'
        ], "M6")
    sr['M6'] = m6

    # ─────────────────────────────────────────────────────────────────
    # MODEL 7: Cluster - early vs late (PPML)
    # ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"MODEL 7: Cluster - Early vs Late (PPML) - {tag}")
    print(f"{'='*70}")

    print(f"Early adopter obs: {df['early_adopter_exp'].sum():,}")
    print(f"Late adopter obs:  {df['late_adopter_exp'].sum():,}")

    m7_rhs = (f'{FULL_RHS} + early_adopter_exp + late_adopter_exp + '
              f'incentive_importer')
    m7 = fit_ppml(m7_rhs, PPML_FE, df, "M7")
    if m7:
        print(f"Pseudo-R²: {pseudo_r2(m7):.4f}, N: {m7._N}")
        print_model_results(m7, [
            'early_adopter_exp', 'late_adopter_exp', 'incentive_importer'
        ], "M7")
    sr['M7'] = m7

    # ─────────────────────────────────────────────────────────────────
    # PREPARE SUBSAMPLES
    # ─────────────────────────────────────────────────────────────────

    print(f"\n--- Preparing subsamples for pair FE models ---")

    df_excl_aa = df[~(df['exporter'].isin(ANGLO)
                      & df['importer'].isin(ANGLO))].copy()
    df_excl_anglo = df[~(df['exporter'].isin(ANGLO)
                         | df['importer'].isin(ANGLO))].copy()

    est_vars_no_efw = [
        OLS_DEP, 'log_gdp_importer', 'log_gdp_exporter',
        'log_dist', 'contig', 'comlang_off', 'col45',
        'rta', 'remoteness_importer', 'remoteness_exporter',
        'incentive_exporter', 'incentive_importer'
    ]
    df_no_efw = merged.dropna(subset=est_vars_no_efw).copy()

    # Ensure pair_id on subsamples that are copies
    for sub_df in [df_excl_aa, df_excl_anglo, df_no_efw]:
        if 'pair_id' not in sub_df.columns:
            sub_df['pair_id'] = (sub_df['exporter'].astype(str) + '_'
                                 + sub_df['importer'].astype(str))

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

    # ─────────────────────────────────────────────────────────────────
    # MODEL 8: Pair FE (OLS) — all subsamples
    # ─────────────────────────────────────────────────────────────────

    print(f"\n\n{'#'*70}")
    print(f"# MODEL 8: Incentive Dummies - Pair FE (OLS) - {tag}")
    print(f"{'#'*70}")

    m8_results = {}
    for ss_name, ss_df, ss_tv in subsamples:
        m8_results[ss_name] = run_pair_fe_model(
            ss_df, ss_tv,
            f'M8 {tag} - {ss_name}')
    sr['M8'] = m8_results

    # ─────────────────────────────────────────────────────────────────
    # MODEL 9: Event study (OLS) — all subsamples
    # ─────────────────────────────────────────────────────────────────

    print(f"\n\n{'#'*70}")
    print(f"# MODEL 9: Event Study - Pair FE (OLS) - {tag}")
    print(f"{'#'*70}")

    m9_results = {}
    for ss_name, ss_df, ss_tv in subsamples:
        m9_results[ss_name] = run_event_study(
            ss_df, ss_tv,
            f'M9 {tag} - {ss_name}')
    sr['M9'] = m9_results

    # ─────────────────────────────────────────────────────────────────
    # MODEL 10: Cluster pair FE (OLS) — all subsamples
    # ─────────────────────────────────────────────────────────────────

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
        m10_results[ss_name] = run_pair_fe_model(
            ss_df, ss_tv,
            f'M10 {tag} - {ss_name}')
    sr['M10'] = m10_results

    # ─────────────────────────────────────────────────────────────────
    # MODEL 11: Incentive Types, pair FE (OLS) — all subsamples
    # Rebate is the omitted reference category (same structure as M4).
    # ─────────────────────────────────────────────────────────────────

    print(f"\n\n{'#'*70}")
    print(f"# MODEL 11: Incentive Types - Pair FE (OLS) - {tag}")
    print(f"{'#'*70}")

    m11_results = {}
    if type_vars_available:
        type_vars_m11 = [v for v in type_vars_available
                         if v != 'is_rebate_exp']

        if type_vars_m11:
            # Keep incentive_exporter as rebate baseline; add tax_credit
            # and tax_shelter dummies as additional effects
            TV_FULL_TYPES = TV_FULL + type_vars_m11
            TV_NO_EFW_TYPES = TV_NO_EFW + type_vars_m11

            subsamples_m11 = [
                ('Full sample', df, TV_FULL_TYPES),
                ('Excl Anglo-to-Anglo', df_excl_aa, TV_FULL_TYPES),
                ('Excl all Anglo', df_excl_anglo, TV_FULL_TYPES),
                ('No EFW', df_no_efw, TV_NO_EFW_TYPES),
            ]

            for ss_name, ss_df, ss_tv in subsamples_m11:
                m11_results[ss_name] = run_pair_fe_model(
                    ss_df, ss_tv,
                    f'M11 {tag} - {ss_name}')
        else:
            print("  SKIPPED - no non-reference type variables with variation")
    else:
        print("  SKIPPED - no incentive type variables with variation")
    sr['M11'] = m11_results

    # ─────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────

    print(f"\n\n{'='*70}")
    print(f"SUMMARY: INCENTIVE EFFECTS - {tag}")
    print(f"{'='*70}")

    # --- Overall incentive effects ---
    print(f"\n--- OVERALL INCENTIVE ---")
    print(f"\n{'Specification':<40} {'inc_exp':>15} {'inc_imp':>15}")
    print("-" * 72)

    for mname in ['M3', 'M6']:
        m = sr.get(mname)
        if m:
            print(f"{mname + ' (PPML)':<40}"
                  f"{fmt_coef(m, 'incentive_exporter'):>15}"
                  f"{fmt_coef(m, 'incentive_importer'):>15}")

    for mname in ['M8', 'M10']:
        mr = sr.get(mname, {})
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

    if sr.get('M4'):
        row_str = f"{'M4 (PPML)':<25}"
        for tv in type_display:
            row_str += fmt_coef(sr['M4'], tv).rjust(20)
        print(row_str)

    # M11 Pair FE type results
    m11_mr = sr.get('M11', {})
    for ss_name, model in m11_mr.items():
        if model is None:
            continue
        lab = f"M11 PFE ({ss_name})"[:25]
        row_str = f"{lab:<25}"
        for tv in type_display:
            row_str += fmt_coef(model, tv).rjust(20)
        print(row_str)

    # --- Early vs late ---
    print(f"\n--- EARLY vs LATE ADOPTERS ---")
    print(f"{'Specification':<25} {'early':>18} {'late':>18} "
          f"{'imp':>18}")
    print("-" * 79)
    if sr.get('M7'):
        print(f"{'M7 (PPML)':<25}"
              f"{fmt_coef(sr['M7'], 'early_adopter_exp'):>18}"
              f"{fmt_coef(sr['M7'], 'late_adopter_exp'):>18}"
              f"{fmt_coef(sr['M7'], 'incentive_importer'):>18}")

    # --- Generosity ---
    print(f"\n--- GENEROSITY ---")
    if sr.get('M5'):
        gen_imp_var = ('generosity_imp' if has_generosity_imp
                       else 'incentive_importer')
        print(f"{'M5 (PPML)':<25}"
              f"{fmt_coef(sr['M5'], 'generosity_exp'):>18}"
              f"{fmt_coef(sr['M5'], gen_imp_var):>18}")

    print(f"\n* p<0.1, ** p<0.05, *** p<0.01")
    print(f"PPML: estimated on levels, coefficients = semi-elasticities")
    print(f"OLS pair FE: within-transformation on log dep var")

    all_results[efw_key] = sr


# =============================================================================
# FINAL SUMMARY
# =============================================================================

elapsed = time.time() - t_start

print(f"\n\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")

for efw_key, sr in all_results.items():
    efw_label = EFW_VARIANTS[efw_key]['label']
    n_estimated = sum(1 for k, v in sr.items()
                      if v is not None and not isinstance(v, dict))
    n_pair_fe = sum(1 for k, v in sr.items()
                    if isinstance(v, dict)
                    for _, m in v.items() if m is not None)
    print(f"  BaTIS [{efw_label}]: {n_estimated} PPML models, "
          f"{n_pair_fe} pair FE specifications estimated")

print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")