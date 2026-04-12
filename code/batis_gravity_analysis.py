"""
batis_gravity_analysis.py
=========================
Gravity model estimation using BaTIS SK1 (audiovisual services) data.
Loads pre-prepared datasets from batis_data_prep.py.

Run batis_data_prep.py FIRST to create the analysis datasets.

Phases:
  1. Load analysis datasets
  2. Incremental gravity models (M1-M9)
  3. Importer + Exporter + Year FE
  4. Country-pair + Year FE (multiple subsamples + types + generosity)
  5. Event study
  6. Cluster effects
  7. Summary
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

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
print(f"Generosity data available: {has_generosity}")


# =============================================================================
# HELPERS
# =============================================================================

def extract_incentive_coef(model, var_name):
    for lookup in [var_name, f'{var_name}_dm']:
        if lookup in model.params:
            coef = model.params[lookup]
            p = model.pvalues[lookup]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            pct = (np.exp(coef) - 1) * 100
            return f"{coef:>6.3f}{stars:<3}({pct:>+.0f}%)"
    return f"{'--':>14}"

def get_model_by_name(models_list, name):
    for mname, model in models_list:
        if mname == name:
            return model
    return None

def run_pair_fe(df_sub, dep_var, tv, label):
    df_sub = df_sub.copy()
    df_sub['pair_id'] = df_sub['exporter'] + '_' + df_sub['importer']
    pc = df_sub.groupby('pair_id').size()
    df_panel = df_sub[df_sub['pair_id'].isin(pc[pc > 1].index)].copy()

    print(f"\n{'='*70}")
    print(f"COUNTRY-PAIR + YEAR FE: {label}")
    print(f"{'='*70}")
    print(f"Observations: {len(df_panel):,}, Pairs: {df_panel['pair_id'].nunique()}")

    for inc_var in ['incentive_exporter', 'incentive_importer']:
        if inc_var in tv:
            iv = df_panel.groupby('pair_id')[inc_var].nunique()
            print(f"  {inc_var} varies: {(iv > 1).sum()} ({(iv > 1).mean()*100:.1f}%)")

    demean_vars = [dep_var] + tv
    df_dm = df_panel.copy()
    pm = df_panel.groupby('pair_id')[demean_vars].transform('mean')
    for var in demean_vars:
        df_dm[f'{var}_dm'] = df_panel[var] - pm[var]

    dm_formula = f'{dep_var}_dm ~ ' + ' + '.join([f'{v}_dm' for v in tv]) + ' + C(year) - 1'
    m = smf.ols(dm_formula, data=df_dm).fit(cov_type='HC1')

    ss_res = np.sum(m.resid ** 2)
    ss_tot = np.sum(df_dm[f'{dep_var}_dm'] ** 2)
    within_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"\nWithin R²: {within_r2:.4f}, N: {int(m.nobs)}")
    print(f"\n{'Variable':<30} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
    print("-" * 65)
    for var in tv:
        dm_var = f'{var}_dm'
        if dm_var in m.params:
            coef = m.params[dm_var]
            se = m.bse[dm_var]
            p = m.pvalues[dm_var]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"{var:<30} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5}")

    return m


# =============================================================================
# PHASE 2: INCREMENTAL GRAVITY MODELS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: INCREMENTAL GRAVITY MODELS")
print("=" * 70)

m1 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist', data=df).fit(cov_type='HC1')
print(f"\nM1 (Baseline): R²={m1.rsquared:.4f}, N={int(m1.nobs)}")

m2 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45', data=df).fit(cov_type='HC1')
print(f"M2 (+Cultural): R²={m2.rsquared:.4f}")

m3 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter', data=df).fit(cov_type='HC1')
print(f"M3 (+Remoteness): R²={m3.rsquared:.4f}")

m4 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter + rta', data=df).fit(cov_type='HC1')
print(f"M4 (+RTA): R²={m4.rsquared:.4f}")

m5 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter + rta + efw_importer + efw_exporter', data=df).fit(cov_type='HC1')
print(f"M5 (+EFW): R²={m5.rsquared:.4f}")

f6 = 'log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter + rta + efw_importer + efw_exporter + C(year)'
m6, m7, m8, m9 = None, None, None, None

type_dummy_vars = [f'{d}_exp' for d in ['is_refundable_credit', 'is_transferable_credit', 'is_standard_credit', 'is_cash_rebate']]
type_vars_with_var = [v for v in type_dummy_vars if v in df.columns and df[v].nunique() > 1]

try:
    m6 = smf.ols(f6, data=df).fit(cov_type='HC1')
    print(f"M6 (+Year FE): R²={m6.rsquared:.4f}")

    m7 = smf.ols(f6 + ' + incentive_exporter + incentive_importer', data=df).fit(cov_type='HC1')
    print(f"M7 (+Incentives): R²={m7.rsquared:.4f}")

    f8 = f6 + ' + ' + ' + '.join(type_vars_with_var) + ' + incentive_importer'
    m8 = smf.ols(f8, data=df).fit(cov_type='HC1')
    print(f"M8 (+Incentive Types): R²={m8.rsquared:.4f}")

    if has_generosity:
        f9 = f6 + ' + generosity_exp + incentive_importer'
        m9 = smf.ols(f9, data=df).fit(cov_type='HC1')
        print(f"M9 (+Generosity): R²={m9.rsquared:.4f}")

except (MemoryError, np.linalg.LinAlgError) as e:
    print(f"  SKIPPED M6+ — {type(e).__name__}")

# Comparison table
key_vars = [
    'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
    'contig', 'comlang_off', 'col45',
    'remoteness_importer', 'remoteness_exporter',
    'rta', 'efw_importer', 'efw_exporter',
    'incentive_exporter', 'incentive_importer', 'generosity_exp'
] + type_vars_with_var

all_models = [('M1', m1), ('M2', m2), ('M3', m3), ('M4', m4), ('M5', m5)]
for name, m in [('M6', m6), ('M7', m7), ('M8', m8), ('M9', m9)]:
    if m is not None:
        all_models.append((name, m))

print(f"\n{'Variable':<25}" + "".join([f"{n:>10}" for n, _ in all_models]))
print("-" * (25 + 10 * len(all_models)))
for var in key_vars:
    row = f"{var:<25}"
    for name, m in all_models:
        if var in m.params:
            coef = m.params[var]
            p = m.pvalues[var]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            row += f"{coef:>7.3f}{stars:<3}"
        else:
            row += f"{'--':>10}"
    print(row)
print("-" * (25 + 10 * len(all_models)))
print(f"{'R-squared':<25}" + "".join([f"{m.rsquared:>10.4f}" for _, m in all_models]))
print(f"{'Observations':<25}" + "".join([f"{int(m.nobs):>10}" for _, m in all_models]))
print("* p<0.1, ** p<0.05, *** p<0.01")


# =============================================================================
# PHASE 3: IMPORTER + EXPORTER + YEAR FE
# =============================================================================

m_fe = None
m_fe_types = None

try:
    print(f"\n{'='*70}")
    print("IMPORTER + EXPORTER + YEAR FE")
    print(f"{'='*70}")

    m_fe = smf.ols('log_trade ~ log_dist + contig + comlang_off + col45 + rta + '
                    'incentive_exporter + incentive_importer + '
                    'C(importer) + C(exporter) + C(year)', data=df).fit(cov_type='HC1')

    fe_vars = ['log_dist', 'contig', 'comlang_off', 'col45', 'rta', 'incentive_exporter', 'incentive_importer']
    print(f"R²: {m_fe.rsquared:.4f}, N: {int(m_fe.nobs)}")
    print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
    print("-" * 60)
    for var in fe_vars:
        if var in m_fe.params:
            print(f"{var:<25} {m_fe.params[var]:>10.4f} {m_fe.bse[var]:>10.4f} "
                  f"{m_fe.pvalues[var]:>8.4f} "
                  f"{'***' if m_fe.pvalues[var]<0.01 else '**' if m_fe.pvalues[var]<0.05 else '*' if m_fe.pvalues[var]<0.1 else '':>5}")
except (MemoryError, np.linalg.LinAlgError) as e:
    print(f"  SKIPPED — {type(e).__name__}")

try:
    print(f"\n{'='*70}")
    print("IMP+EXP+YEAR FE WITH INCENTIVE TYPES")
    print(f"{'='*70}")

    type_vars_in_sample = [v for v in type_dummy_vars if v in df.columns and df[v].nunique() > 1]
    m_fe_types = smf.ols('log_trade ~ log_dist + contig + comlang_off + col45 + rta + '
                          + ' + '.join(type_vars_in_sample) + ' + incentive_importer + '
                          'C(importer) + C(exporter) + C(year)', data=df).fit(cov_type='HC1')
    print(f"R²: {m_fe_types.rsquared:.4f}, N: {int(m_fe_types.nobs)}")
    fe_type_vars = ['log_dist', 'contig', 'comlang_off', 'col45', 'rta', 'incentive_importer'] + type_vars_in_sample
    print(f"\n{'Variable':<30} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
    print("-" * 65)
    for var in fe_type_vars:
        if var in m_fe_types.params:
            print(f"{var:<30} {m_fe_types.params[var]:>10.4f} {m_fe_types.bse[var]:>10.4f} "
                  f"{m_fe_types.pvalues[var]:>8.4f} "
                  f"{'***' if m_fe_types.pvalues[var]<0.01 else '**' if m_fe_types.pvalues[var]<0.05 else '*' if m_fe_types.pvalues[var]<0.1 else '':>5}")
except (MemoryError, np.linalg.LinAlgError) as e:
    print(f"  SKIPPED — {type(e).__name__}")


# =============================================================================
# PHASE 4: COUNTRY-PAIR + YEAR FE
# =============================================================================

time_varying = [
    'log_gdp_importer', 'log_gdp_exporter',
    'remoteness_importer', 'remoteness_exporter',
    'efw_importer', 'efw_exporter',
    'rta', 'incentive_exporter', 'incentive_importer'
]
tv_no_efw = [v for v in time_varying if v not in ['efw_importer', 'efw_exporter']]

pair_fe_results = {}

pair_fe_results['Full sample'] = run_pair_fe(df, 'log_trade', time_varying, 'Full sample')

est_no_efw = ['log_trade', 'log_gdp_importer', 'log_gdp_exporter',
              'log_dist', 'contig', 'comlang_off', 'col45',
              'rta', 'remoteness_importer', 'remoteness_exporter',
              'incentive_exporter', 'incentive_importer']
df_no_efw = merged.dropna(subset=est_no_efw).copy()
pair_fe_results['Full (no EFW)'] = run_pair_fe(df_no_efw, 'log_trade', tv_no_efw, 'Full (no EFW)')

df_naa = df[~((df['exporter'].isin(ANGLO)) & (df['importer'].isin(ANGLO)))].copy()
pair_fe_results['Excl Anglo-to-Anglo'] = run_pair_fe(df_naa, 'log_trade', time_varying, 'Excl Anglo-to-Anglo')

df_nax = df[~((df['exporter'].isin(ANGLO)) | (df['importer'].isin(ANGLO)))].copy()
pair_fe_results['Excl all Anglo'] = run_pair_fe(df_nax, 'log_trade', time_varying, 'Excl all Anglo')

# Types
tv_types = [v for v in time_varying if v != 'incentive_exporter']
type_vars_avail = [v for v in type_dummy_vars if v in df.columns and df[v].nunique() > 1]
tv_types = tv_types + type_vars_avail
pair_fe_results['Types (full sample)'] = run_pair_fe(df, 'log_trade', tv_types, 'Incentive Types')

# Generosity
if has_generosity:
    tv_gen = [v for v in time_varying if v != 'incentive_exporter'] + ['generosity_exp']
    pair_fe_results['Generosity (full sample)'] = run_pair_fe(df, 'log_trade', tv_gen, 'Generosity')


# =============================================================================
# PHASE 5: EVENT STUDY
# =============================================================================

print(f"\n\n{'='*70}")
print("PHASE 5: EVENT STUDY")
print(f"{'='*70}")

df_ev = df[df['exporter'].map(incentive_dict).notna()].copy()
df_ev['intro_year_exp'] = df_ev['exporter'].map(incentive_dict)
df_ev['rel_year'] = (df_ev['year'] - df_ev['intro_year_exp']).astype(int)
df_ev = df_ev[(df_ev['rel_year'] >= EVENT_WINDOW_MIN) & (df_ev['rel_year'] <= EVENT_WINDOW_MAX)].copy()

print(f"Sample: {len(df_ev):,} obs, {df_ev['exporter'].nunique()} exporters with incentive")

for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1):
    if t != -1:
        df_ev[f'rel_t{t}'] = (df_ev['rel_year'] == t).astype(int)

df_ev['pair_id'] = df_ev['exporter'] + '_' + df_ev['importer']
pc = df_ev.groupby('pair_id').size()
df_ev = df_ev[df_ev['pair_id'].isin(pc[pc > 1].index)].copy()
print(f"After dropping singletons: {len(df_ev):,} obs, {df_ev['pair_id'].nunique()} pairs")

controls = ['log_gdp_importer', 'log_gdp_exporter', 'remoteness_importer', 'remoteness_exporter',
            'efw_importer', 'efw_exporter', 'rta']
rel_vars = [f'rel_t{t}' for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1) if t != -1]

all_ev_vars = ['log_trade'] + controls + rel_vars
df_dm_ev = df_ev.copy()
pm_ev = df_ev.groupby('pair_id')[all_ev_vars].transform('mean')
for var in all_ev_vars:
    df_dm_ev[f'{var}_dm'] = df_ev[var] - pm_ev[var]

year_dummies = pd.get_dummies(df_dm_ev['year'], prefix='yr', drop_first=True, dtype=float)
X = pd.concat([df_dm_ev[[f'{v}_dm' for v in controls + rel_vars]], year_dummies], axis=1)
y = df_dm_ev['log_trade_dm']

m_ev = sm.OLS(y, X).fit(cov_type='HC1')

print(f"\n{'Rel Year':>10} {'Coef':>10} {'SE':>10} {'95% CI':>22} {'Sig':>5}")
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
        ci_lo, ci_hi = coef - 1.96*se, coef + 1.96*se
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        event_coefs.append((t, coef, se, ci_lo, ci_hi))
        print(f"{'t'+format(t,'+d'):>10} {coef:>10.4f} {se:>10.4f} [{ci_lo:>8.4f}, {ci_hi:>8.4f}] {stars:>5}")

pre_coefs = [c for t, c, s, lo, hi in event_coefs if EVENT_WINDOW_MIN <= t < -1]
post_coefs = [c for t, c, s, lo, hi in event_coefs if t >= 0]
if pre_coefs and post_coefs:
    print(f"\n  Avg pre-treatment (t-5 to t-2): {np.mean(pre_coefs):.4f}")
    print(f"  Avg post-treatment (t0 to t+10): {np.mean(post_coefs):.4f}")


# =============================================================================
# PHASE 6: CLUSTER EFFECTS
# =============================================================================

print(f"\n\n{'='*70}")
print("PHASE 6: CLUSTER EFFECTS")
print(f"{'='*70}")

df_cl = df.copy()
df_cl['intro_year_exp'] = df_cl['exporter'].map(incentive_dict)
df_cl['years_active_exp'] = np.where(
    (df_cl['incentive_exporter'] == 1) & df_cl['intro_year_exp'].notna(),
    np.minimum(df_cl['year'] - df_cl['intro_year_exp'], MAX_YEARS_ACTIVE), 0
)

# Model A: Year FE
print(f"\n--- Years Active (Year FE) ---")
try:
    f_a = ('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + '
           'contig + comlang_off + col45 + remoteness_importer + remoteness_exporter + '
           'rta + efw_importer + efw_exporter + '
           'incentive_exporter + years_active_exp + incentive_importer + C(year)')
    m_a = smf.ols(f_a, data=df_cl).fit(cov_type='HC1')
    for var in ['incentive_exporter', 'years_active_exp', 'incentive_importer']:
        if var in m_a.params:
            stars = "***" if m_a.pvalues[var] < 0.01 else "**" if m_a.pvalues[var] < 0.05 else "*" if m_a.pvalues[var] < 0.1 else ""
            print(f"  {var:<25} {m_a.params[var]:>8.4f} ({m_a.bse[var]:.4f}) {stars}")
except (MemoryError, np.linalg.LinAlgError) as e:
    print(f"  SKIPPED — {type(e).__name__}")

# Model B: Pair FE with years_active
print(f"\n--- Years Active (Pair FE) ---")
df_cl['pair_id'] = df_cl['exporter'] + '_' + df_cl['importer']
pc = df_cl.groupby('pair_id').size()
df_cl_panel = df_cl[df_cl['pair_id'].isin(pc[pc > 1].index)].copy()

tv_cl = ['log_gdp_importer', 'log_gdp_exporter', 'remoteness_importer', 'remoteness_exporter',
         'efw_importer', 'efw_exporter', 'rta', 'incentive_exporter', 'years_active_exp', 'incentive_importer']

demean_cl = ['log_trade'] + tv_cl
df_dm_cl = df_cl_panel.copy()
pm_cl = df_cl_panel.groupby('pair_id')[demean_cl].transform('mean')
for var in demean_cl:
    df_dm_cl[f'{var}_dm'] = df_cl_panel[var] - pm_cl[var]

m_b = smf.ols('log_trade_dm ~ ' + ' + '.join([f'{v}_dm' for v in tv_cl]) + ' + C(year) - 1',
              data=df_dm_cl).fit(cov_type='HC1')

for var in tv_cl:
    dm_var = f'{var}_dm'
    if dm_var in m_b.params:
        stars = "***" if m_b.pvalues[dm_var] < 0.01 else "**" if m_b.pvalues[dm_var] < 0.05 else "*" if m_b.pvalues[dm_var] < 0.1 else ""
        print(f"  {var:<25} {m_b.params[dm_var]:>8.4f} ({m_b.bse[dm_var]:.4f}) {stars}")

if 'years_active_exp_dm' in m_b.params:
    ya = m_b.params['years_active_exp_dm']
    ya_p = m_b.pvalues['years_active_exp_dm']
    print(f"\n  Cluster test: years_active = {ya:.4f} (p={ya_p:.4f})")

# Model C: Early vs late adopters
print(f"\n--- Early vs Late Adopters (Pair FE) ---")
df_cl_panel['early_adopter_exp'] = df_cl_panel.apply(
    lambda row: (1 if (row['exporter'] in incentive_dict
                      and pd.notna(incentive_dict[row['exporter']])
                      and incentive_dict[row['exporter']] < EARLY_CUTOFF
                      and row['year'] >= incentive_dict[row['exporter']]) else 0), axis=1
)
df_cl_panel['late_adopter_exp'] = df_cl_panel.apply(
    lambda row: (1 if (row['exporter'] in incentive_dict
                      and pd.notna(incentive_dict[row['exporter']])
                      and incentive_dict[row['exporter']] >= EARLY_CUTOFF
                      and row['year'] >= incentive_dict[row['exporter']]) else 0), axis=1
)

print(f"  Early adopter obs: {df_cl_panel['early_adopter_exp'].sum()}")
print(f"  Late adopter obs: {df_cl_panel['late_adopter_exp'].sum()}")

tv_ad = ['log_gdp_importer', 'log_gdp_exporter', 'remoteness_importer', 'remoteness_exporter',
         'efw_importer', 'efw_exporter', 'rta', 'early_adopter_exp', 'late_adopter_exp', 'incentive_importer']

demean_ad = ['log_trade'] + tv_ad
df_dm_ad = df_cl_panel.copy()
pm_ad = df_cl_panel.groupby('pair_id')[demean_ad].transform('mean')
for var in demean_ad:
    df_dm_ad[f'{var}_dm'] = df_cl_panel[var] - pm_ad[var]

m_c = smf.ols('log_trade_dm ~ ' + ' + '.join([f'{v}_dm' for v in tv_ad]) + ' + C(year) - 1',
              data=df_dm_ad).fit(cov_type='HC1')

for var in ['early_adopter_exp', 'late_adopter_exp', 'incentive_importer']:
    dm_var = f'{var}_dm'
    if dm_var in m_c.params:
        coef = m_c.params[dm_var]
        se = m_c.bse[dm_var]
        p = m_c.pvalues[dm_var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        pct = (np.exp(coef) - 1) * 100
        print(f"  {var:<25} {coef:>8.4f} ({se:.4f}) {stars}  [{pct:+.1f}%]")


# =============================================================================
# PHASE 7: SUMMARY
# =============================================================================

print(f"\n\n{'='*70}")
print("SUMMARY: INCENTIVE EFFECTS ACROSS SPECIFICATIONS")
print(f"{'='*70}")

print(f"\n--- OVERALL INCENTIVE ---")
print(f"\n{'Specification':<35} {'inc_exporter':>15} {'inc_importer':>15}")
print("-" * 67)

m7 = get_model_by_name(all_models, 'M7')
if m7:
    print(f"{'M7 (Year FE)':<35}{extract_incentive_coef(m7, 'incentive_exporter'):>15}"
          f"{extract_incentive_coef(m7, 'incentive_importer'):>15}")

if m_fe is not None:
    print(f"{'Imp+Exp+Year FE':<35}{extract_incentive_coef(m_fe, 'incentive_exporter'):>15}"
          f"{extract_incentive_coef(m_fe, 'incentive_importer'):>15}")

for sname, model in pair_fe_results.items():
    lab = f"Pair+Year FE ({sname})"[:35]
    print(f"{lab:<35}{extract_incentive_coef(model, 'incentive_exporter'):>15}"
          f"{extract_incentive_coef(model, 'incentive_importer'):>15}")

# Types
print(f"\n--- INCENTIVE TYPES ---")
type_summary_vars = ['is_refundable_credit_exp', 'is_transferable_credit_exp',
                     'is_standard_credit_exp', 'is_cash_rebate_exp']

print(f"\n{'Specification':<25}", end="")
for tv in type_summary_vars:
    short = tv.replace('is_', '').replace('_exp', '').replace('_credit', '').replace('_', ' ')
    print(f"{short:>18}", end="")
print()
print("-" * (25 + 18 * len(type_summary_vars)))

m8 = get_model_by_name(all_models, 'M8')
if m8:
    row = f"{'M8 (Year FE)':<25}"
    for tv in type_summary_vars:
        row += extract_incentive_coef(m8, tv).rjust(18)
    print(row)

if m_fe_types is not None:
    row = f"{'Imp+Exp+Year FE (Types)':<25}"
    for tv in type_summary_vars:
        row += extract_incentive_coef(m_fe_types, tv).rjust(18)
    print(row)

if 'Types (full sample)' in pair_fe_results:
    row = f"{'Pair+Year FE (Types)':<25}"
    for tv in type_summary_vars:
        row += extract_incentive_coef(pair_fe_results['Types (full sample)'], tv).rjust(18)
    print(row)

# Generosity
m9 = get_model_by_name(all_models, 'M9')
if m9 or 'Generosity (full sample)' in pair_fe_results:
    print(f"\n{'Specification':<25} {'generosity':>18}")
    print("-" * 43)
    if m9:
        print(f"{'M9 (Year FE)':<25}{extract_incentive_coef(m9, 'generosity_exp'):>18}")
    if 'Generosity (full sample)' in pair_fe_results:
        print(f"{'Pair+Year FE (Gen)':<25}{extract_incentive_coef(pair_fe_results['Generosity (full sample)'], 'generosity_exp'):>18}")

print(f"\n* p<0.1, ** p<0.05, *** p<0.01")
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)