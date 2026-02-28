"""
imdb_filming_gravity_analysis.py
================================
Script 2 of 2 for the Filming Location Gravity Model.

Standalone gravity model analysis using IMDb filming location data.
Mirrors the structure of batis_gravity_analysis.py.

Pipeline:
  1. Load bilateral filming trade flows (from Script 1)
  2. Merge with CEPII gravity variables
  3. Merge with World Bank GDP
  4. Merge with EFW (Economic Freedom of the World)
  5. Merge with film incentive dates
  6. Calculate remoteness measures
  7. Estimate gravity models — film count dependent variable (incremental)
  8. Estimate gravity models — budget dependent variable (incremental)
  9. Robustness: importer + exporter + year FE
  10. Robustness: country-pair + year FE
  10b. Pair FE without EFW (more observations)
  10c. Pair FE excluding Anglo-to-Anglo
  10d. Pair FE excluding all Anglo
  11. Summary comparison
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'

# Input: from Script 1
TRADE_FLOWS_FILE = f'{PROJECT_DIR}\\data\\processed\\imdb\\imdb_filming_trade_flows.csv'

# Input: from existing pipeline (already cleaned)
CEPII_FILE = f'{PROJECT_DIR}\\data\\processed\\gravity_vars_cepii.csv'
GDP_FILE = f'{PROJECT_DIR}\\data\\processed\\gdp_cleaned.csv'
EFW_FILE = f'{PROJECT_DIR}\\data\\processed\\efw_cleaned.csv'
INCENTIVE_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\film_incentive_intro_dates.csv'

OUTPUT_DIR = f'{PROJECT_DIR}\\output'

YEAR_MAX = 2023  # World Bank GDP constraint

ANGLO = {'US', 'GB', 'CA', 'AU', 'NZ'}

# =============================================================================
# PHASE 1: LOAD TRADE FLOWS
# =============================================================================

print("=" * 70)
print("PHASE 1: LOAD FILMING LOCATION TRADE FLOWS")
print("=" * 70)

flows = pd.read_csv(TRADE_FLOWS_FILE)
print(f"Loaded: {len(flows):,} bilateral pair-year observations")
print(f"Year range: {flows['year'].min()} - {flows['year'].max()}")
print(f"Unique importers: {flows['importer'].nunique()}")
print(f"Unique exporters: {flows['exporter'].nunique()}")
print(f"Observations with budget: {flows['log_budget'].notna().sum():,}")

# =============================================================================
# PHASE 2: MERGE WITH CEPII GRAVITY VARIABLES
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: MERGE WITH CEPII GRAVITY VARIABLES")
print("=" * 70)

cepii = pd.read_csv(CEPII_FILE)
print(f"Loaded CEPII: {len(cepii):,} rows, years {cepii['year'].min()}-{cepii['year'].max()}")

# Forward-fill CEPII for years beyond its coverage
cepii_max_year = cepii['year'].max()
if YEAR_MAX > cepii_max_year:
    print(f"Forward-filling CEPII from {cepii_max_year} to {YEAR_MAX}...")
    cepii_latest = cepii[cepii['year'] == cepii_max_year].copy()
    new_rows = []
    for fill_year in range(cepii_max_year + 1, YEAR_MAX + 1):
        temp = cepii_latest.copy()
        temp['year'] = fill_year
        new_rows.append(temp)
    cepii = pd.concat([cepii] + new_rows, ignore_index=True)
    print(f"CEPII now covers: {cepii['year'].min()}-{cepii['year'].max()}")

# Merge: importer = iso_d (destination of film finance), exporter = iso_o (origin of filming)
# Note: in this specification, the "exporter" is the filming country (selling production services)
merged = flows.merge(
    cepii,
    left_on=['exporter', 'importer', 'year'],
    right_on=['iso_o', 'iso_d', 'year'],
    how='left'
)
merged = merged.drop(columns=['iso_o', 'iso_d'], errors='ignore')

n_before = len(flows)
n_matched = merged['log_dist'].notna().sum()
print(f"\nAfter CEPII merge: {len(merged):,} rows")
print(f"Matched to CEPII: {n_matched:,} ({100*n_matched/len(merged):.1f}%)")
print(f"Missing distance: {merged['log_dist'].isna().sum()}")

# Show which pairs didn't match
unmatched = merged[merged['log_dist'].isna()][['importer', 'exporter']].drop_duplicates()
if len(unmatched) > 0 and len(unmatched) <= 30:
    print(f"\nUnmatched pairs ({len(unmatched)}):")
    for _, row in unmatched.iterrows():
        print(f"  {row['importer']} → {row['exporter']}")

# =============================================================================
# PHASE 3: MERGE WITH GDP
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: MERGE WITH GDP")
print("=" * 70)

gdp = pd.read_csv(GDP_FILE)
print(f"Loaded GDP: {len(gdp):,} rows, years {gdp['date'].min()}-{gdp['date'].max()}")

# Merge importer GDP
merged = merged.merge(
    gdp[['country', 'date', 'log_gdp']].rename(
        columns={'country': 'importer', 'date': 'year', 'log_gdp': 'log_gdp_importer'}
    ),
    on=['importer', 'year'],
    how='left'
)

# Merge exporter GDP
merged = merged.merge(
    gdp[['country', 'date', 'log_gdp']].rename(
        columns={'country': 'exporter', 'date': 'year', 'log_gdp': 'log_gdp_exporter'}
    ),
    on=['exporter', 'year'],
    how='left'
)

print(f"Missing importer GDP: {merged['log_gdp_importer'].isna().sum()} ({merged['log_gdp_importer'].isna().mean()*100:.1f}%)")
print(f"Missing exporter GDP: {merged['log_gdp_exporter'].isna().sum()} ({merged['log_gdp_exporter'].isna().mean()*100:.1f}%)")

# =============================================================================
# PHASE 4: MERGE WITH EFW
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4: MERGE WITH ECONOMIC FREEDOM (EFW)")
print("=" * 70)

efw = pd.read_csv(EFW_FILE)
print(f"Loaded EFW: {len(efw):,} rows, years {efw['year'].min()}-{efw['year'].max()}")

# Forward-fill EFW if needed
efw_max = efw['year'].max()
if efw_max < YEAR_MAX:
    print(f"Forward-filling EFW from {efw_max} to {YEAR_MAX}...")
    efw_latest = efw[efw['year'] == efw_max].copy()
    new_efw = []
    for fill_year in range(efw_max + 1, YEAR_MAX + 1):
        temp = efw_latest.copy()
        temp['year'] = fill_year
        new_efw.append(temp)
    efw = pd.concat([efw] + new_efw, ignore_index=True)

# Merge importer EFW
merged = merged.merge(
    efw.rename(columns={'country': 'importer', 'efw': 'efw_importer'}),
    on=['importer', 'year'],
    how='left'
)

# Merge exporter EFW
merged = merged.merge(
    efw.rename(columns={'country': 'exporter', 'efw': 'efw_exporter'}),
    on=['exporter', 'year'],
    how='left'
)

print(f"Missing importer EFW: {merged['efw_importer'].isna().sum()} ({merged['efw_importer'].isna().mean()*100:.1f}%)")
print(f"Missing exporter EFW: {merged['efw_exporter'].isna().sum()} ({merged['efw_exporter'].isna().mean()*100:.1f}%)")

# =============================================================================
# PHASE 5: MERGE WITH FILM INCENTIVE DATA
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 5: MERGE WITH FILM INCENTIVE DATA")
print("=" * 70)

incentives = pd.read_csv(INCENTIVE_FILE)
print(f"Loaded incentive data: {len(incentives)} countries")
print(f"Columns: {incentives.columns.tolist()}")

# Build lookup dictionaries from the CSV
incentive_dict = dict(zip(
    incentives['country_iso2'].astype(str),
    incentives['incentive_intro_year']
))

# Build incentive type lookups (country -> dummy value for each type)
TYPE_DUMMIES = ['is_refundable_credit', 'is_transferable_credit', 'is_standard_credit', 'is_cash_rebate']
type_lookups = {}
for dtype in TYPE_DUMMIES:
    type_lookups[dtype] = dict(zip(
        incentives['country_iso2'].astype(str),
        incentives[dtype].fillna(0).astype(int)
    ))

# Build generosity lookup (country -> headline rate as fraction)
if 'headline_rate_pct' in incentives.columns:
    generosity_dict = dict(zip(
        incentives['country_iso2'].astype(str),
        incentives['headline_rate_pct'].fillna(0) / 100  # convert to fraction
    ))
    has_generosity = True
    print(f"  Loaded headline_rate_pct for generosity analysis")
else:
    generosity_dict = {}
    has_generosity = False
    print(f"  WARNING: headline_rate_pct column not found — skipping generosity analysis")
    print(f"  To enable, use the updated film_incentive_intro_dates.csv with Olsberg data")

# Create time-varying incentive dummies (overall)
merged['incentive_exporter'] = merged.apply(
    lambda row: 1 if row['exporter'] in incentive_dict and pd.notna(incentive_dict[row['exporter']]) and row['year'] >= incentive_dict[row['exporter']] else 0,
    axis=1
)
merged['incentive_importer'] = merged.apply(
    lambda row: 1 if row['importer'] in incentive_dict and pd.notna(incentive_dict[row['importer']]) and row['year'] >= incentive_dict[row['importer']] else 0,
    axis=1
)

# Create time-varying incentive TYPE dummies for exporter
# These equal 1 only if the country has an active incentive AND it is of that type
for dtype in TYPE_DUMMIES:
    col_exp = f'{dtype}_exp'
    merged[col_exp] = merged.apply(
        lambda row, dt=dtype: (
            1 if (row['exporter'] in incentive_dict
                  and pd.notna(incentive_dict[row['exporter']])
                  and row['year'] >= incentive_dict[row['exporter']]
                  and type_lookups[dt].get(row['exporter'], 0) == 1)
            else 0
        ),
        axis=1
    )

print(f"\nIncentive coverage:")
print(f"  Exporter incentive active: {merged['incentive_exporter'].sum()} ({merged['incentive_exporter'].mean()*100:.1f}%)")
print(f"  Importer incentive active: {merged['incentive_importer'].sum()} ({merged['incentive_importer'].mean()*100:.1f}%)")

print(f"\nExporter incentive by type:")
for dtype in TYPE_DUMMIES:
    col = f'{dtype}_exp'
    n = merged[col].sum()
    print(f"  {dtype}: {n} ({merged[col].mean()*100:.1f}%)")

# Create generosity variable: headline_rate when incentive is active, 0 otherwise
if has_generosity:
    merged['generosity_exp'] = merged.apply(
        lambda row: (
            generosity_dict.get(row['exporter'], 0)
            if (row['exporter'] in incentive_dict
                and pd.notna(incentive_dict[row['exporter']])
                and row['year'] >= incentive_dict[row['exporter']])
            else 0
        ),
        axis=1
    )

    print(f"\nGenerosity (headline rate) for active exporter incentives:")
    active_gen = merged[merged['generosity_exp'] > 0]['generosity_exp']
    if len(active_gen) > 0:
        print(f"  Mean: {active_gen.mean()*100:.1f}%")
        print(f"  Min: {active_gen.min()*100:.1f}%")
        print(f"  Max: {active_gen.max()*100:.1f}%")
        print(f"  Observations with active incentive: {len(active_gen)}")
else:
    merged['generosity_exp'] = 0

# Verify: type dummies should sum to overall incentive dummy
type_sum = sum(merged[f'{d}_exp'] for d in TYPE_DUMMIES)
mismatch = (type_sum != merged['incentive_exporter']).sum()
if mismatch > 0:
    print(f"\n  WARNING: {mismatch} rows where type dummies don't sum to overall incentive")
else:
    print(f"\n  ✓ Type dummies sum correctly to overall incentive dummy")

# =============================================================================
# PHASE 6: CALCULATE REMOTENESS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 6: CALCULATE REMOTENESS MEASURES")
print("=" * 70)

# Load GDP levels for remoteness calculation
gdp_levels = gdp[['country', 'date', 'gdp']].rename(columns={'date': 'year'}).copy()
gdp_levels = gdp_levels.dropna(subset=['gdp'])

world_gdp = gdp_levels.groupby('year')['gdp'].sum().reset_index()
world_gdp.columns = ['year', 'gdp_world']

# Distance data (time-invariant)
dist_year = cepii[cepii['year'] <= 2020]['year'].max()
dist_data = cepii[cepii['year'] == dist_year][['iso_o', 'iso_d', 'dist']].copy()
dist_data = dist_data.dropna(subset=['dist'])
dist_data = dist_data[dist_data['dist'] > 0]
dist_data = dist_data[dist_data['iso_o'] != dist_data['iso_d']]

# Calculate remoteness for each country-year
remoteness_records = []
years = sorted(merged['year'].unique())
countries = sorted(set(merged['exporter'].unique()) | set(merged['importer'].unique()))

for yr in years:
    gdp_yr = gdp_levels[gdp_levels['year'] == yr].set_index('country')['gdp'].to_dict()
    gdp_world_yr = world_gdp[world_gdp['year'] == yr]['gdp_world'].values
    if len(gdp_world_yr) == 0:
        continue
    gdp_world_yr = gdp_world_yr[0]

    for country in countries:
        dists = dist_data[dist_data['iso_o'] == country][['iso_d', 'dist']]
        if len(dists) == 0:
            continue

        weighted_sum = 0
        for _, row in dists.iterrows():
            partner = row['iso_d']
            if partner == country:
                continue
            partner_gdp = gdp_yr.get(partner, None)
            if partner_gdp and partner_gdp > 0:
                gdp_share = partner_gdp / gdp_world_yr
                weighted_sum += gdp_share / row['dist']

        if weighted_sum > 0:
            remoteness_records.append({
                'country': country,
                'year': yr,
                'remoteness': np.log(1 / weighted_sum)
            })

    if yr % 5 == 0:
        print(f"  Processed remoteness for {yr}")

remoteness_df = pd.DataFrame(remoteness_records)
print(f"Remoteness calculated: {len(remoteness_df):,} country-year observations")

# Merge remoteness
merged = merged.merge(
    remoteness_df.rename(columns={'country': 'importer', 'remoteness': 'remoteness_importer'}),
    on=['importer', 'year'], how='left'
)
merged = merged.merge(
    remoteness_df.rename(columns={'country': 'exporter', 'remoteness': 'remoteness_exporter'}),
    on=['exporter', 'year'], how='left'
)

print(f"Missing importer remoteness: {merged['remoteness_importer'].isna().sum()}")
print(f"Missing exporter remoteness: {merged['remoteness_exporter'].isna().sum()}")

# =============================================================================
# PHASE 7: PREPARE ANALYSIS DATASETS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 7: PREPARE ANALYSIS DATASETS")
print("=" * 70)

# --- Dataset A: Film count specification (primary) ---
est_vars_count = [
    'log_num_films', 'log_gdp_importer', 'log_gdp_exporter',
    'log_dist', 'contig', 'comlang_off', 'col45',
    'rta', 'remoteness_importer', 'remoteness_exporter',
    'efw_importer', 'efw_exporter',
    'incentive_exporter', 'incentive_importer'
]

print(f"Total observations: {len(merged):,}")
print(f"\nMissing values (count specification):")
for var in est_vars_count:
    if var in merged.columns:
        n_miss = merged[var].isna().sum()
        if n_miss > 0:
            print(f"  {var}: {n_miss} ({n_miss/len(merged)*100:.1f}%)")

df_count = merged.dropna(subset=est_vars_count).copy()
print(f"\nFilm count analysis sample: {len(df_count):,}")

# --- Dataset B: Budget specification (robustness) ---
est_vars_budget = est_vars_count.copy()
est_vars_budget[0] = 'log_budget'  # Replace log_num_films with log_budget

df_budget = merged.dropna(subset=est_vars_budget).copy()
print(f"Budget analysis sample: {len(df_budget):,}")

# Summary
for label, df_tmp in [('Count', df_count), ('Budget', df_budget)]:
    print(f"\n{label} sample:")
    print(f"  Observations: {len(df_tmp):,}")
    print(f"  Unique exporters: {df_tmp['exporter'].nunique()}")
    print(f"  Unique importers: {df_tmp['importer'].nunique()}")
    print(f"  Year range: {df_tmp['year'].min()} - {df_tmp['year'].max()}")
    print(f"  Incentive exporter active: {df_tmp['incentive_exporter'].mean()*100:.1f}%")

# Save analysis datasets
df_count.to_csv(f'{PROJECT_DIR}\\data\\processed\\imdb\\filming_gravity_count_analysis.csv', index=False)
df_budget.to_csv(f'{PROJECT_DIR}\\data\\processed\\imdb\\filming_gravity_budget_analysis.csv', index=False)
print(f"\nAnalysis datasets saved.")


# =============================================================================
# HELPER: Run incremental gravity models and print comparison table
# =============================================================================

def run_incremental_models(df, dep_var, dep_label):
    """Run Models 1-7 incrementally and print comparison table."""

    print("\n" + "=" * 70)
    print(f"GRAVITY MODELS: {dep_label}")
    print("=" * 70)

    col45_var = 'col45' if 'col45' in df.columns else 'col_dep_ever'

    # Model 1: Baseline
    f1 = f'{dep_var} ~ log_gdp_importer + log_gdp_exporter + log_dist'
    m1 = smf.ols(f1, data=df).fit(cov_type='HC1')
    print(f"\nModel 1 (Baseline): R²={m1.rsquared:.4f}, N={int(m1.nobs)}")

    # Model 2: + Cultural controls
    f2 = f'{dep_var} ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + {col45_var}'
    m2 = smf.ols(f2, data=df).fit(cov_type='HC1')
    print(f"Model 2 (+Cultural): R²={m2.rsquared:.4f}, N={int(m2.nobs)}")

    # Model 3: + Remoteness
    f3 = f2 + ' + remoteness_importer + remoteness_exporter'
    m3 = smf.ols(f3, data=df).fit(cov_type='HC1')
    print(f"Model 3 (+Remoteness): R²={m3.rsquared:.4f}, N={int(m3.nobs)}")

    # Model 4: + RTA
    f4 = f3 + ' + rta'
    m4 = smf.ols(f4, data=df).fit(cov_type='HC1')
    print(f"Model 4 (+RTA): R²={m4.rsquared:.4f}, N={int(m4.nobs)}")

    # Model 5: + EFW
    f5 = f4 + ' + efw_importer + efw_exporter'
    m5 = smf.ols(f5, data=df).fit(cov_type='HC1')
    print(f"Model 5 (+EFW): R²={m5.rsquared:.4f}, N={int(m5.nobs)}")

    # Model 6: + Year FE
    f6 = f5 + ' + C(year)'
    m6 = smf.ols(f6, data=df).fit(cov_type='HC1')
    print(f"Model 6 (+Year FE): R²={m6.rsquared:.4f}, N={int(m6.nobs)}")

    # Model 7: + Incentives (overall dummy)
    f7 = f6 + ' + incentive_exporter + incentive_importer'
    m7 = smf.ols(f7, data=df).fit(cov_type='HC1')
    print(f"Model 7 (+Incentives): R²={m7.rsquared:.4f}, N={int(m7.nobs)}")

    # Model 8: Replace overall incentive with type-specific dummies
    # Drop incentive_exporter and replace with the four type dummies
    type_dummy_vars = [f'{d}_exp' for d in ['is_refundable_credit', 'is_transferable_credit',
                                             'is_standard_credit', 'is_cash_rebate']]
    # Check which type dummies have variation in the sample
    type_vars_with_variation = [v for v in type_dummy_vars if df[v].nunique() > 1]
    type_vars_no_variation = [v for v in type_dummy_vars if df[v].nunique() <= 1]
    if type_vars_no_variation:
        print(f"  Note: dropping {type_vars_no_variation} (no variation in sample)")

    f8 = (f6 + ' + ' + ' + '.join(type_vars_with_variation) + ' + incentive_importer')
    m8 = smf.ols(f8, data=df).fit(cov_type='HC1')
    print(f"Model 8 (+Incentive Types): R²={m8.rsquared:.4f}, N={int(m8.nobs)}")

    # Model 9: Replace dummy with continuous generosity measure
    if has_generosity and df['generosity_exp'].nunique() > 1:
        f9 = f6 + ' + generosity_exp + incentive_importer'
        m9 = smf.ols(f9, data=df).fit(cov_type='HC1')
        print(f"Model 9 (+Generosity): R²={m9.rsquared:.4f}, N={int(m9.nobs)}")
    else:
        m9 = None
        print(f"Model 9 (+Generosity): SKIPPED — no generosity data")

    # --- Comparison table ---
    key_vars = [
        'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
        'contig', 'comlang_off', col45_var,
        'remoteness_importer', 'remoteness_exporter',
        'rta', 'efw_importer', 'efw_exporter',
        'incentive_exporter', 'incentive_importer',
        'generosity_exp',
    ] + type_vars_with_variation

    all_models = [('M1', m1), ('M2', m2), ('M3', m3), ('M4', m4), ('M5', m5), ('M6', m6), ('M7', m7), ('M8', m8)]
    if m9 is not None:
        all_models.append(('M9', m9))
    models = all_models

    print(f"\n{'Variable':<25}" + "".join([f"{name:>10}" for name, _ in models]))
    print("-" * (25 + 10 * len(models)))

    for var in key_vars:
        row = f"{var:<25}"
        for name, m in models:
            if var in m.params:
                coef = m.params[var]
                p = m.pvalues[var]
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                row += f"{coef:>7.3f}{stars:<3}"
            else:
                row += f"{'--':>10}"
        print(row)

    print("-" * (25 + 10 * len(models)))
    print(f"{'R-squared':<25}" + "".join([f"{m.rsquared:>10.4f}" for _, m in models]))
    print(f"{'Observations':<25}" + "".join([f"{int(m.nobs):>10}" for _, m in models]))
    print("* p<0.1, ** p<0.05, *** p<0.01")

    return models


# =============================================================================
# PHASE 8: FILM COUNT GRAVITY MODELS
# =============================================================================

count_models = run_incremental_models(df_count, 'log_num_films', 'FILM COUNT (log)')

# =============================================================================
# PHASE 8b: BUDGET GRAVITY MODELS
# =============================================================================

if len(df_budget) >= 50:
    budget_models = run_incremental_models(df_budget, 'log_budget', 'BUDGET (log USD)')
else:
    print(f"\nSkipping budget models — only {len(df_budget)} observations (need ≥50)")
    budget_models = None


# =============================================================================
# HELPER: Fixed effects and pair FE robustness
# =============================================================================

def run_fe_robustness(df, dep_var, dep_label, merged_full=None):
    """Run importer+exporter FE and country-pair FE specifications."""

    col45_var = 'col45' if 'col45' in df.columns else 'col_dep_ever'
    if merged_full is None:
        merged_full = df

    # ----- IMPORTER + EXPORTER + YEAR FE -----
    print("\n" + "=" * 70)
    print(f"IMPORTER + EXPORTER + YEAR FE: {dep_label}")
    print("=" * 70)

    formula_fe = (f'{dep_var} ~ log_dist + contig + comlang_off + {col45_var} + rta + '
                  'incentive_exporter + incentive_importer + '
                  'C(importer) + C(exporter) + C(year)')

    m_fe = smf.ols(formula_fe, data=df).fit(cov_type='HC1')
    print(f"R²: {m_fe.rsquared:.4f}, N: {int(m_fe.nobs)}")

    fe_vars = ['log_dist', 'contig', 'comlang_off', col45_var, 'rta',
               'incentive_exporter', 'incentive_importer']

    print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
    print("-" * 60)
    for var in fe_vars:
        if var in m_fe.params:
            print(f"{var:<25} {m_fe.params[var]:>10.4f} {m_fe.bse[var]:>10.4f} "
                  f"{m_fe.pvalues[var]:>8.4f} "
                  f"{'***' if m_fe.pvalues[var] < 0.01 else '**' if m_fe.pvalues[var] < 0.05 else '*' if m_fe.pvalues[var] < 0.1 else '':>5}")

    # ----- IMPORTER + EXPORTER + YEAR FE WITH INCENTIVE TYPES -----
    print(f"\n{'='*70}")
    print(f"IMP+EXP+YEAR FE WITH INCENTIVE TYPES: {dep_label}")
    print(f"{'='*70}")

    type_dummy_vars = [f'{d}_exp' for d in ['is_refundable_credit', 'is_transferable_credit',
                                             'is_standard_credit', 'is_cash_rebate']]
    type_vars_in_sample = [v for v in type_dummy_vars if v in df.columns and df[v].nunique() > 1]

    formula_fe_types = (f'{dep_var} ~ log_dist + contig + comlang_off + {col45_var} + rta + '
                        + ' + '.join(type_vars_in_sample) + ' + incentive_importer + '
                        'C(importer) + C(exporter) + C(year)')

    m_fe_types = smf.ols(formula_fe_types, data=df).fit(cov_type='HC1')
    print(f"R²: {m_fe_types.rsquared:.4f}, N: {int(m_fe_types.nobs)}")

    fe_type_vars = ['log_dist', 'contig', 'comlang_off', col45_var, 'rta',
                    'incentive_importer'] + type_vars_in_sample

    print(f"\n{'Variable':<30} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
    print("-" * 65)
    for var in fe_type_vars:
        if var in m_fe_types.params:
            print(f"{var:<30} {m_fe_types.params[var]:>10.4f} {m_fe_types.bse[var]:>10.4f} "
                  f"{m_fe_types.pvalues[var]:>8.4f} "
                  f"{'***' if m_fe_types.pvalues[var] < 0.01 else '**' if m_fe_types.pvalues[var] < 0.05 else '*' if m_fe_types.pvalues[var] < 0.1 else '':>5}")

    # ----- COUNTRY-PAIR + YEAR FE (within transformation) -----
    pair_fe_results = {}

    time_varying = [
        'log_gdp_importer', 'log_gdp_exporter',
        'remoteness_importer', 'remoteness_exporter',
        'efw_importer', 'efw_exporter',
        'rta', 'incentive_exporter', 'incentive_importer'
    ]

    # Also define time-varying with type dummies (for the type-specific pair FE)
    type_dummy_vars_tv = [f'{d}_exp' for d in ['is_refundable_credit', 'is_transferable_credit',
                                                'is_standard_credit', 'is_cash_rebate']]

    # Time-varying with generosity instead of incentive dummy
    time_varying_gen = [v for v in time_varying if v != 'incentive_exporter'] + ['generosity_exp']

    # Run pair FE for multiple subsamples
    subsamples = {
        'Full sample': df.copy(),
        'Full (no EFW)': None,  # handled separately below
        'Excl Anglo-to-Anglo': df[~((df['exporter'].isin(ANGLO)) & (df['importer'].isin(ANGLO)))].copy(),
        'Excl all Anglo': df[~((df['exporter'].isin(ANGLO)) | (df['importer'].isin(ANGLO)))].copy(),
    }

    for sample_name, df_sub in subsamples.items():
        if sample_name == 'Full (no EFW)':
            # Use broader sample without EFW requirement
            est_no_efw = [v for v in [dep_var, 'log_gdp_importer', 'log_gdp_exporter',
                                       'log_dist', 'contig', 'comlang_off', col45_var,
                                       'rta', 'remoteness_importer', 'remoteness_exporter',
                                       'incentive_exporter', 'incentive_importer']
                          if v in merged_full.columns]
            df_sub = merged_full.dropna(subset=est_no_efw).copy()
            tv = [v for v in time_varying if v not in ['efw_importer', 'efw_exporter']]
        else:
            tv = time_varying

        if df_sub is None or len(df_sub) < 50:
            print(f"\nSkipping {sample_name} — too few observations ({len(df_sub) if df_sub is not None else 0})")
            continue

        print(f"\n{'='*70}")
        print(f"COUNTRY-PAIR + YEAR FE: {dep_label} — {sample_name}")
        print(f"{'='*70}")

        # Create pair id and drop singletons
        df_sub['pair_id'] = df_sub['exporter'] + '_' + df_sub['importer']
        pair_counts = df_sub.groupby('pair_id').size()
        pairs_keep = pair_counts[pair_counts > 1].index
        df_panel = df_sub[df_sub['pair_id'].isin(pairs_keep)].copy()

        print(f"Observations: {len(df_panel):,}, Pairs: {df_panel['pair_id'].nunique()}")

        # Incentive variation check
        for inc_var in ['incentive_exporter', 'incentive_importer']:
            inc_variation = df_panel.groupby('pair_id')[inc_var].nunique()
            n_varies = (inc_variation > 1).sum()
            print(f"  {inc_var} varies within pair: {n_varies} ({(inc_variation > 1).mean()*100:.1f}%)")

        # Within-transformation (demean by pair)
        demean_vars = [dep_var] + tv
        df_dm = df_panel.copy()
        pair_means = df_panel.groupby('pair_id')[demean_vars].transform('mean')
        for var in demean_vars:
            df_dm[f'{var}_dm'] = df_panel[var] - pair_means[var]

        dm_formula = (f'{dep_var}_dm ~ ' +
                      ' + '.join([f'{v}_dm' for v in tv]) +
                      ' + C(year) - 1')

        m_pair = smf.ols(dm_formula, data=df_dm).fit(cov_type='HC1')

        # Within R-squared
        ss_res = np.sum(m_pair.resid ** 2)
        ss_tot = np.sum(df_dm[f'{dep_var}_dm'] ** 2)
        within_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        print(f"\nWithin R²: {within_r2:.4f}, N: {int(m_pair.nobs)}")

        print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
        print("-" * 60)
        for var in tv:
            dm_var = f'{var}_dm'
            if dm_var in m_pair.params:
                coef = m_pair.params[dm_var]
                se = m_pair.bse[dm_var]
                p = m_pair.pvalues[dm_var]
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                print(f"{var:<25} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5}")

        pair_fe_results[sample_name] = m_pair

    # ----- COUNTRY-PAIR + YEAR FE WITH INCENTIVE TYPES (full sample only) -----
    print(f"\n{'='*70}")
    print(f"COUNTRY-PAIR + YEAR FE WITH INCENTIVE TYPES: {dep_label}")
    print(f"{'='*70}")

    # Use the full sample pair panel from earlier
    df_types = df.copy()
    df_types['pair_id'] = df_types['exporter'] + '_' + df_types['importer']
    pc = df_types.groupby('pair_id').size()
    df_types = df_types[df_types['pair_id'].isin(pc[pc > 1].index)].copy()

    # Time-varying vars: replace incentive_exporter with type dummies
    tv_types = [v for v in time_varying if v != 'incentive_exporter']
    type_vars_available = [v for v in type_dummy_vars_tv if v in df_types.columns and df_types[v].nunique() > 1]
    tv_types = tv_types + type_vars_available

    print(f"Observations: {len(df_types):,}, Pairs: {df_types['pair_id'].nunique()}")
    for tvar in type_vars_available:
        inc_var_check = df_types.groupby('pair_id')[tvar].nunique()
        n_var = (inc_var_check > 1).sum()
        print(f"  {tvar} varies within pair: {n_var} ({(inc_var_check > 1).mean()*100:.1f}%)")

    # Within-transformation
    demean_types = [dep_var] + tv_types
    df_dm_types = df_types.copy()
    pm_types = df_types.groupby('pair_id')[demean_types].transform('mean')
    for var in demean_types:
        df_dm_types[f'{var}_dm'] = df_types[var] - pm_types[var]

    dm_formula_types = (f'{dep_var}_dm ~ ' +
                        ' + '.join([f'{v}_dm' for v in tv_types]) +
                        ' + C(year) - 1')

    m_pair_types = smf.ols(dm_formula_types, data=df_dm_types).fit(cov_type='HC1')

    ss_r = np.sum(m_pair_types.resid ** 2)
    ss_t = np.sum(df_dm_types[f'{dep_var}_dm'] ** 2)
    w_r2 = 1 - ss_r / ss_t if ss_t > 0 else np.nan

    print(f"\nWithin R²: {w_r2:.4f}, N: {int(m_pair_types.nobs)}")

    print(f"\n{'Variable':<30} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
    print("-" * 65)
    for var in tv_types:
        dm_var = f'{var}_dm'
        if dm_var in m_pair_types.params:
            coef = m_pair_types.params[dm_var]
            se = m_pair_types.bse[dm_var]
            p = m_pair_types.pvalues[dm_var]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"{var:<30} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5}")

    pair_fe_results['Types (full sample)'] = m_pair_types

    # ----- COUNTRY-PAIR + YEAR FE WITH GENEROSITY (full sample only) -----
    if has_generosity and df['generosity_exp'].nunique() > 1:
        print(f"\n{'='*70}")
        print(f"COUNTRY-PAIR + YEAR FE WITH GENEROSITY: {dep_label}")
        print(f"{'='*70}")

        df_gen = df.copy()
        df_gen['pair_id'] = df_gen['exporter'] + '_' + df_gen['importer']
        pc_g = df_gen.groupby('pair_id').size()
        df_gen = df_gen[df_gen['pair_id'].isin(pc_g[pc_g > 1].index)].copy()

        print(f"Observations: {len(df_gen):,}, Pairs: {df_gen['pair_id'].nunique()}")

        # Check within-pair variation in generosity
        gen_var = df_gen.groupby('pair_id')['generosity_exp'].std()
        print(f"  generosity_exp has within-pair variation: {(gen_var > 0).sum()} pairs ({(gen_var > 0).mean()*100:.1f}%)")

        demean_gen = [dep_var] + time_varying_gen
        df_dm_gen = df_gen.copy()
        pm_gen = df_gen.groupby('pair_id')[demean_gen].transform('mean')
        for var in demean_gen:
            df_dm_gen[f'{var}_dm'] = df_gen[var] - pm_gen[var]

        dm_formula_gen = (f'{dep_var}_dm ~ ' +
                          ' + '.join([f'{v}_dm' for v in time_varying_gen]) +
                          ' + C(year) - 1')

        m_pair_gen = smf.ols(dm_formula_gen, data=df_dm_gen).fit(cov_type='HC1')

        ss_rg = np.sum(m_pair_gen.resid ** 2)
        ss_tg = np.sum(df_dm_gen[f'{dep_var}_dm'] ** 2)
        w_r2g = 1 - ss_rg / ss_tg if ss_tg > 0 else np.nan

        print(f"\nWithin R²: {w_r2g:.4f}, N: {int(m_pair_gen.nobs)}")

        print(f"\n{'Variable':<30} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
        print("-" * 65)
        for var in time_varying_gen:
            dm_var = f'{var}_dm'
            if dm_var in m_pair_gen.params:
                coef = m_pair_gen.params[dm_var]
                se = m_pair_gen.bse[dm_var]
                p = m_pair_gen.pvalues[dm_var]
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                print(f"{var:<30} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5}")

        # Interpret generosity coefficient
        if 'generosity_exp_dm' in m_pair_gen.params:
            gen_coef = m_pair_gen.params['generosity_exp_dm']
            print(f"\n  Interpretation: a 10 percentage point increase in incentive rate")
            print(f"  is associated with a {gen_coef * 0.10:.4f} change in log {dep_var.replace('log_', '')}")
            print(f"  ({(np.exp(gen_coef * 0.10) - 1)*100:.1f}% change)")

        pair_fe_results['Generosity (full sample)'] = m_pair_gen

    return m_fe, m_fe_types, pair_fe_results


# =============================================================================
# PHASE 9-10: ROBUSTNESS — FILM COUNT
# =============================================================================

m_fe_count, m_fe_types_count, pair_results_count = run_fe_robustness(df_count, 'log_num_films', 'FILM COUNT', merged_full=merged)

# =============================================================================
# PHASE 9b-10b: ROBUSTNESS — BUDGET (if enough data)
# =============================================================================

if budget_models is not None and len(df_budget) >= 50:
    m_fe_budget, m_fe_types_budget, pair_results_budget = run_fe_robustness(df_budget, 'log_budget', 'BUDGET', merged_full=merged)
else:
    m_fe_budget = None
    m_fe_types_budget = None
    pair_results_budget = {}

# =============================================================================
# PHASE 11: SUMMARY COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: INCENTIVE EFFECT ACROSS ALL SPECIFICATIONS")
print("=" * 70)


def extract_incentive_coef(model, var_name):
    """Extract coefficient, checking both original and demeaned names."""
    for lookup in [var_name, f'{var_name}_dm']:
        if lookup in model.params:
            coef = model.params[lookup]
            p = model.pvalues[lookup]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            pct = (np.exp(coef) - 1) * 100
            return f"{coef:>6.3f}{stars:<3}({pct:>+.0f}%)"
    return f"{'--':>14}"


# Helper to find a model by name from the models list
def get_model_by_name(models_list, name):
    """Find a model by its label in the (name, model) tuple list."""
    for mname, model in models_list:
        if mname == name:
            return model
    return None


# Film count results
print(f"\n--- FILM COUNT SPECIFICATION ---")
print(f"\n{'Specification':<35} {'inc_exporter':>15} {'inc_importer':>15}")
print("-" * 67)

# M7 from incremental
m7_count = get_model_by_name(count_models, 'M7')
print(f"{'M7 (Year FE)':<35}{extract_incentive_coef(m7_count, 'incentive_exporter'):>15}"
      f"{extract_incentive_coef(m7_count, 'incentive_importer'):>15}")

# Imp+Exp+Year FE
print(f"{'Imp+Exp+Year FE':<35}{extract_incentive_coef(m_fe_count, 'incentive_exporter'):>15}"
      f"{extract_incentive_coef(m_fe_count, 'incentive_importer'):>15}")

# Pair FE results
for sample_name, model in pair_results_count.items():
    label = f"Pair+Year FE ({sample_name})"
    if len(label) > 35:
        label = label[:35]
    print(f"{label:<35}{extract_incentive_coef(model, 'incentive_exporter'):>15}"
          f"{extract_incentive_coef(model, 'incentive_importer'):>15}")

# Budget results (if available)
if budget_models is not None:
    print(f"\n--- BUDGET SPECIFICATION ---")
    print(f"\n{'Specification':<35} {'inc_exporter':>15} {'inc_importer':>15}")
    print("-" * 67)

    m7_budget = get_model_by_name(budget_models, 'M7')
    print(f"{'M7 (Year FE)':<35}{extract_incentive_coef(m7_budget, 'incentive_exporter'):>15}"
          f"{extract_incentive_coef(m7_budget, 'incentive_importer'):>15}")

    if m_fe_budget is not None:
        print(f"{'Imp+Exp+Year FE':<35}{extract_incentive_coef(m_fe_budget, 'incentive_exporter'):>15}"
              f"{extract_incentive_coef(m_fe_budget, 'incentive_importer'):>15}")

    for sample_name, model in pair_results_budget.items():
        label = f"Pair+Year FE ({sample_name})"
        if len(label) > 35:
            label = label[:35]
        print(f"{label:<35}{extract_incentive_coef(model, 'incentive_exporter'):>15}"
              f"{extract_incentive_coef(model, 'incentive_importer'):>15}")

print(f"\n* p<0.1, ** p<0.05, *** p<0.01")
print(f"Percentage effects in parentheses: (exp(coef)-1)*100")

# --- INCENTIVE TYPE COMPARISON ---
print(f"\n\n{'='*70}")
print("INCENTIVE TYPE COMPARISON")
print("="*70)

type_vars_for_summary = ['is_refundable_credit_exp', 'is_transferable_credit_exp',
                         'is_standard_credit_exp', 'is_cash_rebate_exp']

for label, models_list, fe_types_model, pair_dict in [
    ('FILM COUNT', count_models, m_fe_types_count, pair_results_count),
    ('BUDGET', budget_models, m_fe_types_budget, pair_results_budget) if budget_models is not None else (None, None, None, None)
]:
    if label is None:
        continue

    print(f"\n--- {label} ---")
    print(f"\n{'Specification':<25}", end="")
    for tv in type_vars_for_summary:
        short = tv.replace('is_', '').replace('_exp', '').replace('_credit', '').replace('_', ' ')
        print(f"{short:>18}", end="")
    print()
    print("-" * (25 + 18 * len(type_vars_for_summary)))

    # M8 from incremental (by name, not position)
    m8 = get_model_by_name(models_list, 'M8')
    if m8 is not None:
        row = f"{'M8 (Year FE)':<25}"
        for tv in type_vars_for_summary:
            row += extract_incentive_coef(m8, tv).rjust(18)
        print(row)

    # Imp+Exp+Year FE types
    if fe_types_model is not None:
        row = f"{'Imp+Exp+Year FE (Types)':<25}"
        for tv in type_vars_for_summary:
            row += extract_incentive_coef(fe_types_model, tv).rjust(18)
        print(row)

    # Pair FE types
    if 'Types (full sample)' in pair_dict:
        m_types = pair_dict['Types (full sample)']
        row = f"{'Pair+Year FE (Types)':<25}"
        for tv in type_vars_for_summary:
            row += extract_incentive_coef(m_types, tv).rjust(18)
        print(row)

    # Generosity comparison
    print()
    print(f"{'Specification':<25} {'generosity':>18}")
    print("-" * 43)
    m9 = get_model_by_name(models_list, 'M9')
    if m9 is not None:
        print(f"{'M9 (Year FE)':<25}{extract_incentive_coef(m9, 'generosity_exp'):>18}")
    if 'Generosity (full sample)' in pair_dict:
        m_gen = pair_dict['Generosity (full sample)']
        print(f"{'Pair+Year FE (Generosity)':<25}{extract_incentive_coef(m_gen, 'generosity_exp'):>18}")

print(f"\n* p<0.1, ** p<0.05, *** p<0.01")

# =============================================================================
# PHASE 12: EVENT STUDY — EXPORTER INCENTIVE INTRODUCTION
# =============================================================================

print("\n\n" + "=" * 70)
print("PHASE 12: EVENT STUDY — FILMING FLOWS AROUND INCENTIVE INTRODUCTION")
print("=" * 70)

# Create years-relative-to-introduction variable for exporters
merged['intro_year_exp'] = merged['exporter'].map(incentive_dict)
merged['years_since_intro_exp'] = np.where(
    merged['intro_year_exp'].notna(),
    merged['year'] - merged['intro_year_exp'],
    np.nan
)

# For the event study, we need:
# 1. Only pairs where the exporter eventually gets an incentive
# 2. A window around the introduction (e.g., -5 to +10 years)
# 3. Bin the relative-time indicators

EVENT_WINDOW_MIN = -5
EVENT_WINDOW_MAX = 10

# Filter to pairs where exporter has an incentive intro date
df_event = df_count[df_count['exporter'].map(incentive_dict).notna()].copy()
df_event['intro_year_exp'] = df_event['exporter'].map(incentive_dict)
df_event['rel_year'] = (df_event['year'] - df_event['intro_year_exp']).astype(int)

# Trim to event window
df_event = df_event[
    (df_event['rel_year'] >= EVENT_WINDOW_MIN) &
    (df_event['rel_year'] <= EVENT_WINDOW_MAX)
].copy()

print(f"Event study sample: {len(df_event):,} observations")
print(f"Unique exporters with incentive intro: {df_event['exporter'].nunique()}")
print(f"Relative year range: {df_event['rel_year'].min()} to {df_event['rel_year'].max()}")

# Create relative-year dummies (omit t=-1 as reference)
for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1):
    if t == -1:  # reference period
        continue
    df_event[f'rel_t{t}'] = (df_event['rel_year'] == t).astype(int)

# Pair FE event study with within-transformation
df_event['pair_id'] = df_event['exporter'] + '_' + df_event['importer']
pair_counts_ev = df_event.groupby('pair_id').size()
df_event = df_event[df_event['pair_id'].isin(pair_counts_ev[pair_counts_ev > 1].index)].copy()

print(f"After dropping singletons: {len(df_event):,} obs, {df_event['pair_id'].nunique()} pairs")

# Variables: controls + relative-year dummies
controls_ev = ['log_gdp_importer', 'log_gdp_exporter',
               'remoteness_importer', 'remoteness_exporter',
               'efw_importer', 'efw_exporter', 'rta']
rel_year_vars = [f'rel_t{t}' for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1) if t != -1]

# Within-transformation (demean by pair)
all_ev_vars = ['log_num_films'] + controls_ev + rel_year_vars
df_ev_dm = df_event.copy()
pm_ev = df_event.groupby('pair_id')[all_ev_vars].transform('mean')
for var in all_ev_vars:
    df_ev_dm[f'{var}_dm'] = df_event[var] - pm_ev[var]

# Add year dummies (not demeaned — they handle common time effects)
year_dummies_ev = pd.get_dummies(df_ev_dm['year'], prefix='yr', drop_first=True, dtype=float)

# Build X matrix: demeaned controls + demeaned rel_year dummies + year dummies
dm_controls = [f'{v}_dm' for v in controls_ev]
dm_rel_years = [f'{v}_dm' for v in rel_year_vars]

X_ev = pd.concat([df_ev_dm[dm_controls + dm_rel_years], year_dummies_ev], axis=1)
y_ev = df_ev_dm['log_num_films_dm']

m_event = sm.OLS(y_ev, X_ev).fit(cov_type='HC1')

print(f"\nEvent Study Results (pair FE, reference = t-1):")
print(f"{'Rel Year':>10} {'Coef':>10} {'SE':>10} {'95% CI':>22} {'Sig':>5}")
print("-" * 60)

event_coefs = []
for t in range(EVENT_WINDOW_MIN, EVENT_WINDOW_MAX + 1):
    if t == -1:
        event_coefs.append((t, 0, 0, 0, 0))
        print(f"{'t-1':>10} {'0.000':>10} {'(ref)':>10} {'':>22} {'ref':>5}")
        continue

    dm_var = f'rel_t{t}_dm'
    if dm_var in m_event.params:
        coef = m_event.params[dm_var]
        se = m_event.bse[dm_var]
        p = m_event.pvalues[dm_var]
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        event_coefs.append((t, coef, se, ci_lo, ci_hi))
        print(f"{'t'+format(t, '+d'):>10} {coef:>10.4f} {se:>10.4f} [{ci_lo:>8.4f}, {ci_hi:>8.4f}] {stars:>5}")
    else:
        print(f"{'t'+format(t, '+d'):>10} {'(dropped)':>10}")

# Check pre-trends
pre_coefs = [c for t, c, s, lo, hi in event_coefs if EVENT_WINDOW_MIN <= t < -1]
post_coefs = [c for t, c, s, lo, hi in event_coefs if t >= 0]

if pre_coefs and post_coefs:
    avg_pre = np.mean(pre_coefs)
    avg_post = np.mean(post_coefs)
    print(f"\nPre-trend check:")
    print(f"  Average pre-treatment coefficient (t-5 to t-2): {avg_pre:.4f}")
    print(f"  Average post-treatment coefficient (t0 to t+10): {avg_post:.4f}")
    if abs(avg_pre) < 0.02:
        print(f"  Pre-trends appear flat — supports causal interpretation")
    else:
        print(f"  Pre-trends non-zero — exercise caution with causal claims")

# =============================================================================
# PHASE 13: CLUSTER EFFECTS — YEARS SINCE INTRODUCTION
# =============================================================================

print("\n\n" + "=" * 70)
print("PHASE 13: CLUSTER EFFECTS — DO INCENTIVE EFFECTS GROW OVER TIME?")
print("=" * 70)

# Create years_since_intro for exporter (0 in intro year, capped at sensible max)
MAX_YEARS = 20

merged['years_active_exp'] = np.where(
    (merged['incentive_exporter'] == 1) & merged['intro_year_exp'].notna(),
    np.minimum(merged['year'] - merged['intro_year_exp'], MAX_YEARS),
    0
)

# Use the count analysis sample
df_cluster = df_count.copy()
df_cluster['intro_year_exp'] = df_cluster['exporter'].map(incentive_dict)
df_cluster['years_active_exp'] = np.where(
    (df_cluster['incentive_exporter'] == 1) & df_cluster['intro_year_exp'].notna(),
    np.minimum(df_cluster['year'] - df_cluster['intro_year_exp'], MAX_YEARS),
    0
)

print(f"Years active distribution (when incentive active):")
active_years = df_cluster[df_cluster['years_active_exp'] > 0]['years_active_exp']
print(f"  Mean: {active_years.mean():.1f}")
print(f"  Median: {active_years.median():.1f}")
print(f"  Max: {active_years.max():.0f}")

col45_var = 'col45' if 'col45' in df_cluster.columns else 'col_dep_ever'

# Model A: Incentive dummy + years_active interaction (pooled with year FE)
print(f"\n--- Model A: Incentive + Years Active (Year FE) ---")
f_cluster_a = (f'log_num_films ~ log_gdp_importer + log_gdp_exporter + log_dist + '
               f'contig + comlang_off + {col45_var} + remoteness_importer + remoteness_exporter + '
               f'rta + efw_importer + efw_exporter + '
               f'incentive_exporter + years_active_exp + incentive_importer + C(year)')
m_cluster_a = smf.ols(f_cluster_a, data=df_cluster).fit(cov_type='HC1')

print(f"R²: {m_cluster_a.rsquared:.4f}, N: {int(m_cluster_a.nobs)}")
for var in ['incentive_exporter', 'years_active_exp', 'incentive_importer']:
    if var in m_cluster_a.params:
        coef = m_cluster_a.params[var]
        se = m_cluster_a.bse[var]
        p = m_cluster_a.pvalues[var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {var:<25} {coef:>8.4f} ({se:.4f}) {stars}")

print(f"\n  Interpretation: incentive_exporter = immediate effect on introduction")
print(f"  years_active_exp = additional effect per year the incentive has been active")
if 'years_active_exp' in m_cluster_a.params:
    ya_coef = m_cluster_a.params['years_active_exp']
    if ya_coef > 0:
        print(f"  Positive = effect grows over time (cluster formation)")
    else:
        print(f"  Negative/zero = no evidence of cluster formation")

# Model B: Pair FE with years_active
print(f"\n--- Model B: Pair FE + Years Active ---")

df_cl_panel = df_cluster.copy()
df_cl_panel['pair_id'] = df_cl_panel['exporter'] + '_' + df_cl_panel['importer']
pc_cl = df_cl_panel.groupby('pair_id').size()
df_cl_panel = df_cl_panel[df_cl_panel['pair_id'].isin(pc_cl[pc_cl > 1].index)].copy()

tv_cluster = ['log_gdp_importer', 'log_gdp_exporter',
              'remoteness_importer', 'remoteness_exporter',
              'efw_importer', 'efw_exporter', 'rta',
              'incentive_exporter', 'years_active_exp', 'incentive_importer']

demean_cl = ['log_num_films'] + tv_cluster
df_cl_dm = df_cl_panel.copy()
pm_cl = df_cl_panel.groupby('pair_id')[demean_cl].transform('mean')
for var in demean_cl:
    df_cl_dm[f'{var}_dm'] = df_cl_panel[var] - pm_cl[var]

cl_formula = ('log_num_films_dm ~ ' +
              ' + '.join([f'{v}_dm' for v in tv_cluster]) +
              ' + C(year) - 1')

m_cluster_b = smf.ols(cl_formula, data=df_cl_dm).fit(cov_type='HC1')

ss_cl = np.sum(m_cluster_b.resid ** 2)
ss_tl = np.sum(df_cl_dm['log_num_films_dm'] ** 2)
w_r2_cl = 1 - ss_cl / ss_tl if ss_tl > 0 else np.nan

print(f"Within R²: {w_r2_cl:.4f}, N: {int(m_cluster_b.nobs)}")
print(f"Pairs: {df_cl_panel['pair_id'].nunique()}")

for var in tv_cluster:
    dm_var = f'{var}_dm'
    if dm_var in m_cluster_b.params:
        coef = m_cluster_b.params[dm_var]
        se = m_cluster_b.bse[dm_var]
        p = m_cluster_b.pvalues[dm_var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {var:<25} {coef:>8.4f} ({se:.4f}) {stars}")

if 'years_active_exp_dm' in m_cluster_b.params:
    ya_coef_b = m_cluster_b.params['years_active_exp_dm']
    ya_p = m_cluster_b.pvalues['years_active_exp_dm']
    print(f"\n  Cluster effect test:")
    print(f"  years_active coefficient: {ya_coef_b:.4f} (p={ya_p:.4f})")
    if ya_p < 0.05 and ya_coef_b > 0:
        print(f"  Evidence of cluster formation: each additional year adds {(np.exp(ya_coef_b)-1)*100:.1f}% to filming flows")
    elif ya_p < 0.1 and ya_coef_b > 0:
        print(f"  Weak evidence of cluster formation (p<0.10)")
    else:
        print(f"  No significant evidence of cluster formation")
        print(f"  Consistent with Owens & Rennhoff (2020) finding at US state level")

# Model C: Pair FE with early vs late adopter split
print(f"\n--- Model C: Early vs Late Adopters (Pair FE) ---")

# Define early adopters: incentive introduced before 2005
EARLY_CUTOFF = 2005
df_cl_panel['early_adopter_exp'] = df_cl_panel.apply(
    lambda row: (1 if (row['exporter'] in incentive_dict
                       and pd.notna(incentive_dict[row['exporter']])
                       and incentive_dict[row['exporter']] < EARLY_CUTOFF
                       and row['year'] >= incentive_dict[row['exporter']])
                 else 0),
    axis=1
)
df_cl_panel['late_adopter_exp'] = df_cl_panel.apply(
    lambda row: (1 if (row['exporter'] in incentive_dict
                       and pd.notna(incentive_dict[row['exporter']])
                       and incentive_dict[row['exporter']] >= EARLY_CUTOFF
                       and row['year'] >= incentive_dict[row['exporter']])
                 else 0),
    axis=1
)

n_early = df_cl_panel['early_adopter_exp'].sum()
n_late = df_cl_panel['late_adopter_exp'].sum()
print(f"Early adopter obs (intro before {EARLY_CUTOFF}): {n_early}")
print(f"Late adopter obs (intro {EARLY_CUTOFF}+): {n_late}")

tv_adopter = ['log_gdp_importer', 'log_gdp_exporter',
              'remoteness_importer', 'remoteness_exporter',
              'efw_importer', 'efw_exporter', 'rta',
              'early_adopter_exp', 'late_adopter_exp', 'incentive_importer']

demean_ad = ['log_num_films'] + tv_adopter
df_ad_dm = df_cl_panel.copy()
pm_ad = df_cl_panel.groupby('pair_id')[demean_ad].transform('mean')
for var in demean_ad:
    df_ad_dm[f'{var}_dm'] = df_cl_panel[var] - pm_ad[var]

ad_formula = ('log_num_films_dm ~ ' +
              ' + '.join([f'{v}_dm' for v in tv_adopter]) +
              ' + C(year) - 1')

m_adopter = smf.ols(ad_formula, data=df_ad_dm).fit(cov_type='HC1')

for var in ['early_adopter_exp', 'late_adopter_exp', 'incentive_importer']:
    dm_var = f'{var}_dm'
    if dm_var in m_adopter.params:
        coef = m_adopter.params[dm_var]
        se = m_adopter.bse[dm_var]
        p = m_adopter.pvalues[dm_var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        pct = (np.exp(coef) - 1) * 100
        print(f"  {var:<25} {coef:>8.4f} ({se:.4f}) {stars}  [{pct:+.1f}%]")

print(f"\n  If early > late: suggests first-mover advantage / cluster effects")
print(f"  If early ≈ late: incentive effect stable regardless of timing")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)