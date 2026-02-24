"""robustness_checks.py"""
"""Program to test the robustness of the gravity model"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# =============================================================================
# ROBUSTNESS CHECK 1: FILM PROXY vs BaTIS SK2 (OTHER CULTURAL SERVICES)
# =============================================================================

print("=" * 70)
print("ROBUSTNESS CHECK 1: FILM PROXY vs BaTIS SK2 DATA")
print("(SK2 = Other personal, cultural, recreational services)")
print("=" * 70)

# Load datasets
batis = pd.read_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\raw\base\batis_sk2_exports.csv')
gravity = pd.read_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\processed\gravity_dataset_analysis.csv')

print(f"\nBaTIS SK2 rows: {len(batis):,}")
print(f"Gravity rows: {len(gravity):,}")

# --- Prepare BaTIS data ---
batis['Reporter'] = batis['Reporter'].astype(str)
batis['Partner'] = batis['Partner'].astype(str)

# Drop unallocated codes (888) and World (WL)
batis = batis[
    (batis['Reporter'] != '888') &
    (batis['Partner'] != 'WL') &
    (batis['Partner'] != '888')
]

# Rename columns to match gravity dataset
batis = batis.rename(columns={
    'Reporter': 'exporter',
    'Partner': 'importer',
    'Year': 'year',
    'Balanced_value': 'trade_value_batis'
})

print(f"\nBaTIS after cleaning: {len(batis):,} rows")
print(f"BaTIS year range: {batis['year'].min()} - {batis['year'].max()}")
print(f"Gravity year range: {gravity['year'].min()} - {gravity['year'].max()}")

# Filter gravity to overlapping years (2005-2018)
gravity_subset = gravity[(gravity['year'] >= 2005) & (gravity['year'] <= 2018)].copy()
print(f"Gravity observations 2005-2018: {len(gravity_subset):,}")

# Get country-pair-years that exist in BaTIS
batis_pairs = set(zip(batis['importer'], batis['exporter'], batis['year']))

# Filter gravity to pairs that exist in BaTIS
gravity_subset['pair_year'] = list(zip(gravity_subset['importer'], gravity_subset['exporter'], gravity_subset['year']))
gravity_matched = gravity_subset[gravity_subset['pair_year'].isin(batis_pairs)].copy()
gravity_matched = gravity_matched.drop(columns=['pair_year'])

print(f"Gravity observations matching BaTIS pairs: {len(gravity_matched):,}")

# Aggregate BaTIS to country-pair-year level (in case of duplicates)
batis_agg = batis.groupby(['importer', 'exporter', 'year']).agg(
    trade_value_batis=('trade_value_batis', 'sum')
).reset_index()

# Merge gravity with BaTIS data
merged = gravity_matched.merge(
    batis_agg[['importer', 'exporter', 'year', 'trade_value_batis']],
    on=['importer', 'exporter', 'year'],
    how='left'
)

# Create log trade value for BaTIS (add small constant for zeros)
merged['log_trade_batis'] = np.log(merged['trade_value_batis'] + 0.001)

print(f"\n--- MERGED DATASET ---")
print(f"Total observations: {len(merged):,}")
print(f"With BaTIS data: {merged['trade_value_batis'].notna().sum():,}")
print(f"Missing BaTIS data: {merged['trade_value_batis'].isna().sum():,}")

# Check trade value distribution
print(f"\n--- BaTIS SK2 TRADE VALUE STATS ---")
print(merged['trade_value_batis'].describe())

# Correlation between film proxy and BaTIS
valid_both = merged.dropna(subset=['trade_value_batis', 'trade_value'])
corr_levels = valid_both['trade_value'].corr(valid_both['trade_value_batis'])
corr_logs = valid_both['log_trade_real'].corr(valid_both['log_trade_batis'])
print(f"\n--- CORRELATION (where both have data) ---")
print(f"Observations with both measures: {len(valid_both):,}")
print(f"Correlation (levels): {corr_levels:.3f}")
print(f"Correlation (logs): {corr_logs:.3f}")

# --- Run gravity models ---
merged['const'] = 1

independent_vars = [
    'const',
    'log_gdp_importer',
    'log_gdp_exporter',
    'log_dist',
    'contig',
    'comlang_off',
    'rta',
    'remoteness_importer',
    'remoteness_exporter',
    'incentive_exporter'
]

# Filter to complete cases
df_complete = merged.dropna(subset=['log_trade_real', 'log_trade_batis'] + independent_vars)
print(f"\nComplete cases for regression: {len(df_complete):,}")

# Model 1: Film Proxy
print("\n" + "=" * 70)
print("MODEL 1: FILM PRODUCTION PROXY (your approach)")
print("=" * 70)

X = df_complete[independent_vars]
y_film = df_complete['log_trade_real']
model_film = sm.OLS(y_film, X).fit(cov_type='HC1')
print(model_film.summary())

# Model 2: BaTIS SK2 Data
print("\n" + "=" * 70)
print("MODEL 2: BaTIS SK2 (other personal, cultural, recreational services)")
print("=" * 70)

y_batis = df_complete['log_trade_batis']
model_batis = sm.OLS(y_batis, X).fit(cov_type='HC1')
print(model_batis.summary())

# --- Side-by-side comparison ---
print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON: FILM PROXY vs BaTIS SK2")
print("=" * 70)
print(f"\n{'Variable':<25} {'Film Proxy':>12} {'BaTIS SK2':>12} {'Difference':>12}")
print("-" * 63)

for var in independent_vars:
    coef_film = model_film.params[var]
    coef_batis = model_batis.params[var]
    diff = coef_batis - coef_film

    # Add significance stars
    stars_film = ''
    if model_film.pvalues[var] < 0.01: stars_film = '***'
    elif model_film.pvalues[var] < 0.05: stars_film = '**'
    elif model_film.pvalues[var] < 0.1: stars_film = '*'

    stars_batis = ''
    if model_batis.pvalues[var] < 0.01: stars_batis = '***'
    elif model_batis.pvalues[var] < 0.05: stars_batis = '**'
    elif model_batis.pvalues[var] < 0.1: stars_batis = '*'

    print(f"{var:<25} {coef_film:>9.3f}{stars_film:<3} {coef_batis:>9.3f}{stars_batis:<3} {diff:>+10.3f}")

print("-" * 63)
print(f"{'R-squared':<25} {model_film.rsquared:>12.3f} {model_batis.rsquared:>12.3f}")
print(f"{'Observations':<25} {int(model_film.nobs):>12} {int(model_batis.nobs):>12}")

# =============================================================================
# ROBUSTNESS CHECK 2: FILM PROXY vs BaTIS SK1 DATA
# =============================================================================

print("=" * 70)
print("ROBUSTNESS CHECK: FILM PROXY vs BaTIS SK1 DATA")
print("(SK1 = Audiovisual and related services)")
print("=" * 70)

# Load datasets
batis_sk1 = pd.read_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\raw\base\batis_sk1_exports.csv')
gravity = pd.read_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\processed\gravity_dataset_analysis.csv')

print(f"\nBaTIS SK1 rows: {len(batis_sk1):,}")
print(f"Gravity rows: {len(gravity):,}")

# --- Prepare BaTIS data ---
batis_sk1['Reporter'] = batis_sk1['Reporter'].astype(str)
batis_sk1['Partner'] = batis_sk1['Partner'].astype(str)

# Drop unallocated codes (888) and World (WL)
batis_sk1 = batis_sk1[
    (batis_sk1['Reporter'] != '888') &
    (batis_sk1['Partner'] != 'WL') &
    (batis_sk1['Partner'] != '888')
    ]

# Rename columns to match gravity dataset
batis_sk1 = batis_sk1.rename(columns={
    'Reporter': 'exporter',
    'Partner': 'importer',
    'Year': 'year',
    'Balanced_value': 'trade_value_batis'
})

print(f"\nBaTIS SK1 after cleaning: {len(batis_sk1):,} rows")
print(f"BaTIS year range: {batis_sk1['year'].min()} - {batis_sk1['year'].max()}")
print(f"Gravity year range: {gravity['year'].min()} - {gravity['year'].max()}")

# Filter gravity to overlapping years (2005-2018)
gravity_subset = gravity[(gravity['year'] >= 2005) & (gravity['year'] <= 2018)].copy()
print(f"Gravity observations 2005-2018: {len(gravity_subset):,}")

# Get country-pair-years that exist in BaTIS SK1
batis_sk1_pairs = set(zip(batis_sk1['importer'], batis_sk1['exporter'], batis_sk1['year']))

# Filter gravity to pairs that exist in BaTIS
gravity_subset['pair_year'] = list(zip(gravity_subset['importer'], gravity_subset['exporter'], gravity_subset['year']))
gravity_matched = gravity_subset[gravity_subset['pair_year'].isin(batis_sk1_pairs)].copy()
gravity_matched = gravity_matched.drop(columns=['pair_year'])

print(f"Gravity observations matching BaTIS SK1 pairs: {len(gravity_matched):,}")

# Aggregate BaTIS SK1 to country-pair-year level (in case of duplicates)
batis_sk1_agg = batis_sk1.groupby(['importer', 'exporter', 'year']).agg(
    trade_value_batis=('trade_value_batis', 'sum')
).reset_index()

# Merge gravity with BaTIS data
merged = gravity_matched.merge(
    batis_sk1_agg[['importer', 'exporter', 'year', 'trade_value_batis']],
    on=['importer', 'exporter', 'year'],
    how='left'
)

# Create log trade value for BaTIS (add small constant for zeros)
merged['log_trade_batis'] = np.log(merged['trade_value_batis'] + 0.001)

print(f"\n--- MERGED DATASET ---")
print(f"Total observations: {len(merged):,}")
print(f"With BaTIS data: {merged['trade_value_batis'].notna().sum():,}")
print(f"Missing BaTIS data: {merged['trade_value_batis'].isna().sum():,}")

# Check trade value distribution
print(f"\n--- BaTIS SK1 TRADE VALUE STATS ---")
print(merged['trade_value_batis'].describe())

# Correlation between film proxy and BaTIS
valid_both = merged.dropna(subset=['trade_value_batis', 'trade_value'])
corr_levels = valid_both['trade_value'].corr(valid_both['trade_value_batis'])
corr_logs = valid_both['log_trade_real'].corr(valid_both['log_trade_batis'])
print(f"\n--- CORRELATION (where both have data) ---")
print(f"Observations with both measures: {len(valid_both):,}")
print(f"Correlation (levels): {corr_levels:.3f}")
print(f"Correlation (logs): {corr_logs:.3f}")

# --- Run gravity models ---
merged['const'] = 1

independent_vars = [
    'const',
    'log_gdp_importer',
    'log_gdp_exporter',
    'log_dist',
    'contig',
    'comlang_off',
    'rta',
    'remoteness_importer',
    'remoteness_exporter',
    'incentive_exporter'
]

# Filter to complete cases
df_complete = merged.dropna(subset=['log_trade_real', 'log_trade_batis'] + independent_vars[1:])
print(f"\nComplete cases for regression: {len(df_complete):,}")

# Model 1: Film Proxy
print("\n" + "=" * 70)
print("MODEL 1: FILM PRODUCTION PROXY (your approach)")
print("=" * 70)

X = df_complete[independent_vars]
y_film = df_complete['log_trade_real']
model_film = sm.OLS(y_film, X).fit(cov_type='HC1')
print(model_film.summary())

# Model 2: BaTIS SK1 Data
print("\n" + "=" * 70)
print("MODEL 2: BaTIS SK1 (audiovisual and related services)")
print("=" * 70)

y_batis = df_complete['log_trade_batis']
model_batis = sm.OLS(y_batis, X).fit(cov_type='HC1')
print(model_batis.summary())

# --- Side-by-side comparison ---
print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON: FILM PROXY vs BaTIS SK1")
print("=" * 70)
print(f"\n{'Variable':<25} {'Film Proxy':>12} {'BaTIS SK1':>12} {'Difference':>12}")
print("-" * 63)

for var in independent_vars:
    coef_film = model_film.params[var]
    coef_batis = model_batis.params[var]
    diff = coef_batis - coef_film

    # Add significance stars
    stars_film = ''
    if model_film.pvalues[var] < 0.01:
        stars_film = '***'
    elif model_film.pvalues[var] < 0.05:
        stars_film = '**'
    elif model_film.pvalues[var] < 0.1:
        stars_film = '*'

    stars_batis = ''
    if model_batis.pvalues[var] < 0.01:
        stars_batis = '***'
    elif model_batis.pvalues[var] < 0.05:
        stars_batis = '**'
    elif model_batis.pvalues[var] < 0.1:
        stars_batis = '*'

    print(f"{var:<25} {coef_film:>9.3f}{stars_film:<3} {coef_batis:>9.3f}{stars_batis:<3} {diff:>+10.3f}")

print("-" * 63)
print(f"{'R-squared':<25} {model_film.rsquared:>12.3f} {model_batis.rsquared:>12.3f}")
print(f"{'Observations':<25} {int(model_film.nobs):>12} {int(model_batis.nobs):>12}")

# =============================================================================
# ROBUSTNESS CHECK 3: HIGH CONFIDENCE vs ALL MAPPINGS
# =============================================================================

print("\n\n")
print("=" * 70)
print("ROBUSTNESS CHECK 3: HIGH CONFIDENCE vs ALL STUDIO MAPPINGS")
print("=" * 70)

# Load the film data
films = pd.read_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\processed\films_cleaned.csv')

print(f"\nTotal films: {len(films):,}")

# Check mapping confidence distribution in films
print(f"\nMapping confidence in film data:")
print(films['mapping_confidence'].value_counts())

# Create high-confidence-only dataset
films_high_conf = films[films['mapping_confidence'] == 'high'].copy()
print(f"\nFilms with high confidence mapping: {len(films_high_conf):,}")
print(f"Films with low confidence mapping: {len(films[films['mapping_confidence'] == 'low']):,}")

# Load the full gravity dataset
gravity_full = pd.read_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\processed\gravity_dataset_analysis.csv')

# --- Rebuild trade flows using HIGH CONFIDENCE ONLY ---
films_high_conf['year'] = pd.to_datetime(films_high_conf['release_date']).dt.year

# Build trade flows: for each film, create flows from home_country to each production country
trade_flows_high = []

for _, film in films_high_conf.iterrows():
    home = film['home_country']
    year = film['year']
    budget = film['budget'] if pd.notna(film['budget']) and film['budget'] > 0 else np.nan

    # Parse production countries
    try:
        countries = eval(film['country_codes']) if isinstance(film['country_codes'], str) else film['country_codes']
    except:
        continue

    if not isinstance(countries, list):
        continue

    # Create flows to each production country (excluding home country)
    for dest in countries:
        if dest != home and pd.notna(home):
            trade_flows_high.append({
                'year': year,
                'exporter': home,
                'importer': dest,
                'budget': budget
            })

trade_df_high = pd.DataFrame(trade_flows_high)
print(f"\nTrade flow observations (high conf): {len(trade_df_high):,}")

# Aggregate to country-pair-year level
agg_high = trade_df_high.groupby(['year', 'importer', 'exporter']).agg(
    num_films=('budget', 'count'),
    trade_value=('budget', 'sum')
).reset_index()

print(f"Country-pair-year observations (high conf): {len(agg_high):,}")

# Merge with gravity variables (from the full dataset)
gravity_vars = gravity_full[['year', 'importer', 'exporter',
                              'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
                              'contig', 'comlang_off', 'rta',
                              'remoteness_importer', 'remoteness_exporter',
                              'incentive_exporter']].drop_duplicates()

merged_high = agg_high.merge(gravity_vars, on=['year', 'importer', 'exporter'], how='inner')
merged_high['log_trade_real'] = np.log(merged_high['trade_value'])

print(f"Merged observations (high conf): {len(merged_high):,}")

# Also prepare the full dataset
gravity_all = gravity_full.copy()

# Filter both to same year range for fair comparison
year_min = max(merged_high['year'].min(), gravity_all['year'].min())
year_max = min(merged_high['year'].max(), gravity_all['year'].max())

merged_high_filtered = merged_high[(merged_high['year'] >= year_min) & (merged_high['year'] <= year_max)].copy()
gravity_all_filtered = gravity_all[(gravity_all['year'] >= year_min) & (gravity_all['year'] <= year_max)].copy()

print(f"\nYear range for comparison: {year_min} - {year_max}")
print(f"High confidence observations: {len(merged_high_filtered):,}")
print(f"All mappings observations: {len(gravity_all_filtered):,}")

# --- Create year dummies for both datasets ---
all_years = sorted(set(merged_high_filtered['year'].unique()) | set(gravity_all_filtered['year'].unique()))
base_year = min(all_years)  # drop first year as reference

for yr in all_years:
    if yr != base_year:
        merged_high_filtered[f'yr_{yr}'] = (merged_high_filtered['year'] == yr).astype(int)
        gravity_all_filtered[f'yr_{yr}'] = (gravity_all_filtered['year'] == yr).astype(int)

year_dummy_cols = [f'yr_{yr}' for yr in all_years if yr != base_year]

# --- Define variables ---
base_vars = [
    'log_gdp_importer',
    'log_gdp_exporter',
    'log_dist',
    'contig',
    'comlang_off',
    'rta',
    'remoteness_importer',
    'remoteness_exporter',
    'incentive_exporter'
]

# Add constant
merged_high_filtered['const'] = 1
gravity_all_filtered['const'] = 1

# Only use year dummies that exist in both datasets
common_year_cols = [col for col in year_dummy_cols if
                    col in merged_high_filtered.columns and col in gravity_all_filtered.columns]

independent_vars = ['const'] + base_vars + common_year_cols

# Filter to complete cases
df_high = merged_high_filtered.dropna(subset=['log_trade_real'] + base_vars)
df_high = df_high[~np.isinf(df_high['log_trade_real'])]

df_all = gravity_all_filtered.dropna(subset=['log_trade_real'] + base_vars)

print(f"\nComplete cases (high conf): {len(df_high):,}")
print(f"Complete cases (all mappings): {len(df_all):,}")

# Model 1: All mappings (with year FE)
print("\n" + "=" * 70)
print("MODEL 1: ALL STUDIO MAPPINGS (with year fixed effects)")
print("=" * 70)

X_all = df_all[independent_vars]
y_all = df_all['log_trade_real']
model_all = sm.OLS(y_all, X_all).fit(cov_type='HC1')

print(f"R-squared: {model_all.rsquared:.4f}")
print(f"Observations: {int(model_all.nobs)}")
print("\nKey coefficients (year dummies suppressed):")
for var in ['const'] + base_vars:
    coef = model_all.params[var]
    se = model_all.bse[var]
    pval = model_all.pvalues[var]
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    print(f"  {var:<25} {coef:>7.4f} ({se:.4f}) {stars}")

# Model 2: High confidence only (with year FE)
print("\n" + "=" * 70)
print("MODEL 2: HIGH CONFIDENCE MAPPINGS ONLY (with year fixed effects)")
print("=" * 70)

X_high = df_high[independent_vars]
y_high = df_high['log_trade_real']
model_high = sm.OLS(y_high, X_high).fit(cov_type='HC1')

print(f"R-squared: {model_high.rsquared:.4f}")
print(f"Observations: {int(model_high.nobs)}")
print("\nKey coefficients (year dummies suppressed):")
for var in ['const'] + base_vars:
    coef = model_high.params[var]
    se = model_high.bse[var]
    pval = model_high.pvalues[var]
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    print(f"  {var:<25} {coef:>7.4f} ({se:.4f}) {stars}")

# --- Side-by-side comparison ---
print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON: ALL MAPPINGS vs HIGH CONFIDENCE ONLY")
print("=" * 70)
print(f"\n{'Variable':<25} {'All Mappings':>12} {'High Conf':>12} {'Difference':>12}")
print("-" * 63)

for var in base_vars:
    coef_all = model_all.params[var]
    coef_high = model_high.params[var]
    diff = coef_high - coef_all

    # Add significance stars
    stars_all = ''
    if model_all.pvalues[var] < 0.01:
        stars_all = '***'
    elif model_all.pvalues[var] < 0.05:
        stars_all = '**'
    elif model_all.pvalues[var] < 0.1:
        stars_all = '*'

    stars_high = ''
    if model_high.pvalues[var] < 0.01:
        stars_high = '***'
    elif model_high.pvalues[var] < 0.05:
        stars_high = '**'
    elif model_high.pvalues[var] < 0.1:
        stars_high = '*'

    print(f"{var:<25} {coef_all:>9.3f}{stars_all:<3} {coef_high:>9.3f}{stars_high:<3} {diff:>+10.3f}")

print("-" * 63)
print(f"{'R-squared':<25} {model_all.rsquared:>12.3f} {model_high.rsquared:>12.3f}")
print(f"{'Observations':<25} {int(model_all.nobs):>12} {int(model_high.nobs):>12}")

# =============================================================================
# ROBUSTNESS CHECK 4: IMPORTER/EXPORTER FIXED EFFECTS
# =============================================================================

print("\n\n")
print("=" * 70)
print("ROBUSTNESS CHECK 4: IMPORTER/EXPORTER FIXED EFFECTS")
print("=" * 70)
print("""
This specification replaces GDP and remoteness variables with country 
fixed effects, following modern gravity model best practices (Head & Mayer, 2014).
Fixed effects absorb all time-invariant country characteristics.
""")

# Reload fresh gravity data
gravity_fe = pd.read_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\processed\gravity_dataset_analysis.csv')

print(f"Observations loaded: {len(gravity_fe):,}")

# Create importer and exporter dummies
importer_dummies = pd.get_dummies(gravity_fe['importer'], prefix='imp', drop_first=True, dtype=float)
exporter_dummies = pd.get_dummies(gravity_fe['exporter'], prefix='exp', drop_first=True, dtype=float)
year_dummies_fe = pd.get_dummies(gravity_fe['year'], prefix='year', drop_first=True, dtype=float)

print(f"Unique importers: {gravity_fe['importer'].nunique()} ({len(importer_dummies.columns)} dummies)")
print(f"Unique exporters: {gravity_fe['exporter'].nunique()} ({len(exporter_dummies.columns)} dummies)")
print(f"Unique years: {gravity_fe['year'].nunique()} ({len(year_dummies_fe.columns)} dummies)")

# Variables that vary at the PAIR level (not absorbed by country FE)
pair_vars = [
    'log_dist',
    'contig',
    'comlang_off',
    'col_dep_ever',
    'rta',
    'incentive_exporter'
]

# Check for missing values
df_fe = gravity_fe.dropna(subset=['log_trade_real'] + pair_vars)
print(f"Complete cases: {len(df_fe):,}")

# Rebuild dummies for filtered data
importer_dummies_fe = pd.get_dummies(df_fe['importer'], prefix='imp', drop_first=True, dtype=float)
exporter_dummies_fe = pd.get_dummies(df_fe['exporter'], prefix='exp', drop_first=True, dtype=float)
year_dummies_fe = pd.get_dummies(df_fe['year'], prefix='year', drop_first=True, dtype=float)

# --- MODEL A: Main specification (with GDP + remoteness, no country FE) ---
print("\n" + "=" * 70)
print("MODEL A: STANDARD SPECIFICATION (GDP + Remoteness + Year FE)")
print("=" * 70)

df_fe['log_remoteness_imp'] = np.log(df_fe['remoteness_importer'])
df_fe['log_remoteness_exp'] = np.log(df_fe['remoteness_exporter'])

standard_vars = [
    'log_gdp_importer',
    'log_gdp_exporter',
    'log_dist',
    'contig',
    'comlang_off',
    'col_dep_ever',
    'log_remoteness_imp',
    'log_remoteness_exp',
    'rta',
    'incentive_exporter'
]

df_standard = df_fe.dropna(subset=standard_vars)
year_dummies_std = pd.get_dummies(df_standard['year'], prefix='year', drop_first=True, dtype=float)

X_standard = sm.add_constant(pd.concat([
    df_standard[standard_vars].reset_index(drop=True),
    year_dummies_std.reset_index(drop=True)
], axis=1))
y_standard = df_standard['log_trade_real'].reset_index(drop=True)

model_standard = sm.OLS(y_standard, X_standard).fit(cov_type='HC1')

print(f"R-squared: {model_standard.rsquared:.4f}")
print(f"Observations: {int(model_standard.nobs)}")
print("\nKey coefficients (year dummies suppressed):")
for var in standard_vars:
    if var in model_standard.params:
        coef = model_standard.params[var]
        se = model_standard.bse[var]
        pval = model_standard.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {var:<25} {coef:>10.4f} ({se:.4f}) {stars}")

# --- MODEL B: Importer + Exporter + Year Fixed Effects ---
print("\n" + "=" * 70)
print("MODEL B: IMPORTER + EXPORTER + YEAR FIXED EFFECTS")
print("=" * 70)
print("(GDP and remoteness absorbed by country fixed effects)")

# Rebuild dummies for the standard sample (to ensure same observations)
importer_dummies_b = pd.get_dummies(df_standard['importer'], prefix='imp', drop_first=True, dtype=float)
exporter_dummies_b = pd.get_dummies(df_standard['exporter'], prefix='exp', drop_first=True, dtype=float)

X_fe = sm.add_constant(pd.concat([
    df_standard[pair_vars].reset_index(drop=True),
    importer_dummies_b.reset_index(drop=True),
    exporter_dummies_b.reset_index(drop=True),
    year_dummies_std.reset_index(drop=True)
], axis=1))
y_fe = df_standard['log_trade_real'].reset_index(drop=True)

model_fe = sm.OLS(y_fe, X_fe).fit(cov_type='HC1')

print(f"R-squared: {model_fe.rsquared:.4f}")
print(f"Observations: {int(model_fe.nobs)}")
print(f"Number of parameters: {len(model_fe.params)}")
print("\nKey coefficients (country and year dummies suppressed):")
for var in pair_vars:
    if var in model_fe.params:
        coef = model_fe.params[var]
        se = model_fe.bse[var]
        pval = model_fe.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {var:<25} {coef:>10.4f} ({se:.4f}) {stars}")

# --- Side-by-side comparison ---
print("\n" + "=" * 70)
print("COMPARISON: STANDARD vs FIXED EFFECTS SPECIFICATION")
print("=" * 70)

comparison_vars = ['log_dist', 'contig', 'comlang_off', 'col_dep_ever', 'rta', 'incentive_exporter']

print(f"\n{'Variable':<25} {'Standard':>12} {'Fixed Effects':>14} {'Difference':>12}")
print("-" * 65)

for var in comparison_vars:
    coef_std = model_standard.params.get(var, np.nan)
    coef_fe = model_fe.params.get(var, np.nan)

    if pd.notna(coef_std) and pd.notna(coef_fe):
        diff = coef_fe - coef_std

        # Stars for standard
        pval_std = model_standard.pvalues.get(var, 1)
        stars_std = '***' if pval_std < 0.01 else '**' if pval_std < 0.05 else '*' if pval_std < 0.1 else ''

        # Stars for FE
        pval_fe = model_fe.pvalues.get(var, 1)
        stars_fe = '***' if pval_fe < 0.01 else '**' if pval_fe < 0.05 else '*' if pval_fe < 0.1 else ''

        print(f"{var:<25} {coef_std:>9.3f}{stars_std:<3} {coef_fe:>11.3f}{stars_fe:<3} {diff:>+10.3f}")

print("-" * 65)
print(f"{'R-squared':<25} {model_standard.rsquared:>12.3f} {model_fe.rsquared:>14.3f}")
print(f"{'Observations':<25} {int(model_standard.nobs):>12} {int(model_fe.nobs):>14}")

# --- Key finding ---
print("\n" + "=" * 70)
print("KEY FINDING: INCENTIVE EFFECT ROBUSTNESS")
print("=" * 70)

coef_std = model_standard.params['incentive_exporter']
se_std = model_standard.bse['incentive_exporter']
pval_std = model_standard.pvalues['incentive_exporter']
pct_std = (np.exp(coef_std) - 1) * 100

coef_fe = model_fe.params['incentive_exporter']
se_fe = model_fe.bse['incentive_exporter']
pval_fe = model_fe.pvalues['incentive_exporter']
pct_fe = (np.exp(coef_fe) - 1) * 100

print(f"""
Standard specification:
  Coefficient: {coef_std:.4f} (SE: {se_std:.4f})
  Percentage effect: {pct_std:.1f}%
  p-value: {pval_std:.4f}

Fixed effects specification:
  Coefficient: {coef_fe:.4f} (SE: {se_fe:.4f})
  Percentage effect: {pct_fe:.1f}%
  p-value: {pval_fe:.4f}

Interpretation:
  The incentive effect {'remains' if (pval_std < 0.1 and pval_fe < 0.1) else 'changes'} 
  {'positive and significant' if (coef_fe > 0 and pval_fe < 0.1) else 'insignificant'} 
  when controlling for all time-invariant country characteristics.
""")