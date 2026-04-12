"""batis_gravity_analysis.py"""
"""
Standalone gravity model analysis using BaTIS SK1 (audiovisual services) data.
Covers 2005-2024 to capture more recent film incentive introductions.

Pipeline:
  1. Load and clean BaTIS SK1 export data
  2. Merge with CEPII gravity variables (distance, language, colonial ties, RTA)
  3. Merge with World Bank GDP
  4. Merge with EFW (Economic Freedom of the World)
  5. Merge with film incentive dates
  6. Calculate remoteness measures
  7. Estimate gravity models (baseline through full specification)
  8. Run robustness checks (importer+exporter FE, country-pair FE)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'
BATIS_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\batis_sk1_all_years.csv'
CEPII_FILE = f'{PROJECT_DIR}\\data\\processed\\gravity_vars_cepii.csv'
EFW_FILE = f'{PROJECT_DIR}\\data\\processed\\efw_cleaned.csv'
INCENTIVE_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\film_incentive_intro_dates.csv'
OUTPUT_DIR = f'{PROJECT_DIR}\\output'

YEAR_MIN = 2005
YEAR_MAX = 2023  # World Bank GDP not yet available for 2024

# =============================================================================
# PHASE 1: LOAD AND CLEAN BaTIS DATA
# =============================================================================

print("=" * 70)
print("PHASE 1: LOAD AND CLEAN BaTIS SK1 DATA")
print("=" * 70)

batis = pd.read_csv(BATIS_FILE)
print(f"Loaded BaTIS SK1 exports: {len(batis):,} rows")
print(f"Year range: {batis['Year'].min()} - {batis['Year'].max()}")

# Convert codes to string
batis['Reporter'] = batis['Reporter'].astype(str)
batis['Partner'] = batis['Partner'].astype(str)

# Drop aggregate codes
batis = batis[
    (batis['Reporter'] != '888') &
    (batis['Partner'] != '888') &
    (batis['Partner'] != 'WL')
]

# Drop zero and negative trade values
batis = batis[batis['Balanced_value'] > 0].copy()

# Create log trade variable
batis['log_trade'] = np.log(batis['Balanced_value'])

# Rename to match gravity convention
# In BaTIS: Reporter = exporter (country selling services)
#           Partner = importer (country buying services)
batis = batis.rename(columns={
    'Reporter': 'exporter',
    'Partner': 'importer',
    'Year': 'year',
    'Balanced_value': 'trade_value'
})

print(f"After cleaning: {len(batis):,} rows")
print(f"Unique exporters: {batis['exporter'].nunique()}")
print(f"Unique importers: {batis['importer'].nunique()}")
print(f"Year range: {batis['year'].min()} - {batis['year'].max()}")

print(f"\nTrade value statistics:")
print(batis['trade_value'].describe())

# =============================================================================
# PHASE 2: MERGE WITH CEPII GRAVITY VARIABLES
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: MERGE WITH CEPII GRAVITY VARIABLES")
print("=" * 70)

cepii = pd.read_csv(CEPII_FILE)
print(f"Loaded CEPII: {len(cepii):,} rows, years {cepii['year'].min()}-{cepii['year'].max()}")

# CEPII uses iso_o/iso_d, BaTIS uses exporter/importer
# For years beyond CEPII coverage (2021-2024), forward-fill from 2020
cepii_max_year = cepii['year'].max()
print(f"CEPII max year: {cepii_max_year}")

if YEAR_MAX > cepii_max_year:
    print(f"Forward-filling CEPII data for {cepii_max_year + 1}-{YEAR_MAX}...")
    cepii_latest = cepii[cepii['year'] == cepii_max_year].copy()
    new_rows = []
    for fill_year in range(cepii_max_year + 1, YEAR_MAX + 1):
        temp = cepii_latest.copy()
        temp['year'] = fill_year
        new_rows.append(temp)
    cepii = pd.concat([cepii] + new_rows, ignore_index=True)
    print(f"CEPII now covers: {cepii['year'].min()}-{cepii['year'].max()}")

# Merge: BaTIS exporter = CEPII iso_o, BaTIS importer = CEPII iso_d
merged = batis.merge(
    cepii,
    left_on=['exporter', 'importer', 'year'],
    right_on=['iso_o', 'iso_d', 'year'],
    how='left'
)

# Drop the duplicate iso columns
merged = merged.drop(columns=['iso_o', 'iso_d'], errors='ignore')

print(f"\nAfter CEPII merge: {len(merged):,} rows")
print(f"Missing distance: {merged['log_dist'].isna().sum()} ({merged['log_dist'].isna().mean()*100:.1f}%)")
print(f"Missing RTA: {merged['rta'].isna().sum()} ({merged['rta'].isna().mean()*100:.1f}%)")
print(f"Missing comlang_off: {merged['comlang_off'].isna().sum()} ({merged['comlang_off'].isna().mean()*100:.1f}%)")

# =============================================================================
# PHASE 3: MERGE WITH GDP (World Bank)
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: MERGE WITH GDP")
print("=" * 70)

# Load existing cleaned GDP data
gdp_file = f'{PROJECT_DIR}\\data\\processed\\gdp_cleaned.csv'
gdp = pd.read_csv(gdp_file)
print(f"Loaded GDP data: {len(gdp):,} rows, years {gdp['date'].min()}-{gdp['date'].max()}")

# Filter BaTIS data to years where we have GDP
merged = merged[merged['year'] <= YEAR_MAX].copy()
print(f"Filtered to years with GDP data (up to {YEAR_MAX}): {len(merged):,} rows")

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

# Forward-fill EFW if needed (EFW may not go to 2024)
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

# Create a lookup: country -> year incentive was introduced
# The column name may vary; adjust as needed
incentive_col = [c for c in incentives.columns if 'year' in c.lower() or 'intro' in c.lower()]
iso_col = [c for c in incentives.columns if 'iso' in c.lower() or 'code' in c.lower() or 'country' in c.lower()]

print(f"Using incentive year column: {incentive_col}")
print(f"Using country code column: {iso_col}")

# Build a dictionary: country_code -> introduction_year
incentive_dict = dict(zip(
    incentives[iso_col[0]].astype(str),
    incentives[incentive_col[0]]
))

# Create time-varying incentive dummies
merged['incentive_exporter'] = merged.apply(
    lambda row: 1 if row['exporter'] in incentive_dict and row['year'] >= incentive_dict[row['exporter']] else 0,
    axis=1
)
merged['incentive_importer'] = merged.apply(
    lambda row: 1 if row['importer'] in incentive_dict and row['year'] >= incentive_dict[row['importer']] else 0,
    axis=1
)

print(f"\nIncentive coverage:")
print(f"  Exporter incentive active: {merged['incentive_exporter'].sum()} ({merged['incentive_exporter'].mean()*100:.1f}%)")
print(f"  Importer incentive active: {merged['incentive_importer'].sum()} ({merged['incentive_importer'].mean()*100:.1f}%)")

# =============================================================================
# PHASE 6: CALCULATE REMOTENESS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 6: CALCULATE REMOTENESS MEASURES")
print("=" * 70)

# Remoteness = Kimura & Lee method (matching calculate_remoteness.py)
# remoteness_i = log(1 / sum_j(gdp_share_j / dist_ij))
# where gdp_share_j = GDP_j / GDP_world

# We need GDP in levels (not logs) for this
gdp_levels = gdp[['country', 'date', 'gdp']].rename(columns={'date': 'year'}).copy()
gdp_levels = gdp_levels.dropna(subset=['gdp'])

# Calculate world GDP by year
world_gdp = gdp_levels.groupby('year')['gdp'].sum().reset_index()
world_gdp.columns = ['year', 'gdp_world']

# Get distance data (time-invariant, so just use one year)
dist_year = cepii[cepii['year'] <= 2020]['year'].max()
dist_data = cepii[cepii['year'] == dist_year][['iso_o', 'iso_d', 'dist']].copy()
dist_data = dist_data.dropna(subset=['dist'])
dist_data = dist_data[dist_data['dist'] > 0]
# Remove self-pairs
dist_data = dist_data[dist_data['iso_o'] != dist_data['iso_d']]

# Calculate remoteness for each country-year
remoteness_records = []

years = sorted(merged['year'].unique())
countries = sorted(set(merged['exporter'].unique()) | set(merged['importer'].unique()))

for yr in years:
    # Get GDP data and world GDP for this year
    gdp_yr = gdp_levels[gdp_levels['year'] == yr].set_index('country')['gdp'].to_dict()
    gdp_world_yr = world_gdp[world_gdp['year'] == yr]['gdp_world'].values
    if len(gdp_world_yr) == 0:
        continue
    gdp_world_yr = gdp_world_yr[0]

    for country in countries:
        # Get distances from this country to all others
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
            remoteness = np.log(1 / weighted_sum)
            remoteness_records.append({
                'country': country,
                'year': yr,
                'remoteness': remoteness
            })

    if yr % 5 == 0:
        print(f"  Processed remoteness for {yr}")

remoteness_df = pd.DataFrame(remoteness_records)
print(f"Remoteness calculated: {len(remoteness_df):,} country-year observations")

# Merge remoteness
merged = merged.merge(
    remoteness_df.rename(columns={'country': 'importer', 'remoteness': 'remoteness_importer'}),
    on=['importer', 'year'],
    how='left'
)
merged = merged.merge(
    remoteness_df.rename(columns={'country': 'exporter', 'remoteness': 'remoteness_exporter'}),
    on=['exporter', 'year'],
    how='left'
)

print(f"Missing importer remoteness: {merged['remoteness_importer'].isna().sum()}")
print(f"Missing exporter remoteness: {merged['remoteness_exporter'].isna().sum()}")

# =============================================================================
# PHASE 7: PREPARE ANALYSIS DATASET
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 7: PREPARE ANALYSIS DATASET")
print("=" * 70)

# Define estimation variables
est_vars = [
    'log_trade', 'log_gdp_importer', 'log_gdp_exporter',
    'log_dist', 'contig', 'comlang_off', 'col45',
    'rta', 'remoteness_importer', 'remoteness_exporter',
    'efw_importer', 'efw_exporter',
    'incentive_exporter', 'incentive_importer'
]

print(f"Total observations before dropping missing: {len(merged):,}")
print(f"\nMissing values:")
for var in est_vars:
    if var in merged.columns:
        n_miss = merged[var].isna().sum()
        print(f"  {var}: {n_miss} ({n_miss/len(merged)*100:.1f}%)")

# Drop incomplete cases
df = merged.dropna(subset=est_vars).copy()
print(f"\nComplete cases for estimation: {len(df):,}")
print(f"Unique exporters: {df['exporter'].nunique()}")
print(f"Unique importers: {df['importer'].nunique()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# Save the analysis dataset
analysis_file = f'{PROJECT_DIR}\\data\\processed\\batis_gravity_analysis.csv'
df.to_csv(analysis_file, index=False)
print(f"\nSaved analysis dataset to: {analysis_file}")

# =============================================================================
# PHASE 8: BASELINE GRAVITY MODELS (INCREMENTAL)
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 8: GRAVITY MODEL ESTIMATION")
print("=" * 70)

# Model 1: Baseline (GDP + distance)
print("\n--- Model 1: Baseline (GDP + Distance) ---")
m1 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist', data=df).fit(cov_type='HC1')
print(f"R²: {m1.rsquared:.4f}, N: {int(m1.nobs)}")

# Model 2: + Cultural/historical controls
print("\n--- Model 2: + Cultural controls ---")
m2 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45', data=df).fit(cov_type='HC1')
print(f"R²: {m2.rsquared:.4f}, N: {int(m2.nobs)}")

# Model 3: + Remoteness
print("\n--- Model 3: + Remoteness ---")
m3 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter', data=df).fit(cov_type='HC1')
print(f"R²: {m3.rsquared:.4f}, N: {int(m3.nobs)}")

# Model 4: + RTA
print("\n--- Model 4: + RTA ---")
m4 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter + rta', data=df).fit(cov_type='HC1')
print(f"R²: {m4.rsquared:.4f}, N: {int(m4.nobs)}")

# Model 5: + EFW
print("\n--- Model 5: + Economic Freedom ---")
m5 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter + rta + efw_importer + efw_exporter', data=df).fit(cov_type='HC1')
print(f"R²: {m5.rsquared:.4f}, N: {int(m5.nobs)}")

# Model 6: + Year FE
print("\n--- Model 6: + Year Fixed Effects ---")
m6 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter + rta + efw_importer + efw_exporter + C(year)', data=df).fit(cov_type='HC1')
print(f"R²: {m6.rsquared:.4f}, N: {int(m6.nobs)}")

# Model 7: + Film incentives
print("\n--- Model 7: + Film Incentives (Full Model) ---")
m7 = smf.ols('log_trade ~ log_gdp_importer + log_gdp_exporter + log_dist + contig + comlang_off + col45 + remoteness_importer + remoteness_exporter + rta + efw_importer + efw_exporter + incentive_exporter + incentive_importer + C(year)', data=df).fit(cov_type='HC1')
print(f"R²: {m7.rsquared:.4f}, N: {int(m7.nobs)}")

# --- Comparison table ---
print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON (Models 1-7)")
print("=" * 70)

key_vars = [
    'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
    'contig', 'comlang_off', 'col45',
    'remoteness_importer', 'remoteness_exporter',
    'rta', 'efw_importer', 'efw_exporter',
    'incentive_exporter', 'incentive_importer'
]

models = [('M1', m1), ('M2', m2), ('M3', m3), ('M4', m4), ('M5', m5), ('M6', m6), ('M7', m7)]

header = f"{'Variable':<25}" + "".join([f"{name:>10}" for name, _ in models])
print(f"\n{header}")
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
print("\n* p<0.1, ** p<0.05, *** p<0.01")

# =============================================================================
# PHASE 9: ROBUSTNESS — IMPORTER + EXPORTER + YEAR FIXED EFFECTS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 9: IMPORTER + EXPORTER + YEAR FIXED EFFECTS")
print("=" * 70)

formula_fe = ('log_trade ~ log_dist + contig + comlang_off + col45 + rta + '
              'incentive_exporter + incentive_importer + '
              'C(importer) + C(exporter) + C(year)')

m_fe = smf.ols(formula_fe, data=df).fit(cov_type='HC1')

print(f"R²: {m_fe.rsquared:.4f}, N: {int(m_fe.nobs)}")
print(f"Unique importers: {df['importer'].nunique()}, exporters: {df['exporter'].nunique()}")

fe_vars = ['log_dist', 'contig', 'comlang_off', 'col45', 'rta',
           'incentive_exporter', 'incentive_importer']

print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
print("-" * 60)
for var in fe_vars:
    if var in m_fe.params:
        print(f"{var:<25} {m_fe.params[var]:>10.4f} {m_fe.bse[var]:>10.4f} {m_fe.pvalues[var]:>8.4f} {'***' if m_fe.pvalues[var] < 0.01 else '**' if m_fe.pvalues[var] < 0.05 else '*' if m_fe.pvalues[var] < 0.1 else '':>5}")

# =============================================================================
# PHASE 10: ROBUSTNESS — COUNTRY-PAIR + YEAR FIXED EFFECTS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 10: COUNTRY-PAIR + YEAR FIXED EFFECTS")
print("=" * 70)

# Create pair identifier
df['pair_id'] = df['exporter'] + '_' + df['importer']

print(f"Unique country pairs: {df['pair_id'].nunique()}")

# Check observations per pair
pair_counts = df.groupby('pair_id').size()
print(f"\nObservations per pair:")
print(f"  Mean:   {pair_counts.mean():.1f}")
print(f"  Median: {pair_counts.median():.1f}")
print(f"  Min:    {pair_counts.min()}")
print(f"  Max:    {pair_counts.max()}")
print(f"  Pairs with only 1 obs: {(pair_counts == 1).sum()} ({(pair_counts == 1).mean()*100:.1f}%)")

# Drop singleton pairs
pairs_to_keep = pair_counts[pair_counts > 1].index
df_panel = df[df['pair_id'].isin(pairs_to_keep)].copy()
print(f"\nAfter dropping singletons:")
print(f"  Observations: {len(df_panel):,}")
print(f"  Pairs: {df_panel['pair_id'].nunique()}")

# Check within-pair variation in incentives
inc_var = df_panel.groupby('pair_id')['incentive_exporter'].nunique()
print(f"\nIncentive variation check:")
print(f"  Pairs where incentive_exporter changes: {(inc_var > 1).sum()}")
print(f"  Share with variation: {(inc_var > 1).mean()*100:.1f}%")

inc_var_imp = df_panel.groupby('pair_id')['incentive_importer'].nunique()
print(f"  Pairs where incentive_importer changes: {(inc_var_imp > 1).sum()}")
print(f"  Share with variation: {(inc_var_imp > 1).mean()*100:.1f}%")

# Time-varying variables only
time_varying = [
    'log_gdp_importer', 'log_gdp_exporter',
    'remoteness_importer', 'remoteness_exporter',
    'efw_importer', 'efw_exporter',
    'rta', 'incentive_exporter', 'incentive_importer'
]

# Within-transformation: demean by country pair to avoid huge dummy matrix
# This is algebraically equivalent to including C(pair_id) dummies
print("\nApplying within-transformation (demeaning by pair)...")

demean_vars = ['log_trade'] + time_varying
df_demeaned = df_panel.copy()

# Demean each variable by pair
pair_means = df_panel.groupby('pair_id')[demean_vars].transform('mean')
for var in demean_vars:
    df_demeaned[f'{var}_dm'] = df_panel[var] - pair_means[var]

# Also demean by year (two-way FE: pair + year)
# First add year dummies to demeaned data and run OLS
# Simpler approach: demean by pair, then include C(year)
dm_formula = (f'log_trade_dm ~ ' +
              ' + '.join([f'{v}_dm' for v in time_varying]) +
              ' + C(year) - 1')  # no intercept since we demeaned

m_pair = smf.ols(dm_formula, data=df_demeaned).fit(cov_type='HC1')

# Calculate within R-squared
ss_res = np.sum(m_pair.resid ** 2)
ss_tot = np.sum(df_demeaned['log_trade_dm'] ** 2)
within_r2 = 1 - ss_res / ss_tot

print(f"\nWithin R²: {within_r2:.4f}")
print(f"N: {int(m_pair.nobs)}")

print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
print("-" * 60)
for var in time_varying:
    dm_var = f'{var}_dm'
    if dm_var in m_pair.params:
        coef = m_pair.params[dm_var]
        se = m_pair.bse[dm_var]
        p = m_pair.pvalues[dm_var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"{var:<25} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5}")

# =============================================================================
# PHASE 10b: COUNTRY-PAIR + YEAR FE — WITHOUT EFW (more observations)
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 10b: COUNTRY-PAIR + YEAR FE — WITHOUT EFW")
print("=" * 70)

# Redefine estimation variables without EFW
est_vars_no_efw = [
    'log_trade', 'log_gdp_importer', 'log_gdp_exporter',
    'log_dist', 'contig', 'comlang_off', 'col45',
    'rta', 'remoteness_importer', 'remoteness_exporter',
    'incentive_exporter', 'incentive_importer'
]

df_no_efw = merged.dropna(subset=est_vars_no_efw).copy()
print(f"Complete cases (no EFW): {len(df_no_efw):,}")
print(f"Gained vs main sample: {len(df_no_efw) - len(df):,} observations")

# Create pair id and drop singletons
df_no_efw['pair_id'] = df_no_efw['exporter'] + '_' + df_no_efw['importer']
pair_counts_ne = df_no_efw.groupby('pair_id').size()
pairs_keep_ne = pair_counts_ne[pair_counts_ne > 1].index
df_panel_ne = df_no_efw[df_no_efw['pair_id'].isin(pairs_keep_ne)].copy()

print(f"After dropping singletons: {len(df_panel_ne):,} obs, {df_panel_ne['pair_id'].nunique()} pairs")

# Incentive variation check
inc_var_ne = df_panel_ne.groupby('pair_id')['incentive_exporter'].nunique()
inc_var_imp_ne = df_panel_ne.groupby('pair_id')['incentive_importer'].nunique()
print(f"\nIncentive variation check:")
print(f"  Pairs where incentive_exporter changes: {(inc_var_ne > 1).sum()} ({(inc_var_ne > 1).mean()*100:.1f}%)")
print(f"  Pairs where incentive_importer changes: {(inc_var_imp_ne > 1).sum()} ({(inc_var_imp_ne > 1).mean()*100:.1f}%)")

# Within-transformation
time_varying_ne = [
    'log_gdp_importer', 'log_gdp_exporter',
    'remoteness_importer', 'remoteness_exporter',
    'rta', 'incentive_exporter', 'incentive_importer'
]

demean_vars_ne = ['log_trade'] + time_varying_ne
df_dm_ne = df_panel_ne.copy()

pair_means_ne = df_panel_ne.groupby('pair_id')[demean_vars_ne].transform('mean')
for var in demean_vars_ne:
    df_dm_ne[f'{var}_dm'] = df_panel_ne[var] - pair_means_ne[var]

dm_formula_ne = (f'log_trade_dm ~ ' +
                 ' + '.join([f'{v}_dm' for v in time_varying_ne]) +
                 ' + C(year) - 1')

m_pair_ne = smf.ols(dm_formula_ne, data=df_dm_ne).fit(cov_type='HC1')

ss_res_ne = np.sum(m_pair_ne.resid ** 2)
ss_tot_ne = np.sum(df_dm_ne['log_trade_dm'] ** 2)
within_r2_ne = 1 - ss_res_ne / ss_tot_ne

print(f"\nWithin R²: {within_r2_ne:.4f}")
print(f"N: {int(m_pair_ne.nobs)}")

print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
print("-" * 60)
for var in time_varying_ne:
    dm_var = f'{var}_dm'
    if dm_var in m_pair_ne.params:
        coef = m_pair_ne.params[dm_var]
        se = m_pair_ne.bse[dm_var]
        p = m_pair_ne.pvalues[dm_var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"{var:<25} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5}")

# =============================================================================
# PHASE 10c: COUNTRY-PAIR + YEAR FE — EXCLUDING ANGLO-TO-ANGLO PAIRS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 10c: COUNTRY-PAIR + YEAR FE — EXCLUDING ANGLO-TO-ANGLO")
print("=" * 70)

ANGLO = {'US', 'GB', 'CA', 'AU', 'NZ'}

# Exclude pairs where BOTH countries are Anglo
df_no_anglo_pair = df[
    ~((df['exporter'].isin(ANGLO)) & (df['importer'].isin(ANGLO)))
].copy()
print(f"Observations after excluding Anglo-to-Anglo: {len(df_no_anglo_pair):,}")
print(f"Dropped: {len(df) - len(df_no_anglo_pair):,}")

# Create pair id and drop singletons
df_no_anglo_pair['pair_id'] = df_no_anglo_pair['exporter'] + '_' + df_no_anglo_pair['importer']
pair_counts_naa = df_no_anglo_pair.groupby('pair_id').size()
pairs_keep_naa = pair_counts_naa[pair_counts_naa > 1].index
df_panel_naa = df_no_anglo_pair[df_no_anglo_pair['pair_id'].isin(pairs_keep_naa)].copy()

print(f"After dropping singletons: {len(df_panel_naa):,} obs, {df_panel_naa['pair_id'].nunique()} pairs")

# Incentive variation
inc_var_naa = df_panel_naa.groupby('pair_id')['incentive_exporter'].nunique()
inc_var_imp_naa = df_panel_naa.groupby('pair_id')['incentive_importer'].nunique()
print(f"\nIncentive variation:")
print(f"  incentive_exporter changes: {(inc_var_naa > 1).sum()} ({(inc_var_naa > 1).mean()*100:.1f}%)")
print(f"  incentive_importer changes: {(inc_var_imp_naa > 1).sum()} ({(inc_var_imp_naa > 1).mean()*100:.1f}%)")

# Within-transformation
demean_vars_naa = ['log_trade'] + time_varying
df_dm_naa = df_panel_naa.copy()
pair_means_naa = df_panel_naa.groupby('pair_id')[demean_vars_naa].transform('mean')
for var in demean_vars_naa:
    df_dm_naa[f'{var}_dm'] = df_panel_naa[var] - pair_means_naa[var]

dm_formula_naa = (f'log_trade_dm ~ ' +
                  ' + '.join([f'{v}_dm' for v in time_varying]) +
                  ' + C(year) - 1')

m_pair_naa = smf.ols(dm_formula_naa, data=df_dm_naa).fit(cov_type='HC1')

ss_res_naa = np.sum(m_pair_naa.resid ** 2)
ss_tot_naa = np.sum(df_dm_naa['log_trade_dm'] ** 2)
within_r2_naa = 1 - ss_res_naa / ss_tot_naa

print(f"\nWithin R²: {within_r2_naa:.4f}")
print(f"N: {int(m_pair_naa.nobs)}")

print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
print("-" * 60)
for var in time_varying:
    dm_var = f'{var}_dm'
    if dm_var in m_pair_naa.params:
        coef = m_pair_naa.params[dm_var]
        se = m_pair_naa.bse[dm_var]
        p = m_pair_naa.pvalues[dm_var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"{var:<25} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5}")

# =============================================================================
# PHASE 10d: COUNTRY-PAIR + YEAR FE — EXCLUDING ALL ANGLO COUNTRIES
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 10d: COUNTRY-PAIR + YEAR FE — EXCLUDING ALL ANGLO")
print("=" * 70)

# Exclude pairs where EITHER country is Anglo
df_no_anglo_any = df[
    ~((df['exporter'].isin(ANGLO)) | (df['importer'].isin(ANGLO)))
].copy()
print(f"Observations after excluding all Anglo: {len(df_no_anglo_any):,}")
print(f"Dropped: {len(df) - len(df_no_anglo_any):,}")

# Create pair id and drop singletons
df_no_anglo_any['pair_id'] = df_no_anglo_any['exporter'] + '_' + df_no_anglo_any['importer']
pair_counts_nax = df_no_anglo_any.groupby('pair_id').size()
pairs_keep_nax = pair_counts_nax[pair_counts_nax > 1].index
df_panel_nax = df_no_anglo_any[df_no_anglo_any['pair_id'].isin(pairs_keep_nax)].copy()

print(f"After dropping singletons: {len(df_panel_nax):,} obs, {df_panel_nax['pair_id'].nunique()} pairs")

# Incentive variation
inc_var_nax = df_panel_nax.groupby('pair_id')['incentive_exporter'].nunique()
inc_var_imp_nax = df_panel_nax.groupby('pair_id')['incentive_importer'].nunique()
print(f"\nIncentive variation:")
print(f"  incentive_exporter changes: {(inc_var_nax > 1).sum()} ({(inc_var_nax > 1).mean()*100:.1f}%)")
print(f"  incentive_importer changes: {(inc_var_imp_nax > 1).sum()} ({(inc_var_imp_nax > 1).mean()*100:.1f}%)")

# Within-transformation
demean_vars_nax = ['log_trade'] + time_varying
df_dm_nax = df_panel_nax.copy()
pair_means_nax = df_panel_nax.groupby('pair_id')[demean_vars_nax].transform('mean')
for var in demean_vars_nax:
    df_dm_nax[f'{var}_dm'] = df_panel_nax[var] - pair_means_nax[var]

dm_formula_nax = (f'log_trade_dm ~ ' +
                  ' + '.join([f'{v}_dm' for v in time_varying]) +
                  ' + C(year) - 1')

m_pair_nax = smf.ols(dm_formula_nax, data=df_dm_nax).fit(cov_type='HC1')

ss_res_nax = np.sum(m_pair_nax.resid ** 2)
ss_tot_nax = np.sum(df_dm_nax['log_trade_dm'] ** 2)
within_r2_nax = 1 - ss_res_nax / ss_tot_nax

print(f"\nWithin R²: {within_r2_nax:.4f}")
print(f"N: {int(m_pair_nax.nobs)}")

print(f"\n{'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>8} {'Sig':>5}")
print("-" * 60)
for var in time_varying:
    dm_var = f'{var}_dm'
    if dm_var in m_pair_nax.params:
        coef = m_pair_nax.params[dm_var]
        se = m_pair_nax.bse[dm_var]
        p = m_pair_nax.pvalues[dm_var]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"{var:<25} {coef:>10.4f} {se:>10.4f} {p:>8.4f} {stars:>5}")

# =============================================================================
# PHASE 11: SUMMARY COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: INCENTIVE EFFECT ACROSS SPECIFICATIONS")
print("=" * 70)

specs = [
    ('Model 7 (Year FE)', m7),
    ('Imp+Exp+Year FE', m_fe),
    ('Country-pair+Year FE', m_pair),
    ('Pair+Year FE (no EFW)', m_pair_ne),
    ('Pair+Year FE (excl A-A)', m_pair_naa),
    ('Pair+Year FE (excl Anglo)', m_pair_nax)
]

print(f"\n{'Specification':<30} {'inc_exporter':>15} {'inc_importer':>15}")
print("-" * 62)

for name, model in specs:
    row = f"{name:<30}"
    for var in ['incentive_exporter', 'incentive_importer']:
        # Check both original and demeaned variable names
        lookup = var if var in model.params else f'{var}_dm'
        if lookup in model.params:
            coef = model.params[lookup]
            p = model.pvalues[lookup]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            pct = (np.exp(coef) - 1) * 100
            row += f" {coef:>6.3f}{stars:<3}({pct:>+.0f}%)"
        else:
            row += f"{'absorbed':>15}"
    print(row)

print("\n* p<0.1, ** p<0.05, *** p<0.01")
print("Percentage effects in parentheses: (exp(coef)-1)*100")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)