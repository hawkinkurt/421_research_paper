"""
batis_data_prep.py
==================
Data preparation for BaTIS SK1 (audiovisual services) gravity analysis.
Loads, cleans, merges all variables, and saves analysis-ready datasets.

Run this FIRST, then run batis_gravity_analysis.py for estimation.

Outputs:
  - data/processed/batis_sk1_gravity_analysis.csv  (complete cases with EFW)
  - data/processed/batis_sk1_gravity_merged.csv     (all merged obs, for no-EFW specs)
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'

SK1_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\batis_sk1_all_years.csv'

CEPII_FILE = f'{PROJECT_DIR}\\data\\processed\\gravity_vars_cepii.csv'
GDP_FILE = f'{PROJECT_DIR}\\data\\processed\\gdp_cleaned.csv'
EFW_FILE = f'{PROJECT_DIR}\\data\\processed\\efw_cleaned.csv'
INCENTIVE_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\film_incentive_intro_dates.csv'
CPI_FILE = f'{PROJECT_DIR}\\data\\raw\\world_bank_cpi.csv'

YEAR_MAX = 2023  # World Bank GDP constraint
BASE_YEAR = 2018  # CPI deflation base

# Output paths
ANALYSIS_FILE = f'{PROJECT_DIR}\\data\\processed\\batis_sk1_gravity_analysis.csv'
MERGED_FILE = f'{PROJECT_DIR}\\data\\processed\\batis_sk1_gravity_merged.csv'

# =============================================================================
# PHASE 1: LOAD SHARED DATA
# =============================================================================

print("=" * 70)
print("PHASE 1: LOAD SHARED DATA")
print("=" * 70)

# CEPII
cepii = pd.read_csv(CEPII_FILE)
print(f"CEPII: {len(cepii):,} rows, years {cepii['year'].min()}-{cepii['year'].max()}")

cepii_max = cepii['year'].max()
if YEAR_MAX > cepii_max:
    print(f"Forward-filling CEPII from {cepii_max} to {YEAR_MAX}...")
    cepii_latest = cepii[cepii['year'] == cepii_max].copy()
    new_rows = [cepii_latest.assign(year=y) for y in range(cepii_max + 1, YEAR_MAX + 1)]
    cepii = pd.concat([cepii] + new_rows, ignore_index=True)

# GDP
gdp = pd.read_csv(GDP_FILE)
print(f"GDP: {len(gdp):,} rows, years {gdp['date'].min()}-{gdp['date'].max()}")

# CPI
cpi = pd.read_csv(CPI_FILE)
cpi = cpi.rename(columns={'date': 'year'}) if 'date' in cpi.columns else cpi
cpi['year'] = cpi['year'].astype(int)
print(f"CPI: {len(cpi)} years, range {cpi['year'].min()}-{cpi['year'].max()}")

# EFW
efw = pd.read_csv(EFW_FILE)
print(f"EFW: {len(efw):,} rows, years {efw['year'].min()}-{efw['year'].max()}")
efw_max = efw['year'].max()
if efw_max < YEAR_MAX:
    print(f"Forward-filling EFW from {efw_max} to {YEAR_MAX}...")
    efw_latest = efw[efw['year'] == efw_max].copy()
    new_efw = [efw_latest.assign(year=y) for y in range(efw_max + 1, YEAR_MAX + 1)]
    efw = pd.concat([efw] + new_efw, ignore_index=True)

# Incentives
incentives = pd.read_csv(INCENTIVE_FILE)
print(f"Incentives: {len(incentives)} countries")
print(f"Columns: {incentives.columns.tolist()}")

incentive_dict = dict(zip(
    incentives['country_iso2'].astype(str),
    incentives['incentive_intro_year']
))

TYPE_DUMMIES = ['is_refundable_credit', 'is_transferable_credit', 'is_standard_credit', 'is_cash_rebate']
type_lookups = {}
for dtype in TYPE_DUMMIES:
    if dtype in incentives.columns:
        type_lookups[dtype] = dict(zip(
            incentives['country_iso2'].astype(str),
            incentives[dtype].fillna(0).astype(int)
        ))
    else:
        type_lookups[dtype] = {}

has_generosity = 'headline_rate_pct' in incentives.columns
generosity_dict = {}
if has_generosity:
    generosity_dict = dict(zip(
        incentives['country_iso2'].astype(str),
        incentives['headline_rate_pct'].fillna(0) / 100
    ))
    print(f"  Loaded headline_rate_pct for generosity analysis")
else:
    print(f"  No headline_rate_pct — generosity analysis will be skipped")

# =============================================================================
# PHASE 2: LOAD AND CLEAN BaTIS SK1
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: LOAD AND CLEAN BaTIS SK1")
print("=" * 70)

batis = pd.read_csv(SK1_FILE)
print(f"Loaded: {len(batis):,} rows, years {batis['Year'].min()}-{batis['Year'].max()}")

batis['Reporter'] = batis['Reporter'].astype(str)
batis['Partner'] = batis['Partner'].astype(str)

# Drop aggregates
batis = batis[
    (batis['Reporter'] != '888') &
    (batis['Partner'] != '888') &
    (batis['Partner'] != 'WL')
]

# Drop zero/negative
batis = batis[batis['Balanced_value'] > 0].copy()

batis = batis.rename(columns={
    'Reporter': 'exporter',
    'Partner': 'importer',
    'Year': 'year',
    'Balanced_value': 'trade_value_nominal'
})

# Deflate to 2018 USD
base_cpi = cpi[cpi['year'] == BASE_YEAR]['cpi'].values[0]
cpi_adj = cpi[['year', 'cpi']].copy()
cpi_adj['adjustment_factor'] = base_cpi / cpi_adj['cpi']

batis = batis.merge(cpi_adj[['year', 'adjustment_factor']], on='year', how='left')

if batis['adjustment_factor'].isna().any():
    latest_factor = cpi_adj.loc[cpi_adj['year'] == cpi_adj['year'].max(), 'adjustment_factor'].values[0]
    n_filled = batis['adjustment_factor'].isna().sum()
    batis['adjustment_factor'] = batis['adjustment_factor'].fillna(latest_factor)
    print(f"  Filled {n_filled} obs with latest CPI factor for years beyond CPI coverage")

batis['trade_value'] = batis['trade_value_nominal'] * batis['adjustment_factor']
batis['log_trade'] = np.log(batis['trade_value'])
batis = batis.drop(columns=['adjustment_factor'])

print(f"After cleaning: {len(batis):,} rows")
print(f"Exporters: {batis['exporter'].nunique()}, Importers: {batis['importer'].nunique()}")
print(f"Years: {batis['year'].min()}-{batis['year'].max()}")
print(f"Deflated to {BASE_YEAR} USD (nominal: {batis['trade_value_nominal'].sum():,.0f}, real: {batis['trade_value'].sum():,.0f})")

# =============================================================================
# PHASE 3: MERGE WITH CEPII GRAVITY VARIABLES
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: MERGE WITH CEPII")
print("=" * 70)

merged = batis.merge(
    cepii, left_on=['exporter', 'importer', 'year'],
    right_on=['iso_o', 'iso_d', 'year'], how='left'
)
merged = merged.drop(columns=['iso_o', 'iso_d'], errors='ignore')
merged = merged[merged['year'] <= YEAR_MAX].copy()

print(f"After CEPII merge: {len(merged):,} rows")
print(f"Missing distance: {merged['log_dist'].isna().sum()} ({merged['log_dist'].isna().mean()*100:.1f}%)")

# =============================================================================
# PHASE 4: MERGE WITH GDP
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4: MERGE WITH GDP")
print("=" * 70)

merged = merged.merge(
    gdp[['country', 'date', 'log_gdp']].rename(
        columns={'country': 'importer', 'date': 'year', 'log_gdp': 'log_gdp_importer'}),
    on=['importer', 'year'], how='left'
)
merged = merged.merge(
    gdp[['country', 'date', 'log_gdp']].rename(
        columns={'country': 'exporter', 'date': 'year', 'log_gdp': 'log_gdp_exporter'}),
    on=['exporter', 'year'], how='left'
)

print(f"Missing importer GDP: {merged['log_gdp_importer'].isna().sum()} ({merged['log_gdp_importer'].isna().mean()*100:.1f}%)")
print(f"Missing exporter GDP: {merged['log_gdp_exporter'].isna().sum()} ({merged['log_gdp_exporter'].isna().mean()*100:.1f}%)")

# =============================================================================
# PHASE 5: MERGE WITH EFW
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 5: MERGE WITH EFW")
print("=" * 70)

merged = merged.merge(
    efw.rename(columns={'country': 'importer', 'efw': 'efw_importer'}),
    on=['importer', 'year'], how='left'
)
merged = merged.merge(
    efw.rename(columns={'country': 'exporter', 'efw': 'efw_exporter'}),
    on=['exporter', 'year'], how='left'
)

print(f"Missing importer EFW: {merged['efw_importer'].isna().sum()} ({merged['efw_importer'].isna().mean()*100:.1f}%)")
print(f"Missing exporter EFW: {merged['efw_exporter'].isna().sum()} ({merged['efw_exporter'].isna().mean()*100:.1f}%)")

# =============================================================================
# PHASE 6: CREATE INCENTIVE VARIABLES
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 6: CREATE INCENTIVE VARIABLES")
print("=" * 70)

# Overall incentive dummies
merged['incentive_exporter'] = merged.apply(
    lambda row: 1 if row['exporter'] in incentive_dict and pd.notna(incentive_dict[row['exporter']]) and row['year'] >= incentive_dict[row['exporter']] else 0,
    axis=1
)
merged['incentive_importer'] = merged.apply(
    lambda row: 1 if row['importer'] in incentive_dict and pd.notna(incentive_dict[row['importer']]) and row['year'] >= incentive_dict[row['importer']] else 0,
    axis=1
)

# Type dummies
for dtype in TYPE_DUMMIES:
    col_exp = f'{dtype}_exp'
    merged[col_exp] = merged.apply(
        lambda row, dt=dtype: (
            1 if (row['exporter'] in incentive_dict
                  and pd.notna(incentive_dict[row['exporter']])
                  and row['year'] >= incentive_dict[row['exporter']]
                  and type_lookups[dt].get(row['exporter'], 0) == 1)
            else 0
        ), axis=1
    )

# Generosity
if has_generosity:
    merged['generosity_exp'] = merged.apply(
        lambda row: (
            generosity_dict.get(row['exporter'], 0)
            if (row['exporter'] in incentive_dict
                and pd.notna(incentive_dict[row['exporter']])
                and row['year'] >= incentive_dict[row['exporter']])
            else 0
        ), axis=1
    )
else:
    merged['generosity_exp'] = 0

print(f"Exporter incentive active: {merged['incentive_exporter'].sum()} ({merged['incentive_exporter'].mean()*100:.1f}%)")
print(f"Importer incentive active: {merged['incentive_importer'].sum()} ({merged['incentive_importer'].mean()*100:.1f}%)")

print(f"\nExporter incentive by type:")
for dtype in TYPE_DUMMIES:
    col = f'{dtype}_exp'
    print(f"  {dtype}: {merged[col].sum()} ({merged[col].mean()*100:.1f}%)")

if has_generosity:
    active_gen = merged[merged['generosity_exp'] > 0]['generosity_exp']
    if len(active_gen) > 0:
        print(f"\nGenerosity (active incentives): mean={active_gen.mean()*100:.1f}%, "
              f"min={active_gen.min()*100:.1f}%, max={active_gen.max()*100:.1f}%")

# =============================================================================
# PHASE 7: CALCULATE REMOTENESS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 7: CALCULATE REMOTENESS")
print("=" * 70)

gdp_levels = gdp[['country', 'date', 'gdp']].rename(columns={'date': 'year'}).copy()
gdp_levels = gdp_levels.dropna(subset=['gdp'])
world_gdp = gdp_levels.groupby('year')['gdp'].sum().reset_index()
world_gdp.columns = ['year', 'gdp_world']

dist_year = cepii[cepii['year'] <= 2020]['year'].max()
dist_data = cepii[cepii['year'] == dist_year][['iso_o', 'iso_d', 'dist']].copy()
dist_data = dist_data.dropna(subset=['dist'])
dist_data = dist_data[(dist_data['dist'] > 0) & (dist_data['iso_o'] != dist_data['iso_d'])]

all_countries = sorted(
    (set(dist_data['iso_o'].dropna().unique()) | set(dist_data['iso_d'].dropna().unique())) - {np.nan}
)
all_years = [y for y in sorted(gdp_levels['year'].unique()) if 2005 <= y <= YEAR_MAX]

remoteness_records = []
for yr in all_years:
    gdp_yr = gdp_levels[gdp_levels['year'] == yr].set_index('country')['gdp'].to_dict()
    gdp_world_yr = world_gdp[world_gdp['year'] == yr]['gdp_world'].values
    if len(gdp_world_yr) == 0:
        continue
    gdp_world_yr = gdp_world_yr[0]

    for country in all_countries:
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
                weighted_sum += (partner_gdp / gdp_world_yr) / row['dist']
        if weighted_sum > 0:
            remoteness_records.append({'country': country, 'year': yr, 'remoteness': np.log(1 / weighted_sum)})

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
# PHASE 8: PREPARE AND SAVE ANALYSIS DATASETS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 8: PREPARE ANALYSIS DATASETS")
print("=" * 70)

est_vars = [
    'log_trade', 'log_gdp_importer', 'log_gdp_exporter',
    'log_dist', 'contig', 'comlang_off', 'col45',
    'rta', 'remoteness_importer', 'remoteness_exporter',
    'efw_importer', 'efw_exporter',
    'incentive_exporter', 'incentive_importer'
]

print(f"Total observations: {len(merged):,}")
print(f"\nMissing values:")
for var in est_vars:
    n_miss = merged[var].isna().sum()
    if n_miss > 0:
        print(f"  {var}: {n_miss} ({n_miss/len(merged)*100:.1f}%)")

# Complete cases (with EFW)
df = merged.dropna(subset=est_vars).copy()
print(f"\nComplete cases (with EFW): {len(df):,}")
print(f"Exporters: {df['exporter'].nunique()}, Importers: {df['importer'].nunique()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")

# Save both datasets
merged.to_csv(MERGED_FILE, index=False)
df.to_csv(ANALYSIS_FILE, index=False)

print(f"\nSaved:")
print(f"  Merged (all obs):     {MERGED_FILE} ({len(merged):,} rows)")
print(f"  Analysis (complete):  {ANALYSIS_FILE} ({len(df):,} rows)")

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE — now run batis_gravity_analysis.py")
print("=" * 70)