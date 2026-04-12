"""merge_film_incentives.py"""
"""Create time-varying film incentive dummy and merge into gravity dataset"""

import pandas as pd

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")
df_incentives = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/base/film_incentive_intro_dates.csv")

print("=" * 70)
print("MERGING FILM PRODUCTION INCENTIVE DATA")
print("=" * 70)
print(f"\nGravity dataset: {len(df)} observations")
print(f"Incentive data: {len(df_incentives)} countries")

# =============================================================================
# VALIDATE INCENTIVE DATA
# =============================================================================

exporters_in_data = set(df['exporter'].unique())
importers_in_data = set(df['importer'].unique())
incentive_countries = set(df_incentives['country_iso2'].unique())

matched_exporters = incentive_countries & exporters_in_data
matched_importers = incentive_countries & importers_in_data
unmatched_incentive = incentive_countries - (exporters_in_data | importers_in_data)

print(f"\n=== Incentive Data Validation ===")
print(f"Incentive countries matched as exporters: {len(matched_exporters)}/{len(incentive_countries)}")
print(f"Incentive countries matched as importers: {len(matched_importers)}/{len(incentive_countries)}")

if unmatched_incentive:
    print(f"Incentive countries not in gravity data: {sorted(unmatched_incentive)}")

# =============================================================================
# CREATE EXPORTER INCENTIVE VARIABLE
# =============================================================================

print("\nCreating time-varying incentive dummy for exporters...")

# Merge incentive intro year for exporters
df = df.merge(
    df_incentives[['country_iso2', 'incentive_intro_year']].rename(
        columns={'country_iso2': 'exporter', 'incentive_intro_year': 'intro_year_exp'}
    ),
    on='exporter',
    how='left'
)

# Create dummy: 1 if year >= introduction year, 0 otherwise
df['incentive_exporter'] = ((df['year'] >= df['intro_year_exp']) & df['intro_year_exp'].notna()).astype(int)

# Drop temporary column
df = df.drop(columns=['intro_year_exp'])

# =============================================================================
# CREATE IMPORTER INCENTIVE VARIABLE
# =============================================================================

print("Creating time-varying incentive dummy for importers...")

# Merge incentive intro year for importers
df = df.merge(
    df_incentives[['country_iso2', 'incentive_intro_year']].rename(
        columns={'country_iso2': 'importer', 'incentive_intro_year': 'intro_year_imp'}
    ),
    on='importer',
    how='left'
)

# Create dummy: 1 if year >= introduction year, 0 otherwise
df['incentive_importer'] = ((df['year'] >= df['intro_year_imp']) & df['intro_year_imp'].notna()).astype(int)

# Drop temporary column
df = df.drop(columns=['intro_year_imp'])

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("INCENTIVE VARIABLE SUMMARY")
print("=" * 70)

print("\n=== Exporter Incentive ===")
exp_with = df['incentive_exporter'].sum()
exp_total = len(df)
print(f"Observations with incentive = 1: {exp_with} ({exp_with/exp_total*100:.1f}%)")
print(f"Observations with incentive = 0: {exp_total - exp_with} ({(exp_total - exp_with)/exp_total*100:.1f}%)")

print("\n=== Importer Incentive ===")
imp_with = df['incentive_importer'].sum()
print(f"Observations with incentive = 1: {imp_with} ({imp_with/exp_total*100:.1f}%)")
print(f"Observations with incentive = 0: {exp_total - imp_with} ({(exp_total - imp_with)/exp_total*100:.1f}%)")

# Show breakdown by country
print("\n=== Exporter Incentive by Country (Top 15) ===")
exporter_summary = df.groupby('exporter').agg(
    obs_with_incentive=('incentive_exporter', 'sum'),
    total_obs=('incentive_exporter', 'count'),
    share_with_incentive=('incentive_exporter', 'mean')
).round(2)
print(exporter_summary.sort_values('total_obs', ascending=False).head(15).to_string())

# Show time pattern
print("\n=== Incentive Coverage Over Time ===")
yearly = df.groupby('year').agg(
    share_exp_incentive=('incentive_exporter', 'mean'),
    share_imp_incentive=('incentive_importer', 'mean')
).round(3)
print(yearly.to_string())

# =============================================================================
# SAVE UPDATED DATASET
# =============================================================================

df.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv",
    index=False
)

print("\n" + "=" * 70)
print("DATASET UPDATED")
print("=" * 70)
print(f"Added columns: 'incentive_exporter', 'incentive_importer'")
print(f"Saved to: gravity_dataset_analysis.csv")
print(f"Total observations: {len(df)}")