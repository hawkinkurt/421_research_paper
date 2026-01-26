""" merge_film_incentives.py: Create time-varying film incentive dummy and merge into gravity dataset """

import pandas as pd
import numpy as np

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")
df_incentives = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/film_incentive_intro_dates.csv")

print("=" * 70)
print("MERGING FILM PRODUCTION INCENTIVE DATA")
print("=" * 70)
print(f"\nGravity dataset: {len(df)} observations")
print(f"Incentive data: {len(df_incentives)} countries")


# =============================================================================
# CREATE TIME-VARYING INCENTIVE DUMMY
# =============================================================================

# For each exporter-year, check if incentive was in place
def has_incentive(row, incentive_df):
    """Returns 1 if country had incentive scheme in that year, 0 otherwise"""
    country = row['exporter']
    year = row['year']

    # Look up introduction year for this country
    match = incentive_df[incentive_df['country_iso2'] == country]

    if len(match) == 0:
        # Country not in our list - assume no incentive
        return 0

    intro_year = match['incentive_intro_year'].values[0]

    if pd.isna(intro_year):
        # No incentive for this country
        return 0

    # Check if year >= introduction year
    return 1 if year >= intro_year else 0


print("\nCreating time-varying incentive dummy for exporters...")
df['incentive_exporter'] = df.apply(lambda row: has_incentive(row, df_incentives), axis=1)


# Also create for importers (in case useful)
def has_incentive_importer(row, incentive_df):
    """Returns 1 if importer country had incentive scheme in that year"""
    country = row['importer']
    year = row['year']

    match = incentive_df[incentive_df['country_iso2'] == country]

    if len(match) == 0:
        return 0

    intro_year = match['incentive_intro_year'].values[0]

    if pd.isna(intro_year):
        return 0

    return 1 if year >= intro_year else 0


print("Creating time-varying incentive dummy for importers...")
df['incentive_importer'] = df.apply(lambda row: has_incentive_importer(row, df_incentives), axis=1)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("INCENTIVE VARIABLE SUMMARY")
print("=" * 70)

print("\n=== Exporter Incentive ===")
print(
    f"Observations with exporter incentive = 1: {df['incentive_exporter'].sum()} ({df['incentive_exporter'].mean() * 100:.1f}%)")
print(
    f"Observations with exporter incentive = 0: {(df['incentive_exporter'] == 0).sum()} ({(1 - df['incentive_exporter'].mean()) * 100:.1f}%)")

print("\n=== Importer Incentive ===")
print(
    f"Observations with importer incentive = 1: {df['incentive_importer'].sum()} ({df['incentive_importer'].mean() * 100:.1f}%)")
print(
    f"Observations with importer incentive = 0: {(df['incentive_importer'] == 0).sum()} ({(1 - df['incentive_importer'].mean()) * 100:.1f}%)")

# Show breakdown by country
print("\n=== Exporter Incentive by Country (Top 15) ===")
exporter_incentive = df.groupby('exporter').agg({
    'incentive_exporter': ['sum', 'count', 'mean']
}).round(2)
exporter_incentive.columns = ['obs_with_incentive', 'total_obs', 'share_with_incentive']
exporter_incentive = exporter_incentive.sort_values('total_obs', ascending=False).head(15)
print(exporter_incentive.to_string())

# Show time pattern
print("\n=== Incentive Coverage Over Time ===")
yearly = df.groupby('year').agg({
    'incentive_exporter': 'mean',
    'incentive_importer': 'mean'
}).round(3)
yearly.columns = ['share_exp_incentive', 'share_imp_incentive']
print(yearly.to_string())

# =============================================================================
# SAVE UPDATED DATASET
# =============================================================================

df.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv", index=False)

print("\n" + "=" * 70)
print("DATASET UPDATED")
print("=" * 70)
print(f"Added columns: 'incentive_exporter', 'incentive_importer'")
print(f"Saved to: gravity_dataset_analysis.csv")
print(f"Total observations: {len(df)}")