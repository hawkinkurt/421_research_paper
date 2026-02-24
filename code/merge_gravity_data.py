"""merge_gravity_data.py"""
"""Merging all data into gravity dataset"""

import pandas as pd
import numpy as np

# =============================================================================
# LOAD DATA
# =============================================================================

df_trade = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_bilateral_real.csv")
df_gdp = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gdp_cleaned.csv")
df_grav = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_vars_cepii.csv")
df_efw = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/efw_cleaned.csv")
df_remote = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/remoteness.csv")

print("=== Data Loaded ===")
print(f"Trade observations: {len(df_trade)}")
print(f"GDP observations: {len(df_gdp)}")
print(f"Gravity (CEPII) observations: {len(df_grav)}")
print(f"EFW observations: {len(df_efw)}")
print(f"Remoteness observations: {len(df_remote)}")

# =============================================================================
# CHECK COLUMN NAMES
# =============================================================================

print("\n=== Column Names ===")
print(f"Trade: {df_trade.columns.tolist()}")
print(f"GDP: {df_gdp.columns.tolist()}")
print(f"Gravity: {df_grav.columns.tolist()}")
print(f"EFW: {df_efw.columns.tolist()}")

# =============================================================================
# MERGE GDP FOR IMPORTER (HOME COUNTRY)
# =============================================================================

df_gravity = df_trade.merge(
    df_gdp.rename(columns={'country': 'importer', 'gdp': 'gdp_importer', 'date': 'year', 'log_gdp': 'log_gdp_importer'}),
    on=['importer', 'year'],
    how='left'
)

print(f"\n=== After Importer GDP Merge ===")
print(f"Observations: {len(df_gravity)}")
print(f"Missing importer GDP: {df_gravity['gdp_importer'].isna().sum()}")

# =============================================================================
# MERGE GDP FOR EXPORTER (PRODUCTION COUNTRY)
# =============================================================================

df_gravity = df_gravity.merge(
    df_gdp.rename(columns={'country': 'exporter', 'gdp': 'gdp_exporter', 'date': 'year', 'log_gdp': 'log_gdp_exporter'}),
    on=['exporter', 'year'],
    how='left'
)

print(f"\n=== After Exporter GDP Merge ===")
print(f"Observations: {len(df_gravity)}")
print(f"Missing exporter GDP: {df_gravity['gdp_exporter'].isna().sum()}")

# =============================================================================
# MERGE CEPII GRAVITY VARIABLES (including RTA)
# =============================================================================

df_gravity = df_gravity.merge(
    df_grav,
    left_on=['importer', 'exporter', 'year'],
    right_on=['iso_o', 'iso_d', 'year'],
    how='left'
)

print(f"\n=== After CEPII Gravity Merge ===")
print(f"Observations: {len(df_gravity)}")
print(f"Missing distance: {df_gravity['dist'].isna().sum()}")
print(f"Missing RTA: {df_gravity['rta'].isna().sum()}")

# =============================================================================
# MERGE EFW FOR IMPORTER
# =============================================================================

df_gravity = df_gravity.merge(
    df_efw.rename(columns={'country': 'importer', 'efw': 'efw_importer'}),
    on=['importer', 'year'],
    how='left'
)

print(f"\n=== After Importer EFW Merge ===")
print(f"Observations: {len(df_gravity)}")
print(f"Missing importer EFW: {df_gravity['efw_importer'].isna().sum()}")

# =============================================================================
# MERGE EFW FOR EXPORTER
# =============================================================================

df_gravity = df_gravity.merge(
    df_efw.rename(columns={'country': 'exporter', 'efw': 'efw_exporter'}),
    on=['exporter', 'year'],
    how='left'
)

print(f"\n=== After Exporter EFW Merge ===")
print(f"Observations: {len(df_gravity)}")
print(f"Missing exporter EFW: {df_gravity['efw_exporter'].isna().sum()}")

# =============================================================================
# MERGE REMOTENESS FOR IMPORTER
# =============================================================================

df_gravity = df_gravity.merge(
    df_remote.rename(columns={'country': 'importer', 'remoteness': 'remoteness_importer'}),
    on=['importer', 'year'],
    how='left'
)

print(f"\n=== After Importer Remoteness Merge ===")
print(f"Observations: {len(df_gravity)}")
print(f"Missing importer remoteness: {df_gravity['remoteness_importer'].isna().sum()}")

# =============================================================================
# MERGE REMOTENESS FOR EXPORTER
# =============================================================================

df_gravity = df_gravity.merge(
    df_remote.rename(columns={'country': 'exporter', 'remoteness': 'remoteness_exporter'}),
    on=['exporter', 'year'],
    how='left'
)

print(f"\n=== After Exporter Remoteness Merge ===")
print(f"Observations: {len(df_gravity)}")
print(f"Missing exporter remoteness: {df_gravity['remoteness_exporter'].isna().sum()}")

# =============================================================================
# CHECK FOR DUPLICATES
# =============================================================================

duplicates = df_gravity.duplicated(subset=['year', 'importer', 'exporter']).sum()
if duplicates > 0:
    print(f"\nWARNING: {duplicates} duplicate observations found")
else:
    print(f"\nNo duplicate observations")

# =============================================================================
# CHECK FOR MISSING VALUES
# =============================================================================

print("\n=== Missing Values Summary ===")
key_vars = ['log_trade_real', 'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
            'contig', 'comlang_off', 'col_dep_ever', 'rta', 'both_eu',
            'efw_importer', 'efw_exporter', 'remoteness_importer', 'remoteness_exporter']
for var in key_vars:
    if var in df_gravity.columns:
        missing = df_gravity[var].isna().sum()
        print(f"{var}: {missing} missing ({missing/len(df_gravity)*100:.1f}%)")

# =============================================================================
# CREATE ANALYSIS SAMPLE (DROP MISSING)
# =============================================================================

df_analysis = df_gravity.dropna(subset=['log_trade_real', 'log_gdp_importer', 'log_gdp_exporter', 'log_dist', 'rta'])

print(f"\n=== Analysis Sample ===")
print(f"Original observations: {len(df_gravity)}")
print(f"Complete cases: {len(df_analysis)}")
print(f"Dropped: {len(df_gravity) - len(df_analysis)}")

# =============================================================================
# SAVE
# =============================================================================

df_gravity.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_full.csv", index=False)
df_analysis.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv", index=False)

print(f"\n=== FILES SAVED ===")
print(f"Full dataset: {len(df_gravity)} observations")
print(f"Analysis dataset: {len(df_analysis)} observations")

# =============================================================================
# QUICK SUMMARY STATS
# =============================================================================

print("\n=== Summary Statistics (Analysis Sample) ===")
summary_vars = ['log_trade_real', 'log_gdp_importer', 'log_gdp_exporter', 'log_dist',
                'rta', 'efw_importer', 'efw_exporter', 'remoteness_importer', 'remoteness_exporter']
print(df_analysis[summary_vars].describe())

print("\n=== RTA Distribution in Analysis Sample ===")
print(df_analysis['rta'].value_counts())