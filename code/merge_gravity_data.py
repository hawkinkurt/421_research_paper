import pandas as pd
import numpy as np

# =============================================================================
# LOAD DATA
# =============================================================================

df_trade = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_bilateral_real.csv")
df_gdp = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gdp_cleaned.csv")
df_dist = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/distance_cleaned.csv")

print("=== Data Loaded ===")
print(f"Trade observations: {len(df_trade)}")
print(f"GDP observations: {len(df_gdp)}")
print(f"Distance pairs: {len(df_dist)}")

# =============================================================================
# CHECK COLUMN NAMES
# =============================================================================

print("\n=== Column Names ===")
print(f"Trade: {df_trade.columns.tolist()}")
print(f"GDP: {df_gdp.columns.tolist()}")
print(f"Distance: {df_dist.columns.tolist()}")

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
# MERGE DISTANCE VARIABLES
# =============================================================================

df_gravity = df_gravity.merge(
    df_dist,
    left_on=['importer', 'exporter'],
    right_on=['iso_o', 'iso_d'],
    how='left'
)

print(f"\n=== After Distance Merge ===")
print(f"Observations: {len(df_gravity)}")
print(f"Missing distance: {df_gravity['dist'].isna().sum()}")

# =============================================================================
# CHECK FOR MISSING VALUES
# =============================================================================

print("\n=== Missing Values Summary ===")
key_vars = ['log_trade_real', 'log_gdp_importer', 'log_gdp_exporter', 'log_dist', 'contig', 'comlang_off']
for var in key_vars:
    if var in df_gravity.columns:
        missing = df_gravity[var].isna().sum()
        print(f"{var}: {missing} missing ({missing/len(df_gravity)*100:.1f}%)")

# =============================================================================
# CREATE ANALYSIS SAMPLE (DROP MISSING)
# =============================================================================

df_analysis = df_gravity.dropna(subset=['log_trade_real', 'log_gdp_importer', 'log_gdp_exporter', 'log_dist'])

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
print(df_analysis[['log_trade_real', 'log_gdp_importer', 'log_gdp_exporter', 'log_dist']].describe())