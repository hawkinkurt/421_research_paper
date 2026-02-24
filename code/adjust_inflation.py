"""adjust_inflation.py"""
"""Adjusting trade values for inflation using US CPI (base year: 2018)"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_YEAR = 2018

# =============================================================================
# LOAD DATA
# =============================================================================

df_bilateral = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_bilateral.csv")
df_flows = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_by_film.csv")
df_cpi = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/world_bank_cpi.csv")

print("=== DATA LOADED ===")
print(f"Bilateral observations: {len(df_bilateral)}")
print(f"Film-level records: {len(df_flows)}")
print(f"CPI years available: {df_cpi['date'].min()} - {df_cpi['date'].max()}")

# Remove incomplete records
bilateral_before = len(df_bilateral)
flows_before = len(df_flows)

df_bilateral = df_bilateral.dropna(subset=['exporter', 'importer'])
df_flows = df_flows.dropna(subset=['exporter', 'importer'])

bilateral_dropped = bilateral_before - len(df_bilateral)
flows_dropped = flows_before - len(df_flows)

if bilateral_dropped > 0 or flows_dropped > 0:
    print(f"\nDropped incomplete records:")
    print(f"  Bilateral: {bilateral_dropped}")
    print(f"  Film-level: {flows_dropped}")

# =============================================================================
# PREPARE CPI ADJUSTMENT FACTORS
# =============================================================================

df_cpi = df_cpi.rename(columns={'date': 'year', 'value': 'cpi'})
df_cpi = df_cpi[['year', 'cpi']].dropna()

base_cpi = df_cpi.loc[df_cpi['year'] == BASE_YEAR, 'cpi'].values[0]
df_cpi['adjustment_factor'] = base_cpi / df_cpi['cpi']

print(f"\n=== CPI ADJUSTMENT FACTORS (to {BASE_YEAR} USD) ===")
sample_years = [1990, 1995, 2000, 2005, 2010, 2015, BASE_YEAR]
print(df_cpi[df_cpi['year'].isin(sample_years)].to_string(index=False))

# =============================================================================
# ADJUST TRADE VALUES
# =============================================================================

def adjust_for_inflation(df, cpi_df):
    """Apply inflation adjustment to trade values."""
    df = df.merge(cpi_df[['year', 'adjustment_factor']], on='year', how='left')
    df['trade_value_nominal'] = df['trade_value']
    df['trade_value'] = df['trade_value'] * df['adjustment_factor']
    df = df.drop(columns=['adjustment_factor'])
    return df

df_bilateral = adjust_for_inflation(df_bilateral, df_cpi)
df_flows = adjust_for_inflation(df_flows, df_cpi)

# Create log variable for gravity model
df_bilateral['log_trade_real'] = np.log(df_bilateral['trade_value'] + 1)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n=== INFLATION ADJUSTMENT RESULTS ===")
print(f"Total nominal trade: ${df_bilateral['trade_value_nominal'].sum():,.0f}")
print(f"Total real trade ({BASE_YEAR} USD): ${df_bilateral['trade_value'].sum():,.0f}")

print("\n=== LOG TRADE VARIABLE ===")
print(df_bilateral['log_trade_real'].describe())

# Check for missing adjustments
missing_bilateral = df_bilateral['trade_value'].isna().sum()
missing_flows = df_flows['trade_value'].isna().sum()
if missing_bilateral > 0 or missing_flows > 0:
    print(f"\nWARNING: Missing CPI data for some years")
    print(f"  Bilateral missing: {missing_bilateral}")
    print(f"  Film-level missing: {missing_flows}")

# =============================================================================
# SAVE
# =============================================================================

df_bilateral.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_bilateral_real.csv",
    index=False
)
df_flows.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_by_film_real.csv",
    index=False
)

print(f"\n=== FILES SAVED ===")
print(f"Bilateral: {len(df_bilateral)} observations")
print(f"Film-level: {len(df_flows)} records")

small_flows = df_flows[df_flows['trade_value'] < 10000]
print(small_flows[['film_title', 'year', 'trade_value_nominal']].drop_duplicates())