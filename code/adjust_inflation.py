import pandas as pd
import numpy as np

# =============================================================================
# LOAD DATA
# =============================================================================

df_bilateral = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_bilateral.csv")
df_flows = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_by_film.csv")
df_cpi = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/world_bank_cpi.csv")

print("=== Data Loaded ===")
print(f"Bilateral observations: {len(df_bilateral)}")
print(f"Film-level records: {len(df_flows)}")
print(f"CPI years available: {df_cpi['date'].min()} to {df_cpi['date'].max()}")

# =============================================================================
# PREPARE CPI DATA
# =============================================================================

df_cpi = df_cpi.rename(columns={'date': 'year', 'value': 'cpi'})
df_cpi = df_cpi[['year', 'cpi']].dropna()

base_year = 2018
base_cpi = df_cpi[df_cpi['year'] == base_year]['cpi'].values[0]

df_cpi['adjustment_factor'] = base_cpi / df_cpi['cpi']

print("\n=== CPI Adjustment Factors (to 2018 USD) ===")
print(df_cpi[df_cpi['year'].isin([1990, 1995, 2000, 2005, 2010, 2015, 2018])])

# =============================================================================
# ADJUST BILATERAL DATA
# =============================================================================

df_bilateral = df_bilateral.merge(df_cpi[['year', 'adjustment_factor']], on='year', how='left')
df_bilateral['trade_value_nominal'] = df_bilateral['trade_value']
df_bilateral['trade_value_real'] = df_bilateral['trade_value'] * df_bilateral['adjustment_factor']

print("\n=== Bilateral Data After Inflation Adjustment ===")
print(f"Total nominal trade: ${df_bilateral['trade_value_nominal'].sum():,.0f}")
print(f"Total real trade (2018 USD): ${df_bilateral['trade_value_real'].sum():,.0f}")

# =============================================================================
# ADJUST FILM-LEVEL DATA
# =============================================================================

df_flows = df_flows.merge(df_cpi[['year', 'adjustment_factor']], on='year', how='left')
df_flows['trade_value_nominal'] = df_flows['trade_value']
df_flows['trade_value_real'] = df_flows['trade_value'] * df_flows['adjustment_factor']

# =============================================================================
# CREATE LOG VARIABLES FOR GRAVITY MODEL
# =============================================================================

df_bilateral['log_trade_real'] = np.log(df_bilateral['trade_value_real'] + 1)

print("\n=== Log Trade Variable ===")
print(df_bilateral['log_trade_real'].describe())

# =============================================================================
# SAVE ADJUSTED DATA
# =============================================================================

df_bilateral = df_bilateral.drop(columns=['adjustment_factor', 'trade_value'])
df_flows = df_flows.drop(columns=['adjustment_factor', 'trade_value'])

df_bilateral = df_bilateral.rename(columns={'trade_value_real': 'trade_value'})
df_flows = df_flows.rename(columns={'trade_value_real': 'trade_value'})

df_bilateral.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_bilateral_real.csv", index=False)
df_flows.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_by_film_real.csv", index=False)

print(f"\n=== FILES SAVED ===")
print(f"Bilateral (inflation adjusted): {len(df_bilateral)} observations")
print(f"Film-level (inflation adjusted): {len(df_flows)} records")