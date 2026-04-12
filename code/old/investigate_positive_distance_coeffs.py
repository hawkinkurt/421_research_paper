"""investigate_positive_distance_coeffs"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")

print("=" * 70)
print("DISTANCE COEFFICIENT INVESTIGATION")
print("=" * 70)

# =============================================================================
# 1. WHICH COUNTRY PAIRS DOMINATE THE SAMPLE?
# =============================================================================

print("\n=== TOP 20 COUNTRY PAIRS BY OBSERVATION COUNT ===")
pair_counts = df.groupby(['importer', 'exporter']).size().reset_index(name='obs_count')
pair_counts = pair_counts.sort_values('obs_count', ascending=False).head(20)
print(pair_counts.to_string(index=False))

print("\n=== TOP 20 COUNTRY PAIRS BY TOTAL TRADE VALUE ===")
pair_value = df.groupby(['importer', 'exporter'])['trade_value'].sum().reset_index()
pair_value = pair_value.sort_values('trade_value', ascending=False).head(20)
pair_value['trade_value_millions'] = pair_value['trade_value'] / 1e6
print(pair_value[['importer', 'exporter', 'trade_value_millions']].to_string(index=False))

# =============================================================================
# 2. DISTANCE DISTRIBUTION BY MAJOR IMPORTERS
# =============================================================================

print("\n=== AVERAGE DISTANCE BY IMPORTER (Top 10 by obs) ===")
top_importers = df.groupby('importer').size().nlargest(10).index.tolist()
for imp in top_importers:
    subset = df[df['importer'] == imp]
    avg_dist = subset['dist'].mean()
    avg_trade = subset['trade_value'].mean()
    n_obs = len(subset)
    print(f"{imp}: avg distance = {avg_dist:,.0f} km, avg trade = ${avg_trade/1e6:.1f}M, n = {n_obs}")

# =============================================================================
# 3. CORRELATION: DISTANCE VS TRADE BY IMPORTER
# =============================================================================

print("\n=== DISTANCE-TRADE CORRELATION BY IMPORTER ===")
print("(Positive = more distant partners have higher trade)")
for imp in top_importers:
    subset = df[df['importer'] == imp]
    if len(subset) > 10:
        corr = subset['log_dist'].corr(subset['log_trade_real'])
        print(f"{imp}: correlation = {corr:.3f}")

# =============================================================================
# 4. US-SPECIFIC ANALYSIS (likely dominant)
# =============================================================================

print("\n=== US AS IMPORTER: TOP EXPORT PARTNERS ===")
us_imports = df[df['importer'] == 'US'].groupby('exporter').agg({
    'trade_value': 'sum',
    'dist': 'first',
    'num_films': 'sum'
}).reset_index()
us_imports = us_imports.sort_values('trade_value', ascending=False).head(15)
us_imports['trade_millions'] = us_imports['trade_value'] / 1e6
print(us_imports[['exporter', 'trade_millions', 'dist', 'num_films']].to_string(index=False))

# =============================================================================
# 5. ENGLISH LANGUAGE EFFECT
# =============================================================================

print("\n=== DISTANCE BY LANGUAGE STATUS ===")
english_pairs = df[df['comlang_off'] == 1]
non_english = df[df['comlang_off'] == 0]

print(f"Common language pairs:")
print(f"  Avg distance: {english_pairs['dist'].mean():,.0f} km")
print(f"  Avg trade: ${english_pairs['trade_value'].mean()/1e6:.1f}M")
print(f"  N: {len(english_pairs)}")

print(f"\nNo common language:")
print(f"  Avg distance: {non_english['dist'].mean():,.0f} km")
print(f"  Avg trade: ${non_english['trade_value'].mean()/1e6:.1f}M")
print(f"  N: {len(non_english)}")

# =============================================================================
# 6. TIME TREND IN DISTANCE
# =============================================================================

print("\n=== AVERAGE DISTANCE OVER TIME ===")
yearly_dist = df.groupby('year').agg({
    'dist': 'mean',
    'trade_value': 'sum',
    'log_trade_real': 'count'
}).reset_index()
yearly_dist.columns = ['year', 'avg_dist', 'total_trade', 'n_obs']
print(yearly_dist.to_string(index=False))