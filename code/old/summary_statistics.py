"""summary_statistics.py"""
"""Generates summary statistics for gravity model dataset"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")

print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# =============================================================================
# DEFINE VARIABLES
# =============================================================================

# Continuous variables
continuous_vars = {
    'trade_value': 'Trade Value (real 2018 USD)',
    'trade_value_nominal': 'Trade Value (nominal USD)',
    'log_trade_real': 'Log Trade Value (real)',
    'gdp_importer': 'GDP Importer (USD)',
    'gdp_exporter': 'GDP Exporter (USD)',
    'log_gdp_importer': 'Log GDP Importer',
    'log_gdp_exporter': 'Log GDP Exporter',
    'dist': 'Distance (km)',
    'log_dist': 'Log Distance',
    'efw_importer': 'Economic Freedom (Importer)',
    'efw_exporter': 'Economic Freedom (Exporter)',
    'remoteness_importer': 'Remoteness (Importer)',
    'remoteness_exporter': 'Remoteness (Exporter)',
    'num_films': 'Number of Films'
}

# Binary/dummy variables
binary_vars = {
    'contig': 'Contiguity (shared border)',
    'comlang_off': 'Common Official Language',
    'col_dep_ever': 'Colonial Relationship',
    'rta': 'Regional Trade Agreement',
    'both_eu': 'Both EU Members',
    'incentive_exporter': 'Film Incentive (Exporter)',
    'incentive_importer': 'Film Incentive (Importer)'
}

# =============================================================================
# CHECK FOR MISSING VARIABLES
# =============================================================================

missing_vars = []
for var in list(continuous_vars.keys()) + list(binary_vars.keys()):
    if var not in df.columns:
        missing_vars.append(var)

if missing_vars:
    print(f"\nWARNING: Missing variables: {missing_vars}")

# =============================================================================
# CONTINUOUS VARIABLES SUMMARY
# =============================================================================

print("\n=== Continuous Variables ===")
print(f"\n{'Variable':<35} {'N':>6} {'Mean':>12} {'Std Dev':>10} {'Median':>12} {'Min':>12} {'Max':>12}")
print("-" * 103)

continuous_stats = []
for var, label in continuous_vars.items():
    if var not in df.columns:
        continue

    stats = {
        'Variable': label,
        'N': df[var].notna().sum(),
        'Mean': df[var].mean(),
        'Std Dev': df[var].std(),
        'Median': df[var].median(),
        'Min': df[var].min(),
        'Max': df[var].max()
    }
    continuous_stats.append(stats)

    # Format output based on variable type
    n = int(stats['N'])

    if var == 'num_films':
        # Count variable - show as integers
        print(f"{label:<35} {n:>6} {stats['Mean']:>12.2f} {stats['Std Dev']:>10.2f} "
              f"{stats['Median']:>12.0f} {stats['Min']:>12.0f} {stats['Max']:>12.0f}")
    elif abs(stats['Mean']) > 10000:
        # Large numbers - use comma formatting
        print(f"{label:<35} {n:>6} {stats['Mean']:>12,.0f} {stats['Std Dev']:>10,.0f} "
              f"{stats['Median']:>12,.0f} {stats['Min']:>12,.0f} {stats['Max']:>12,.0f}")
    else:
        # Standard decimals
        print(f"{label:<35} {n:>6} {stats['Mean']:>12.3f} {stats['Std Dev']:>10.3f} "
              f"{stats['Median']:>12.3f} {stats['Min']:>12.3f} {stats['Max']:>12.3f}")

print("-" * 103)

# =============================================================================
# BINARY VARIABLES SUMMARY
# =============================================================================

print("\n=== Binary Variables ===")
print(f"\n{'Variable':<35} {'N':>6} {'N = 1':>8} {'N = 0':>8} {'Proportion':>12}")
print("-" * 75)

binary_stats = []
for var, label in binary_vars.items():
    if var not in df.columns:
        continue

    n_total = df[var].notna().sum()
    n_one = int(df[var].sum())
    n_zero = int(n_total - n_one)
    proportion = df[var].mean()

    stats = {
        'Variable': label,
        'N': n_total,
        'N = 1': n_one,
        'N = 0': n_zero,
        'Proportion': proportion
    }
    binary_stats.append(stats)

    print(f"{label:<35} {n_total:>6} {n_one:>8} {n_zero:>8} {proportion:>12.3f}")

print("-" * 75)

# =============================================================================
# ADDITIONAL DATASET INFORMATION
# =============================================================================

print(f"\n{'=' * 70}")
print("DATASET INFORMATION")
print("=" * 70)
print(f"Total observations (pair-years): {len(df)}")
print(f"Unique importers: {df['importer'].nunique()}")
print(f"Unique exporters: {df['exporter'].nunique()}")
print(f"Unique country pairs: {df.groupby(['importer', 'exporter']).ngroups}")
print(f"Year range: {int(df['year'].min())} - {int(df['year'].max())}")

if 'trade_value_nominal' in df.columns:
    print(f"Total trade (nominal): ${df['trade_value_nominal'].sum():,.0f}")
if 'trade_value' in df.columns:
    print(f"Total trade (real 2018 USD): ${df['trade_value'].sum():,.0f}")
print(f"Total film flows: {int(df['num_films'].sum())}")

# =============================================================================
# BILATERAL PAIR SUMMARY (AGGREGATED ACROSS YEARS)
# =============================================================================

print(f"\n{'=' * 70}")
print("BILATERAL PAIR SUMMARY (aggregated across all years)")
print("=" * 70)

pair_summary = df.groupby(['importer', 'exporter']).agg(
    total_films=('num_films', 'sum'),
    total_trade_nominal=('trade_value_nominal', 'sum'),
    total_trade_real=('trade_value', 'sum'),
    years_active=('year', 'count'),
    first_year=('year', 'min'),
    last_year=('year', 'max')
).reset_index()

print(f"\nUnique country pairs: {len(pair_summary)}")

print(f"\n=== Films per Country Pair ===")
print(f"{'Statistic':<15} {'Value':>12}")
print("-" * 30)
print(f"{'Mean':<15} {pair_summary['total_films'].mean():>12.2f}")
print(f"{'Std Dev':<15} {pair_summary['total_films'].std():>12.2f}")
print(f"{'Median':<15} {pair_summary['total_films'].median():>12.0f}")
print(f"{'Min':<15} {pair_summary['total_films'].min():>12.0f}")
print(f"{'Max':<15} {pair_summary['total_films'].max():>12.0f}")

print(f"\n=== Years Active per Country Pair ===")
print(f"{'Statistic':<15} {'Value':>12}")
print("-" * 30)
print(f"{'Mean':<15} {pair_summary['years_active'].mean():>12.2f}")
print(f"{'Median':<15} {pair_summary['years_active'].median():>12.0f}")
print(f"{'Min':<15} {pair_summary['years_active'].min():>12.0f}")
print(f"{'Max':<15} {pair_summary['years_active'].max():>12.0f}")

print(f"\n=== Top 10 Country Pairs by Total Films ===")
top_pairs = pair_summary.nlargest(10, 'total_films')[
    ['importer', 'exporter', 'total_films', 'total_trade_real', 'years_active']
]
top_pairs['total_trade_real'] = top_pairs['total_trade_real'].apply(lambda x: f"${x:,.0f}")
top_pairs.columns = ['Importer', 'Exporter', 'Films', 'Trade (real)', 'Years']
print(top_pairs.to_string(index=False))

# =============================================================================
# SUMMARY STATISTICS AT BILATERAL PAIR LEVEL
# =============================================================================

print(f"\n{'=' * 70}")
print("SUMMARY STATISTICS (at bilateral pair level, aggregated across years)")
print("=" * 70)

# Aggregate to pair level
pair_stats = df.groupby(['importer', 'exporter']).agg(
    total_films=('num_films', 'sum'),
    total_trade_real=('trade_value', 'sum'),
    total_trade_nominal=('trade_value_nominal', 'sum'),
    mean_dist=('dist', 'mean'),
    mean_log_dist=('log_dist', 'mean'),
    contig=('contig', 'max'),
    comlang_off=('comlang_off', 'max'),
    col_dep_ever=('col_dep_ever', 'max'),
    rta_ever=('rta', 'max'),
    both_eu_ever=('both_eu', 'max'),
    incentive_exporter_ever=('incentive_exporter', 'max'),
    incentive_importer_ever=('incentive_importer', 'max'),
    years_active=('year', 'count')
).reset_index()

# Add log trade
pair_stats['log_total_trade_real'] = np.log(pair_stats['total_trade_real'])

print(f"\nNumber of unique country pairs: {len(pair_stats)}")

# Continuous variables at pair level
pair_continuous = {
    'total_films': 'Total Films',
    'total_trade_real': 'Total Trade (real 2018 USD)',
    'log_total_trade_real': 'Log Total Trade (real)',
    'mean_dist': 'Distance (km)',
    'mean_log_dist': 'Log Distance',
    'years_active': 'Years Active'
}

print(f"\n=== Continuous Variables (Pair Level) ===")
print(f"\n{'Variable':<30} {'N':>6} {'Mean':>14} {'Std Dev':>12} {'Median':>14} {'Min':>12} {'Max':>14}")
print("-" * 108)

pair_continuous_stats = []
for var, label in pair_continuous.items():
    if var not in pair_stats.columns:
        continue

    stats = {
        'Variable': label,
        'N': pair_stats[var].notna().sum(),
        'Mean': pair_stats[var].mean(),
        'Std Dev': pair_stats[var].std(),
        'Median': pair_stats[var].median(),
        'Min': pair_stats[var].min(),
        'Max': pair_stats[var].max()
    }
    pair_continuous_stats.append(stats)

    n = int(stats['N'])

    if var in ['total_films', 'years_active']:
        print(f"{label:<30} {n:>6} {stats['Mean']:>14.2f} {stats['Std Dev']:>12.2f} "
              f"{stats['Median']:>14.0f} {stats['Min']:>12.0f} {stats['Max']:>14.0f}")
    elif abs(stats['Mean']) > 10000:
        print(f"{label:<30} {n:>6} {stats['Mean']:>14,.0f} {stats['Std Dev']:>12,.0f} "
              f"{stats['Median']:>14,.0f} {stats['Min']:>12,.0f} {stats['Max']:>14,.0f}")
    else:
        print(f"{label:<30} {n:>6} {stats['Mean']:>14.3f} {stats['Std Dev']:>12.3f} "
              f"{stats['Median']:>14.3f} {stats['Min']:>12.3f} {stats['Max']:>14.3f}")

print("-" * 108)

# Binary variables at pair level (ever = 1 if true in any year)
pair_binary = {
    'contig': 'Contiguity (shared border)',
    'comlang_off': 'Common Official Language',
    'col_dep_ever': 'Colonial Relationship',
    'rta_ever': 'Regional Trade Agreement (ever)',
    'both_eu_ever': 'Both EU Members (ever)',
    'incentive_exporter_ever': 'Film Incentive Exporter (ever)',
    'incentive_importer_ever': 'Film Incentive Importer (ever)'
}

print(f"\n=== Binary Variables (Pair Level) ===")
print(f"\n{'Variable':<35} {'N':>6} {'N = 1':>8} {'N = 0':>8} {'Proportion':>12}")
print("-" * 75)

pair_binary_stats = []
for var, label in pair_binary.items():
    if var not in pair_stats.columns:
        continue

    n_total = pair_stats[var].notna().sum()
    n_one = int(pair_stats[var].sum())
    n_zero = int(n_total - n_one)
    proportion = pair_stats[var].mean()

    stats = {
        'Variable': label,
        'N': n_total,
        'N = 1': n_one,
        'N = 0': n_zero,
        'Proportion': proportion
    }
    pair_binary_stats.append(stats)

    print(f"{label:<35} {n_total:>6} {n_one:>8} {n_zero:>8} {proportion:>12.3f}")

print("-" * 75)

# =============================================================================
# SAVE OUTPUT
# =============================================================================

# Ensure output directory exists
output_dir = "C:/Users/kurtl/PycharmProjects/gravity_model/output/tables"
os.makedirs(output_dir, exist_ok=True)

# Combine stats for export
continuous_df = pd.DataFrame(continuous_stats)
continuous_df['Type'] = 'Continuous'

binary_df = pd.DataFrame(binary_stats)
binary_df['Type'] = 'Binary'

all_stats = pd.concat([continuous_df, binary_df], ignore_index=True)
all_stats.to_csv(f"{output_dir}/summary_statistics.csv", index=False)

# Save bilateral pair summary
pair_summary.to_csv(f"{output_dir}/bilateral_pair_summary.csv", index=False)

# Save pair-level statistics
pair_level_continuous_df = pd.DataFrame(pair_continuous_stats)
pair_level_continuous_df['Type'] = 'Continuous'
pair_level_continuous_df['Level'] = 'Pair'

pair_level_binary_df = pd.DataFrame(pair_binary_stats)
pair_level_binary_df['Type'] = 'Binary'
pair_level_binary_df['Level'] = 'Pair'

pair_level_stats = pd.concat([pair_level_continuous_df, pair_level_binary_df], ignore_index=True)
pair_level_stats.to_csv(f"{output_dir}/summary_statistics_pair_level.csv", index=False)

print(f"\nSaved to:")
print(f"  {output_dir}/summary_statistics.csv")
print(f"  {output_dir}/bilateral_pair_summary.csv")
print(f"  {output_dir}/summary_statistics_pair_level.csv")