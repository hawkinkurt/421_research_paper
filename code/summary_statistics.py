"""summary_statistics"""
import pandas as pd
import numpy as np

# Load the analysis dataset
df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_analysis.csv")

print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# Define variables to summarise
variables = {
    'trade_value': 'Trade Value (nominal USD)',
    'log_trade_real': 'Log Trade Value (real 2018 USD)',
    'gdp_importer': 'GDP Importer (USD)',
    'gdp_exporter': 'GDP Exporter (USD)',
    'log_gdp_importer': 'Log GDP Importer',
    'log_gdp_exporter': 'Log GDP Exporter',
    'dist': 'Distance (km)',
    'log_dist': 'Log Distance',
    'contig': 'Contiguity (shared border)',
    'comlang_off': 'Common Official Language',
    'col_dep_ever': 'Colonial Relationship',
    'rta': 'Regional Trade Agreement',
    'efw_importer': 'Economic Freedom (Importer)',
    'efw_exporter': 'Economic Freedom (Exporter)',
    'remoteness_importer': 'Remoteness (Importer)',
    'remoteness_exporter': 'Remoteness (Exporter)',
    'incentive_exporter': 'Film Incentive (Exporter)',
    'incentive_importer': 'Film Incentive (Importer)',
    'num_films': 'Number of Films'
}

# Calculate statistics
stats_list = []
for var, label in variables.items():
    if var in df.columns:
        stats_list.append({
            'Variable': label,
            'N': df[var].notna().sum(),
            'Mean': df[var].mean(),
            'Std Dev': df[var].std(),
            'Min': df[var].min(),
            'Max': df[var].max()
        })

stats_df = pd.DataFrame(stats_list)

# Print formatted table
print(f"\n{'Variable':<35} {'N':>6} {'Mean':>12} {'Std Dev':>12} {'Min':>12} {'Max':>12}")
print("-" * 91)

for _, row in stats_df.iterrows():
    # Format large numbers with commas, small numbers with decimals
    mean_val = row['Mean']
    std_val = row['Std Dev']
    min_val = row['Min']
    max_val = row['Max']

    if abs(mean_val) > 1000:
        mean_str = f"{mean_val:>12,.0f}"
        std_str = f"{std_val:>12,.0f}"
        min_str = f"{min_val:>12,.0f}"
        max_str = f"{max_val:>12,.0f}"
    else:
        mean_str = f"{mean_val:>12.3f}"
        std_str = f"{std_val:>12.3f}"
        min_str = f"{min_val:>12.3f}"
        max_str = f"{max_val:>12.3f}"

    print(f"{row['Variable']:<35} {int(row['N']):>6} {mean_str} {std_str} {min_str} {max_str}")

print("-" * 91)

# Additional statistics
print(f"\n{'=' * 70}")
print("ADDITIONAL DATASET INFORMATION")
print("=" * 70)
print(f"Total observations: {len(df)}")
print(f"Unique importers: {df['importer'].nunique()}")
print(f"Unique exporters: {df['exporter'].nunique()}")
print(f"Unique country pairs: {df.groupby(['importer', 'exporter']).ngroups}")
print(f"Year range: {int(df['year'].min())} to {int(df['year'].max())}")
print(f"Total trade value (nominal): ${df['trade_value_nominal'].sum():,.0f}")
print(f"Total films: {int(df['num_films'].sum())}")

# Save to CSV for easy export to LaTeX later
stats_df.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/output/tables/summary_statistics.csv", index=False)
print(f"\nSaved to: output/tables/summary_statistics.csv")