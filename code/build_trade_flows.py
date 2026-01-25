"""build_trade_flows.py"""
import pandas as pd
import numpy as np
import ast

# =============================================================================
# FUNCTION TO BUILD TRADE FLOWS
# =============================================================================

def build_trade_flows(df, dataset_name):
    """Build bilateral trade flows from film data"""

    print(f"\n{'='*60}")
    print(f"PROCESSING: {dataset_name}")
    print(f"{'='*60}")

    # Remove films with no lead company or home country
    print(f"Films with valid data: {len(df)}")

    # Parse country_codes if stored as string
    def parse_country_codes(value):
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        try:
            return ast.literal_eval(value)
        except:
            return []

    df['country_codes'] = df['country_codes'].apply(parse_country_codes)

    # Calculate budget share per country
    df['budget_per_country'] = df['budget'] / df['num_countries']

    print(f"\n=== Budget Statistics ===")
    print(f"Total budget (all films): ${df['budget'].sum():,.0f}")
    print(f"Mean budget: ${df['budget'].mean():,.0f}")
    print(f"Median budget: ${df['budget'].median():,.0f}")

    # Create bilateral trade flows
    trade_flows = []

    for idx, row in df.iterrows():
        home = row['home_country']
        budget_share = row['budget_per_country']
        year = row['year']
        title = row['title']

        for production_country in row['country_codes']:
            if production_country == home:
                continue

            trade_flows.append({
                'year': int(year),
                'importer': home,
                'exporter': production_country,
                'trade_value': budget_share,
                'film_title': title
            })

    df_flows = pd.DataFrame(trade_flows)
    print(f"\n=== Trade Flow Records ===")
    print(f"Total bilateral records: {len(df_flows)}")
    print(f"Unique country pairs: {df_flows.groupby(['importer', 'exporter']).ngroups}")

    # Aggregate by country pair and year
    df_bilateral = df_flows.groupby(['year', 'importer', 'exporter']).agg(
        trade_value=('trade_value', 'sum'),
        num_films=('film_title', 'count')
    ).reset_index()

    print(f"\n=== Aggregated Bilateral Data ===")
    print(f"Total observations (pair-years): {len(df_bilateral)}")
    print(f"Year range: {df_bilateral['year'].min()} to {df_bilateral['year'].max()}")
    print(f"Unique importers: {df_bilateral['importer'].nunique()}")
    print(f"Unique exporters: {df_bilateral['exporter'].nunique()}")

    print("\n=== Top 10 Trade Flows (by value) ===")
    print(df_bilateral.nlargest(10, 'trade_value')[['year', 'importer', 'exporter', 'trade_value', 'num_films']])

    return df_flows, df_bilateral

# =============================================================================
# LOAD AND PROCESS BOTH DATASETS
# =============================================================================

# Load all mapped films (corrected)
df_all = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_with_home_country_mapped_corrected.csv")
print(f"All mapped films loaded: {len(df_all)}")

# Load high confidence only (corrected)
df_high = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_with_home_country_high_confidence_corrected.csv")
print(f"High confidence films loaded: {len(df_high)}")

# Build trade flows for all mapped films
df_flows_all, df_bilateral_all = build_trade_flows(df_all, "ALL MAPPED FILMS")

# Build trade flows for high confidence only
df_flows_high, df_bilateral_high = build_trade_flows(df_high, "HIGH CONFIDENCE ONLY")

# =============================================================================
# SAVE OUTPUT
# =============================================================================

print(f"\n{'='*60}")
print("SAVING FILES")
print(f"{'='*60}")

# Save all mapped films trade flows
df_flows_all.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_by_film_all.csv", index=False)
df_bilateral_all.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_bilateral_all.csv", index=False)

# Save high confidence trade flows
df_flows_high.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_by_film_high_conf.csv", index=False)
df_bilateral_high.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/trade_flows_bilateral_high_conf.csv", index=False)

print(f"\n=== ALL MAPPED FILES ===")
print(f"Film-level flows: {len(df_flows_all)} records")
print(f"Bilateral aggregated: {len(df_bilateral_all)} pair-year observations")

print(f"\n=== HIGH CONFIDENCE FILES ===")
print(f"Film-level flows: {len(df_flows_high)} records")
print(f"Bilateral aggregated: {len(df_bilateral_high)} pair-year observations")