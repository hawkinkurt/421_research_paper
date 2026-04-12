"""build_trade_flows.py"""
"""Building bilateral trade flows from film data"""

import pandas as pd
import ast

# =============================================================================
# US MAJOR CORRECTIONS
# =============================================================================
# Rule: If any US major studio appears anywhere in production_companies,
# recode home_country = 'US' and lead_company = [that US major]
#
# Rationale: US majors control financing and distribution. When Warner Bros.,
# Fox, Universal, etc. are involved, they're typically the money behind the
# production, even if a local company is listed first (e.g., tax incentive
# vehicles like Village Roadshow, Ingenious Film Partners, WingNut Films).
#
# Corrections are loaded from: us_major_corrections.csv


def apply_us_major_corrections(df, corrections_path):
    """Apply US major studio corrections to film data from CSV file"""

    # Load corrections from CSV
    corrections_df = pd.read_csv(corrections_path)

    print(f"\n=== US Major Corrections ===")
    print(f"Corrections loaded from: {corrections_path}")
    print(f"Total corrections defined: {len(corrections_df)}")

    corrections_applied = 0
    corrections_not_found = []

    for _, row in corrections_df.iterrows():
        title = row['title']
        year = row['year']
        new_lead = row['corrected_lead_company']
        new_country = row['corrected_country']

        mask = (df['title'] == title) & (df['year'] == year)

        if mask.any():
            df.loc[mask, 'lead_company'] = new_lead
            df.loc[mask, 'home_country'] = new_country
            corrections_applied += 1
        else:
            corrections_not_found.append((title, year))

    print(f"Corrections applied: {corrections_applied}")

    if corrections_not_found:
        print(f"Corrections not found in dataset: {len(corrections_not_found)}")
        for title, year in corrections_not_found[:10]:  # Show first 10
            print(f"  - {title} ({year})")
        if len(corrections_not_found) > 10:
            print(f"  ... and {len(corrections_not_found) - 10} more")

    return df


# =============================================================================
# FUNCTION TO BUILD TRADE FLOWS
# =============================================================================

def build_trade_flows(df, dataset_name):
    """Build bilateral trade flows from film data"""

    print(f"\n{'='*60}")
    print(f"PROCESSING: {dataset_name}")
    print(f"{'='*60}")

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

# File paths
DATA_DIR = "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed"
CORRECTIONS_PATH = "C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/base/us_major_corrections.csv"

# Load all mapped films (corrected)
df_all = pd.read_csv(f"{DATA_DIR}/films_cleaned.csv")
print(f"All mapped films loaded: {len(df_all)}")

# Load high confidence only (corrected)
df_high = pd.read_csv(f"{DATA_DIR}/films_cleaned_high_confidence.csv")
print(f"High confidence films loaded: {len(df_high)}")

# Apply US major corrections BEFORE building trade flows
print("\n" + "="*60)
print("APPLYING US MAJOR CORRECTIONS")
print("="*60)

df_all = apply_us_major_corrections(df_all, CORRECTIONS_PATH)
df_high = apply_us_major_corrections(df_high, CORRECTIONS_PATH)

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
df_flows_all.to_csv(f"{DATA_DIR}/trade_flows_by_film.csv", index=False)
df_bilateral_all.to_csv(f"{DATA_DIR}/trade_flows_bilateral.csv", index=False)

# Save high confidence trade flows
df_flows_high.to_csv(f"{DATA_DIR}/trade_flows_by_film_high_conf.csv", index=False)
df_bilateral_high.to_csv(f"{DATA_DIR}/trade_flows_bilateral_high_conf.csv", index=False)

# Also save the corrected films dataframes for reference
df_all.to_csv(f"{DATA_DIR}/films_cleaned_corrected.csv", index=False)
df_high.to_csv(f"{DATA_DIR}/films_cleaned_high_confidence_corrected.csv", index=False)

print(f"\n=== ALL MAPPED FILES ===")
print(f"Film-level flows: {len(df_flows_all)} records")
print(f"Bilateral aggregated: {len(df_bilateral_all)} pair-year observations")

print(f"\n=== HIGH CONFIDENCE FILES ===")
print(f"Film-level flows: {len(df_flows_high)} records")
print(f"Bilateral aggregated: {len(df_bilateral_high)} pair-year observations")

print(f"\n=== CORRECTED FILM FILES SAVED ===")
print(f"films_cleaned_corrected.csv")
print(f"films_cleaned_high_confidence_corrected.csv")