"""film_data_cleaning.py"""
"""Cleaning film dataset"""

import pandas as pd
import ast

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/movie_data_1.csv")
print(f"Total films loaded: {len(df)}")

# Load studio mappings
high_confidence = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/studio_mappings_high_confidence.csv")
low_confidence = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/studio_mappings_low_confidence.csv")

high_conf_dict = dict(zip(high_confidence['company_name'], high_confidence['home_country']))
low_conf_dict = dict(zip(low_confidence['company_name'], low_confidence['home_country']))

print(f"High confidence mappings: {len(high_conf_dict)}")
print(f"Low confidence mappings: {len(low_conf_dict)}")

# =============================================================================
# CLEAN AND FILTER DATA
# =============================================================================

df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

df_clean = df[
    (df['budget'] > 0) &
    (df['production_companies'].notna()) &
    (df['production_countries'].notna())
].copy()
print(f"Films with usable data: {len(df_clean)}")

df_clean['year'] = pd.to_datetime(df_clean['release_date'], errors='coerce').dt.year

# Filter to 1990-2018 to match other datasets
df_clean = df_clean[(df_clean['year'] >= 1990) & (df_clean['year'] <= 2018)]
print(f"Films in 1990-2018: {len(df_clean)}")

# =============================================================================
# PARSE PRODUCTION DATA
# =============================================================================

def parse_list_column(value):
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value)
    except:
        return []

df_clean['production_companies_parsed'] = df_clean['production_companies'].apply(parse_list_column)
df_clean['production_countries_parsed'] = df_clean['production_countries'].apply(parse_list_column)

def get_country_codes(parsed_list):
    return [item['iso_3166_1'] for item in parsed_list if 'iso_3166_1' in item]

df_clean['country_codes'] = df_clean['production_countries_parsed'].apply(get_country_codes)
df_clean['num_countries'] = df_clean['country_codes'].apply(len)

# Filter to international co-productions (2+ countries)
df_multi = df_clean[df_clean['num_countries'] >= 2].copy()
print(f"Films with 2+ production countries: {len(df_multi)}")

# =============================================================================
# EXTRACT LEAD PRODUCTION COMPANY
# =============================================================================

def get_lead_company(parsed_list):
    if len(parsed_list) > 0 and 'name' in parsed_list[0]:
        return parsed_list[0]['name']
    return None

df_multi['lead_company'] = df_multi['production_companies_parsed'].apply(get_lead_company)

# =============================================================================
# ASSIGN HOME COUNTRY
# =============================================================================

# Try high confidence first
df_multi['home_country'] = df_multi['lead_company'].map(high_conf_dict)
df_multi['mapping_confidence'] = df_multi['home_country'].apply(
    lambda x: 'high' if pd.notna(x) else None
)

# Then low confidence for remaining
unmapped_mask = df_multi['home_country'].isna()
df_multi.loc[unmapped_mask, 'home_country'] = df_multi.loc[unmapped_mask, 'lead_company'].map(low_conf_dict)
df_multi.loc[unmapped_mask & df_multi['home_country'].notna(), 'mapping_confidence'] = 'low'

# Filter to films with valid lead company and home country
df_multi = df_multi[
    (df_multi['lead_company'].notna()) &
    (df_multi['home_country'].notna())
].copy()

# =============================================================================
# BUDGET CORRECTIONS
# =============================================================================

print("\n=== BUDGET CORRECTIONS ===")

# Load corrections and drops
df_corrections = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/budget_corrections.csv")
df_drop = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_to_drop.csv")

# Apply corrections
corrections_made = 0
for _, row in df_corrections.iterrows():
    mask = (df_multi['title'] == row['title']) & (df_multi['year'] == row['year'])
    if mask.any():
        old_budget = df_multi.loc[mask, 'budget'].values[0]
        df_multi.loc[mask, 'budget'] = row['corrected_budget']
        print(f"Corrected: {row['title']} ({int(row['year'])}): ${old_budget:,.0f} -> ${row['corrected_budget']:,.0f}")
        corrections_made += 1

print(f"Budget corrections applied: {corrections_made}")

# Drop unverifiable films
films_before = len(df_multi)
for _, row in df_drop.iterrows():
    mask = (df_multi['title'] == row['title']) & (df_multi['year'] == row['year'])
    df_multi = df_multi[~mask]

print(f"Films dropped (unverifiable): {films_before - len(df_multi)}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n=== FINAL MAPPING RESULTS ===")
total = len(df_multi)
high_mapped = (df_multi['mapping_confidence'] == 'high').sum()
low_mapped = (df_multi['mapping_confidence'] == 'low').sum()

print(f"Total films: {total}")
print(f"High confidence: {high_mapped} ({high_mapped/total*100:.1f}%)")
print(f"Low confidence: {low_mapped} ({low_mapped/total*100:.1f}%)")

print("\n=== HOME COUNTRY DISTRIBUTION (Top 15) ===")
print(df_multi['home_country'].value_counts().head(15))

print("\n=== YEAR DISTRIBUTION ===")
print(df_multi['year'].value_counts().sort_index())
print(f"Year range: {df_multi['year'].min()} to {df_multi['year'].max()}")

# Check for remaining suspicious budgets
suspicious = df_multi[df_multi['budget'] < 100000]
print(f"\nFilms with budget < $100,000: {len(suspicious)}")
if len(suspicious) > 0:
    print(suspicious[['title', 'year', 'budget']])

# =============================================================================
# SAVE FINAL DATA
# =============================================================================

# All mapped films
df_multi.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_cleaned.csv",
    index=False
)

# High confidence only
df_high_conf = df_multi[df_multi['mapping_confidence'] == 'high'].copy()
df_high_conf.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_cleaned_high_confidence.csv",
    index=False
)

print(f"\n=== FILES SAVED ===")
print(f"All mapped films: {len(df_multi)}")
print(f"High confidence only: {len(df_high_conf)}")