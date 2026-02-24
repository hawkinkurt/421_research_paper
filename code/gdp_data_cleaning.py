"""gdp_data_cleaning.py"""
"""Cleaning GDP data"""

import pandas as pd
import numpy as np
import pycountry

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/world_bank_gdp.csv")

print("=== Data Loaded ===")
print(f"Observations: {len(df)}")
print(f"Missing GDP values: {df['gdp'].isnull().sum()}")
print(f"Zero or negative GDP values: {(df['gdp'] <= 0).sum()}")

# =============================================================================
# REMOVE REGIONAL AGGREGATES
# =============================================================================

aggregates = [
    'Africa Eastern and Southern', 'Africa Western and Central', 'Arab World',
    'Caribbean small states', 'Central Europe and the Baltics',
    'Early-demographic dividend', 'East Asia & Pacific',
    'East Asia & Pacific (excluding high income)',
    'East Asia & Pacific (IDA & IBRD countries)', 'Euro area',
    'Europe & Central Asia', 'Europe & Central Asia (excluding high income)',
    'Europe & Central Asia (IDA & IBRD countries)', 'European Union',
    'Fragile and conflict affected situations',
    'Heavily indebted poor countries (HIPC)', 'High income', 'IBRD only',
    'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total',
    'Late-demographic dividend', 'Latin America & Caribbean',
    'Latin America & Caribbean (excluding high income)',
    'Latin America & the Caribbean (IDA & IBRD countries)',
    'Least developed countries: UN classification', 'Low & middle income',
    'Low income', 'Lower middle income', 'Middle East & North Africa',
    'Middle East & North Africa (excluding high income)',
    'Middle East & North Africa (IDA & IBRD countries)', 'Middle income',
    'North America', 'OECD members', 'Other small states',
    'Pacific island small states', 'Post-demographic dividend',
    'Pre-demographic dividend', 'Small states', 'South Asia',
    'South Asia (IDA & IBRD)', 'Sub-Saharan Africa',
    'Sub-Saharan Africa (excluding high income)',
    'Sub-Saharan Africa (IDA & IBRD countries)', 'Upper middle income',
    'World', 'Middle East, North Africa, Afghanistan & Pakistan',
    'Middle East, North Africa, Afghanistan & Pakistan (excluding high income)',
    'Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)'
]

df = df[~df['country'].isin(aggregates)]
print(f"\nAfter removing aggregates: {len(df)} rows, {df['country'].nunique()} countries")

# =============================================================================
# DROP MISSING GDP
# =============================================================================

df = df.dropna(subset=['gdp'])
print(f"After dropping missing GDP: {len(df)} rows")

# =============================================================================
# CONVERT COUNTRY NAMES TO ISO2
# =============================================================================

manual_mappings = {
    'Korea, Rep.': 'KR',
    "Korea, Dem. People's Rep.": 'KP',
    'Hong Kong SAR, China': 'HK',
    'Iran, Islamic Rep.': 'IR',
    'Venezuela, RB': 'VE',
    'Bahamas, The': 'BS',
    'Lao PDR': 'LA',
    'Egypt, Arab Rep.': 'EG',
    'Yemen, Rep.': 'YE',
    'Syria, Arab Republic': 'SY',
    'Turkiye': 'TR',
    'Slovak Republic': 'SK',
    'Czech Republic': 'CZ',
    'Russia': 'RU',
    'Russian Federation': 'RU',
    'Congo, Dem. Rep.': 'CD',
    'Congo, Rep.': 'CG',
    'Gambia, The': 'GM',
    'Kyrgyz Republic': 'KG',
    'Micronesia, Fed. Sts.': 'FM',
    'St. Lucia': 'LC',
    'St. Vincent and the Grenadines': 'VC',
    'St. Kitts and Nevis': 'KN',
}

def get_iso2(country_name):
    if country_name in manual_mappings:
        return manual_mappings[country_name]
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_2
    except:
        return None

df['iso2'] = df['country'].apply(get_iso2)

matched = df['iso2'].notna().sum()
unmatched = df['iso2'].isna().sum()

print(f"\n=== ISO Conversion Results ===")
print(f"Matched: {matched} observations")
print(f"Unmatched: {unmatched} observations")

if unmatched > 0:
    unmatched_countries = df[df['iso2'].isna()]['country'].unique()
    print(f"\nUnmatched country names ({len(unmatched_countries)}):")
    for c in unmatched_countries:
        print(f"  - {c}")

df['country'] = df['iso2']
df = df.drop(columns=['iso2'])
df = df.dropna(subset=['country'])

# =============================================================================
# DEDUPLICATE
# =============================================================================

before_dedup = len(df)
df = df.drop_duplicates(subset=['country', 'date'], keep='first')
after_dedup = len(df)

if before_dedup > after_dedup:
    print(f"\n=== Deduplication ===")
    print(f"Dropped {before_dedup - after_dedup} duplicate country-year observations")

# =============================================================================
# CREATE LOG GDP
# =============================================================================

df['log_gdp'] = np.log(df['gdp'])

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n=== Final Dataset ===")
print(f"Observations: {len(df)}")
print(f"Countries: {df['country'].nunique()}")
print(f"Year range: {df['date'].min()} - {df['date'].max()}")

# =============================================================================
# SAVE
# =============================================================================

df.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gdp_cleaned.csv", index=False)
print("\n=== Saved to gdp_cleaned.csv ===")