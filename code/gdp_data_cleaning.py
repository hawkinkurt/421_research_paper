"""gdp_data_cleaning.py"""
import pandas as pd
import numpy as np
import pycountry

# Read World Bank GDP data
df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/world_bank_gdp.csv")

# Explore data
print(df.head())
print(df.shape)
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())

# Check for zero/negative GDP values
print(f"Zero or negative GDP values: {(df['gdp'] <= 0).sum()}")

# Find regional aggregates
print(df['country'].unique()[:30])

# List of regional aggregates to remove
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

# Keep rows where country is not in the aggregates list
df = df[~df['country'].isin(aggregates)]

print(f"Rows remaining: {len(df)}")
print(f"Countries remaining: {df['country'].nunique()}")

# Drop missing GDP rows
df = df.dropna(subset=['gdp'])

# Manual mappings for World Bank names that pycountry can't match
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
    # Check manual mappings first
    if country_name in manual_mappings:
        return manual_mappings[country_name]
    # Fall back to pycountry
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_2
    except:
        return None

# Apply conversion
df['iso2'] = df['country'].apply(get_iso2)

# Check results (before dropping the column)
matched = df['iso2'].notna().sum()
unmatched = df['iso2'].isna().sum()

print(f"\n=== Conversion Results ===")
print(f"Matched: {matched} observations")
print(f"Unmatched: {unmatched} observations")

# Show unmatched countries
if unmatched > 0:
    unmatched_countries = df[df['iso2'].isna()]['country'].unique()
    print(f"\nUnmatched country names ({len(unmatched_countries)}):")
    for c in unmatched_countries:
        print(f"  - {c}")

# Overwrite country names with ISO2 codes and drop the iso2 column
df['country'] = df['iso2']
df = df.drop(columns=['iso2'])
df = df.dropna(subset=['country'])

# Create log GDP variable
df['log_gdp'] = np.log(df['gdp'])

# Check the result
print(df.head())
print(f"Final dataset: {len(df)} rows, {df['country'].nunique()} countries")

# Save cleaned data
df.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gdp_cleaned.csv", index=False)

print("Cleaned GDP data saved")