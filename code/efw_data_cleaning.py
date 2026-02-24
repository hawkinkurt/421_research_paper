"""efw_data_cleaning.py"""
"""Cleaning Economic Freedom of the World data"""

import pandas as pd
import pycountry

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_excel(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/base/efotw-2025-master-index-data-for-researchers-iso.xlsx",
    header=3
)

print("=== Data Loaded ===")
print(f"Observations: {len(df)}")
print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")

# =============================================================================
# SELECT RELEVANT COLUMNS
# =============================================================================

df = df[['Year', 'ISO_Code', 'Countries', 'ECONOMIC FREEDOM ALL AREAS']].copy()
df.columns = ['year', 'iso3', 'country_name', 'efw']

# =============================================================================
# FILTER TO RELEVANT YEARS (1990-2018)
# =============================================================================

df = df[(df['year'] >= 1990) & (df['year'] <= 2018)]
print(f"After filtering to 1990-2018: {len(df)} observations")

# =============================================================================
# CONVERT ISO3 TO ISO2
# =============================================================================

# Manual mappings for non-standard codes
manual_iso3_mappings = {
    'ROM': 'RO',
    'ZAR': 'CD',
    'TMP': 'TL',
    'PAL': 'PS',
    'YUG': 'RS',
    'ANT': 'CW',
    'SCG': 'RS',
}

def iso3_to_iso2(iso3_code):
    if pd.isna(iso3_code):
        return None
    if iso3_code in manual_iso3_mappings:
        return manual_iso3_mappings[iso3_code]
    try:
        country = pycountry.countries.get(alpha_3=iso3_code)
        if country:
            return country.alpha_2
        return None
    except:
        return None

df['iso2'] = df['iso3'].apply(iso3_to_iso2)

# Check conversion results
matched = df['iso2'].notna().sum()
unmatched = df['iso2'].isna().sum()

print(f"\n=== ISO Conversion Results ===")
print(f"Matched: {matched} observations")
print(f"Unmatched: {unmatched} observations")

if unmatched > 0:
    failed_codes = df[df['iso2'].isna()][['iso3', 'country_name']].drop_duplicates()
    print(f"Failed codes:")
    print(failed_codes)

# Drop rows where conversion failed
df = df.dropna(subset=['iso2'])

# =============================================================================
# CHECK FOR MISSING EFW VALUES
# =============================================================================

print(f"\n=== EFW Coverage ===")
print(f"Total observations: {len(df)}")
print(f"Missing EFW: {df['efw'].isna().sum()}")

# Drop missing EFW
df = df.dropna(subset=['efw'])

# =============================================================================
# CHECK YEAR COVERAGE
# =============================================================================

print(f"\n=== Year Coverage ===")
year_counts = df.groupby('year').size()
print(year_counts)

# =============================================================================
# INTERPOLATE MISSING YEARS (1991-1994, 1996-1999)
# =============================================================================

print("\n=== Interpolating missing years ===")

df_final = df[['year', 'iso2', 'efw']].copy()
df_final.columns = ['year', 'country', 'efw']

# Get all unique countries
countries = df_final['country'].unique()

# Create full year range
all_years = list(range(1990, 2019))

# Create complete country-year grid
full_grid = pd.DataFrame([
    {'country': c, 'year': y}
    for c in countries
    for y in all_years
])

# Merge existing data onto full grid
df_full = full_grid.merge(df_final, on=['country', 'year'], how='left')

# Interpolate within each country
df_full = df_full.sort_values(['country', 'year'])
df_full['efw'] = df_full.groupby('country')['efw'].transform(
    lambda x: x.interpolate(method='linear')
)

# Check interpolation results
before_interp = df_final['efw'].notna().sum()
after_interp = df_full['efw'].notna().sum()
print(f"Observations with EFW before interpolation: {before_interp}")
print(f"Observations with EFW after interpolation: {after_interp}")
print(f"Interpolated values added: {after_interp - before_interp}")

# Check year coverage after interpolation
print("\n=== Year Coverage After Interpolation ===")
year_counts = df_full[df_full['efw'].notna()].groupby('year').size()
print(year_counts)

df_final = df_full

print(f"\n=== Final Dataset ===")
print(f"Observations: {len(df_final)}")
print(f"Unique countries: {df_final['country'].nunique()}")
print(f"Year range: {df_final['year'].min()} - {df_final['year'].max()}")

print(f"\n=== EFW Summary Statistics ===")
print(df_final['efw'].describe())

df_final.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/efw_cleaned.csv",
    index=False
)
print("\n=== Saved to efw_cleaned.csv ===")


# Check which countries have missing values and when
missing = df_final[df_final['efw'].isna()]
print(missing.groupby('country')['year'].agg(['min', 'max', 'count']))