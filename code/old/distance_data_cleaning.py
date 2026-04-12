"""gravity_cepii_cleaning.py"""
"""Cleaning CEPII distance/gravity data"""

import pandas as pd
import numpy as np
import pycountry

# LOAD DATA

cols_to_keep = [
    'year', 'iso3_o', 'iso3_d',
    'dist', 'distcap', 'contig',
    'comlang_off', 'comlang_ethno',
    'col45', 'col_dep_ever',
    'comrelig',
    'fta_wto', 'rta_type',
    'eu_o', 'eu_d',
    'wto_o', 'wto_d'
]

print("Loading CEPII Gravity dataset (this may take a moment)...")
df = pd.read_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/base/Gravity_V202211.csv",
    usecols=cols_to_keep
)
print(f"Loaded: {len(df):,} observations")

# FILTER TO RELEVANT YEARS (1990-2018)

df = df[(df['year'] >= 1990) & (df['year'] <= 2018)]
print(f"After filtering to 1990-2018: {len(df):,} observations")

# CONVERT ISO3 TO ISO2

# Manual mappings for old/non-standard ISO3 codes
manual_iso3_mappings = {
    'ROM': 'RO',  # Romania (old code)
    'ZAR': 'CD',  # Zaire → DR Congo
    'TMP': 'TL',  # East Timor (old code)
    'PAL': 'PS',  # Palestine
    'YUG': 'RS',  # Yugoslavia → Serbia
    'ANT': 'CW',  # Netherlands Antilles → Curaçao
    'SCG': 'RS',  # Serbia and Montenegro → Serbia
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

print("\nConverting ISO3 to ISO2 codes...")
df['iso_o'] = df['iso3_o'].apply(iso3_to_iso2)
df['iso_d'] = df['iso3_d'].apply(iso3_to_iso2)

# Check conversion results
matched = df['iso_o'].notna().sum()
total = len(df)
print(f"Origin conversion: {matched:,} / {total:,} ({matched/total*100:.1f}%)")

# Show failed conversions
failed_o = df[df['iso_o'].isna()]['iso3_o'].unique()
if len(failed_o) > 0:
    print(f"Failed origin codes ({len(failed_o)}): {failed_o[:20].tolist()}")

# Drop rows where conversion failed
df = df.dropna(subset=['iso_o', 'iso_d'])
print(f"After dropping failed conversions: {len(df):,} observations")

# CREATE DERIVED VARIABLES

# Log distance
df['log_dist'] = np.log(df['dist'])

# RTA dummy (1 if any FTA/RTA exists)
df['rta'] = (df['fta_wto'] == 1).astype(int)

# Both countries in EU
df['both_eu'] = ((df['eu_o'] == 1) & (df['eu_d'] == 1)).astype(int)

# SELECT AND RENAME FINAL COLUMNS

df_final = df[[
    'year', 'iso_o', 'iso_d',
    'dist', 'distcap', 'log_dist',
    'contig', 'comlang_off', 'comlang_ethno',
    'col45', 'col_dep_ever', 'comrelig',
    'rta', 'both_eu'
]].copy()

# DEDUPLICATE: Keep row with most complete data for each country-pair-year

print("\n=== Deduplicating ===")
print(f"Before deduplication: {len(df_final):,} observations")

# Count non-missing values per row for key variables
df_final['completeness'] = df_final[['dist', 'contig', 'comlang_off', 'rta']].notna().sum(axis=1)

# Sort by completeness (descending) and keep first occurrence
df_final = df_final.sort_values('completeness', ascending=False)
df_final = df_final.drop_duplicates(subset=['iso_o', 'iso_d', 'year'], keep='first')
df_final = df_final.drop(columns=['completeness'])

print(f"After deduplication: {len(df_final):,} observations")

# Remove self-pairs (country paired with itself)
before_count = len(df_final)
df_final = df_final[df_final['iso_o'] != df_final['iso_d']]
after_count = len(df_final)

print(f"Removed {before_count - after_count:,} self-pair observations")
print(f"Remaining: {after_count:,} observations")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n=== Final Dataset Summary ===")
print(f"Observations: {len(df_final):,}")
print(f"Unique origin countries: {df_final['iso_o'].nunique()}")
print(f"Unique destination countries: {df_final['iso_d'].nunique()}")
print(f"Year range: {df_final['year'].min()} - {df_final['year'].max()}")

print("\n=== Variable Coverage ===")
for col in ['dist', 'contig', 'comlang_off', 'rta', 'comrelig']:
    non_missing = df_final[col].notna().sum()
    print(f"{col}: {non_missing:,} non-missing ({non_missing/len(df_final)*100:.1f}%)")

print("\n=== RTA Distribution ===")
print(df_final['rta'].value_counts())

# =============================================================================
# SAVE
# =============================================================================

df_final.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_vars_cepii.csv",
    index=False
)
print("\n=== Saved to gravity_vars_cepii.csv ===")