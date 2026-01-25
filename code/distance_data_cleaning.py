"""distance_data_cleaning.py"""
import pandas as pd
import numpy as np
import pycountry

# Load the CEPII distance data
df = pd.read_excel("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/dist_cepii.xls")

# Replace '.' with NaN so pandas treats it as missing
df['distw'] = df['distw'].replace('.', np.nan)
df['distwces'] = df['distwces'].replace('.', np.nan)

# Convert to numeric
df['distw'] = pd.to_numeric(df['distw'])
df['distwces'] = pd.to_numeric(df['distwces'])

# Check how many rows have missing weighted distance
print(f"Missing distw values: {df['distw'].isnull().sum()}")

# =============================================================================
# CONVERT ISO3 TO ISO2
# =============================================================================

# Manual mappings for old/non-standard ISO3 codes
manual_iso3_mappings = {
    'ROM': 'RO',  # Romania (old code)
    'ZAR': 'CD',  # Zaire → DR Congo
    'TMP': 'TL',  # East Timor (old code)
    'PAL': 'PS',  # Palestine
    'YUG': 'RS',  # Yugoslavia → Serbia (largest successor)
    'ANT': 'CW',  # Netherlands Antilles → Curaçao (largest successor)
    'SCG': 'RS',  # Serbia and Montenegro → Serbia
}

def iso3_to_iso2(iso3_code):
    # Check manual mappings first
    if iso3_code in manual_iso3_mappings:
        return manual_iso3_mappings[iso3_code]
    # Fall back to pycountry
    try:
        country = pycountry.countries.get(alpha_3=iso3_code)
        return country.alpha_2
    except:
        return None

# Apply conversion to both origin and destination
df['iso_o_2'] = df['iso_o'].apply(iso3_to_iso2)
df['iso_d_2'] = df['iso_d'].apply(iso3_to_iso2)

# Check results
matched_o = df['iso_o_2'].notna().sum()
unmatched_o = df['iso_o_2'].isna().sum()

print(f"\n=== ISO Conversion Results ===")
print(f"Origin matched: {matched_o}, unmatched: {unmatched_o}")

# Show unmatched codes
if unmatched_o > 0:
    unmatched_codes = df[df['iso_o_2'].isna()]['iso_o'].unique()
    print(f"Unmatched origin codes ({len(unmatched_codes)}): {unmatched_codes[:20].tolist()}")

# Overwrite original columns with ISO2 codes
df['iso_o'] = df['iso_o_2']
df['iso_d'] = df['iso_d_2']
df = df.drop(columns=['iso_o_2', 'iso_d_2'])

# Drop rows where either code couldn't be converted
df = df.dropna(subset=['iso_o', 'iso_d'])

# =============================================================================
# CREATE LOG DISTANCE
# =============================================================================

df['log_dist'] = np.log(df['dist'])

# Check the result
print(f"\n=== Final Dataset ===")
print(f"Observations: {len(df)}")
print(f"Sample iso_o codes: {df['iso_o'].unique()[:10].tolist()}")
print(df.head())
print(df.dtypes)

# Save cleaned data
df.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/distance_cleaned.csv", index=False)

print("\nCleaned data saved")