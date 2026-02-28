"""
imdb_data_fetch_and_clean.py
============================
Script 1 of 2 for the Filming Location Gravity Model.

Downloads the IMDb dataset via Kaggle API, combines yearly files (1990-2023),
extracts filming countries from location strings, harmonises country names,
and builds bilateral trade flows (origin country → filming country).

Output:
  - data/processed/imdb/imdb_filming_trade_flows.csv  (bilateral pair-year aggregated)
  - data/processed/imdb/imdb_filming_flows_by_film.csv (film-level flows)
  - data/processed/imdb/imdb_combined_cleaned.csv      (cleaned combined dataset)
"""

import pandas as pd
import numpy as np
import ast
import re
import os
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'
OUTPUT_DIR = f'{PROJECT_DIR}\\data\\processed\\imdb'
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_MIN = 1990
YEAR_MAX = 2023  # Constrained by World Bank GDP availability

# =============================================================================
# PHASE 1: DOWNLOAD / LOCATE DATASET
# =============================================================================

print("=" * 70)
print("PHASE 1: LOCATE IMDb DATASET")
print("=" * 70)

import kagglehub
data_path = kagglehub.dataset_download("raedaddala/imdb-movies-from-1960-to-2023")
print(f"Dataset path: {data_path}")

# Find the CSV files — this dataset uses year subfolders: Data/1990/, Data/1991/, etc.
print(f"Searching for CSV files under: {data_path}")

# Build a mapping: year -> filepath
year_file_map = {}
for root, dirs, files in os.walk(data_path):
    for f in files:
        if f.startswith('merged_movies_data_') and f.endswith('.csv'):
            # Extract year from filename
            try:
                year_str = f.replace('merged_movies_data_', '').replace('.csv', '')
                year_int = int(year_str)
                year_file_map[year_int] = os.path.join(root, f)
            except ValueError:
                pass

if not year_file_map:
    # Show what IS in the path to help debug
    print(f"\nCould not find yearly CSVs. Contents of {data_path}:")
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more")
    raise FileNotFoundError("Cannot locate merged_movies_data_YYYY.csv files in the downloaded dataset")

print(f"Found {len(year_file_map)} yearly CSV files (years {min(year_file_map)}-{max(year_file_map)})")
print(f"Sample path: {year_file_map[min(year_file_map)]}")

# =============================================================================
# PHASE 2: COMBINE YEARLY FILES
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: COMBINE YEARLY FILES (1990-2023)")
print("=" * 70)

frames = []
year_stats = []

for year in range(YEAR_MIN, YEAR_MAX + 1):
    if year not in year_file_map:
        print(f"  WARNING: year {year} not found in dataset — skipping")
        continue

    filepath = year_file_map[year]
    df_year = pd.read_csv(filepath)
    n_total = len(df_year)
    n_filming = df_year['filming_locations'].notna().sum()
    n_budget = df_year['budget'].notna().sum()

    year_stats.append({
        'year': year,
        'total_films': n_total,
        'has_filming_loc': n_filming,
        'has_budget': n_budget,
        'filming_pct': 100 * n_filming / n_total if n_total > 0 else 0,
        'budget_pct': 100 * n_budget / n_total if n_total > 0 else 0
    })

    frames.append(df_year)
    print(f"  {year}: {n_total} films, {n_filming} with filming loc ({100*n_filming/n_total:.0f}%), "
          f"{n_budget} with budget ({100*n_budget/n_total:.0f}%)")

df = pd.concat(frames, ignore_index=True)
print(f"\nCombined dataset: {len(df):,} films across {len(frames)} years")

# Year coverage summary
stats_df = pd.DataFrame(year_stats)
print(f"\nCoverage summary:")
print(f"  Filming location: {stats_df['filming_pct'].mean():.1f}% average across years")
print(f"  Budget: {stats_df['budget_pct'].mean():.1f}% average across years")
print(f"  Filming loc range: {stats_df['filming_pct'].min():.0f}% - {stats_df['filming_pct'].max():.0f}%")

# =============================================================================
# PHASE 3: PARSE LIST FIELDS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: PARSE LIST FIELDS")
print("=" * 70)


def safe_parse_list(val):
    """Parse string representation of a Python list."""
    if pd.isna(val):
        return []
    try:
        result = ast.literal_eval(val)
        if isinstance(result, list):
            return result
        return [result]
    except (ValueError, SyntaxError):
        return []


df['origin_countries'] = df['countries_origin'].apply(safe_parse_list)
df['languages_list'] = df['Languages'].apply(safe_parse_list)
df['n_origin_countries'] = df['origin_countries'].apply(len)

print(f"Films with valid countries_origin: {(df['n_origin_countries'] > 0).sum():,}")
print(f"Films with no countries_origin: {(df['n_origin_countries'] == 0).sum():,}")

# =============================================================================
# PHASE 4: EXTRACT FILMING COUNTRY
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4: EXTRACT FILMING COUNTRY FROM LOCATION STRINGS")
print("=" * 70)

# --- Filming location country name → standardised name ---
# These map the final comma-separated element in the IMDb filming location string
# --- Load filming location mappings from CSV ---
# Maps raw strings (last comma element in IMDb filming_locations) to standardised country names
# Empty standardised_name means "drop this entry"
filming_map_file = f'{PROJECT_DIR}\\data\\raw\\base\\imdb_filming_location_mappings.csv'
filming_map_df = pd.read_csv(filming_map_file, keep_default_na=False)
FILMING_LOC_TO_STANDARD = {}
for _, row in filming_map_df.iterrows():
    raw = row['raw_string']
    std = row['standardised_name'] if row['standardised_name'] != '' else None
    FILMING_LOC_TO_STANDARD[raw] = std
print(f"Loaded {len(FILMING_LOC_TO_STANDARD)} filming location mappings from CSV")


def extract_filming_country(loc_str):
    """Extract the country from an IMDb filming_locations string.

    The field contains a Python list with typically one address string.
    The country is the last comma-separated element after removing
    parenthetical notes.
    """
    if pd.isna(loc_str):
        return None
    try:
        locs = ast.literal_eval(loc_str)
    except (ValueError, SyntaxError):
        return None

    for loc in locs:
        # Remove parenthetical descriptions e.g. "(McCallister home)"
        loc_clean = re.sub(r'\(.*?\)', '', loc).strip()
        parts = [p.strip() for p in loc_clean.split(',')]
        if not parts:
            continue
        raw_country = parts[-1]

        # Apply mapping
        if raw_country in FILMING_LOC_TO_STANDARD:
            return FILMING_LOC_TO_STANDARD[raw_country]
        else:
            return raw_country

    return None


df['filming_country'] = df['filming_locations'].apply(extract_filming_country)

n_with_filming = df['filming_country'].notna().sum()
print(f"Films with extracted filming country: {n_with_filming:,} / {len(df):,} ({100*n_with_filming/len(df):.1f}%)")

# Show distribution
filming_counts = df['filming_country'].value_counts().head(25)
print(f"\nTop 25 filming countries:")
for country, count in filming_counts.items():
    print(f"  {country}: {count}")

# Flag films where filming country is NOT in the origin countries
# These are the key "attracted production" cases
df['filming_abroad'] = df.apply(
    lambda row: (row['filming_country'] is not None and
                 row['filming_country'] not in row['origin_countries']),
    axis=1
)
n_abroad = df['filming_abroad'].sum()
n_with_both = df[(df['filming_country'].notna()) & (df['n_origin_countries'] > 0)].shape[0]
print(f"\nFilms where filming country ≠ any origin country: {n_abroad:,} / {n_with_both:,} ({100*n_abroad/n_with_both:.1f}%)")

# =============================================================================
# PHASE 5: COUNTRY NAME HARMONISATION TO ISO2
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 5: HARMONISE COUNTRY NAMES TO ISO2 CODES")
print("=" * 70)

# --- Load country name to ISO2 mappings from CSV ---
# Covers both countries_origin names and standardised filming location names
iso_map_file = f'{PROJECT_DIR}\\data\\raw\\base\\imdb_country_name_to_iso2.csv'
iso_map_df = pd.read_csv(iso_map_file)
COUNTRY_TO_ISO2 = dict(zip(iso_map_df['country_name'], iso_map_df['iso2']))
print(f"Loaded {len(COUNTRY_TO_ISO2)} country-to-ISO2 mappings from CSV")


def country_name_to_iso2(name):
    """Convert a country name to ISO2 code."""
    if pd.isna(name) or name is None:
        return None
    name = name.strip()
    return COUNTRY_TO_ISO2.get(name, None)


# Convert origin countries to ISO2
def origins_to_iso2(country_list):
    """Convert list of origin country names to list of ISO2 codes."""
    codes = []
    for name in country_list:
        code = country_name_to_iso2(name)
        if code:
            codes.append(code)
    return codes


df['origin_iso2'] = df['origin_countries'].apply(origins_to_iso2)
df['filming_iso2'] = df['filming_country'].apply(country_name_to_iso2)

# Diagnostic: what didn't map?
all_origin_names = set()
for lst in df['origin_countries']:
    all_origin_names.update(lst)

unmapped_origins = {n for n in all_origin_names if country_name_to_iso2(n) is None}
unmapped_filming = set(df[df['filming_country'].notna() & df['filming_iso2'].isna()]['filming_country'].unique())

print(f"Origin countries: {len(all_origin_names)} unique names, {len(unmapped_origins)} unmapped")
if unmapped_origins:
    print(f"  Unmapped origins: {sorted(unmapped_origins)}")

print(f"Filming countries: {df['filming_country'].nunique()} unique names, {len(unmapped_filming)} unmapped")
if unmapped_filming:
    print(f"  Unmapped filming: {sorted(unmapped_filming)}")

# =============================================================================
# PHASE 6: PARSE BUDGET (USD only)
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 6: PARSE BUDGET")
print("=" * 70)


def parse_budget_usd(budget_str):
    """Parse budget string, returning value in USD or None if non-USD."""
    if pd.isna(budget_str):
        return None
    s = str(budget_str).strip()
    # Only keep USD budgets (starting with $)
    if not s.startswith('$'):
        return None
    # Remove $, commas, and any text like "(estimated)"
    cleaned = re.sub(r'[^\d.]', '', s.split('(')[0])
    try:
        return float(cleaned)
    except ValueError:
        return None


df['budget_usd'] = df['budget'].apply(parse_budget_usd)

n_with_budget = df['budget_usd'].notna().sum()
print(f"Films with USD budget: {n_with_budget:,} / {len(df):,} ({100*n_with_budget/len(df):.1f}%)")
print(f"Budget range: ${df['budget_usd'].min():,.0f} - ${df['budget_usd'].max():,.0f}")
print(f"Median budget: ${df['budget_usd'].median():,.0f}")

# =============================================================================
# PHASE 7: BUILD BILATERAL TRADE FLOWS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 7: BUILD BILATERAL TRADE FLOWS")
print("=" * 70)

# Logic:
#   For each film with a valid filming_iso2:
#     For each country in origin_iso2:
#       If origin ≠ filming country → create a flow
#       flow count = 1
#       flow budget = full budget_usd (NOT split by number of origins)

trade_flows = []
skipped_domestic = 0
skipped_no_filming = 0
skipped_no_origin = 0

for idx, row in df.iterrows():
    filming = row['filming_iso2']
    origins = row['origin_iso2']
    year = row['Year']
    budget = row['budget_usd']  # May be None

    if filming is None:
        skipped_no_filming += 1
        continue
    if not origins:
        skipped_no_origin += 1
        continue

    for origin in origins:
        if origin == filming:
            skipped_domestic += 1
            continue

        trade_flows.append({
            'year': int(year),
            'importer': origin,       # Country that financed the film
            'exporter': filming,      # Country where filming took place
            'film_count': 1,
            'budget_usd': budget,     # Full budget assigned to each flow
            'title': row.get('Title', '')
        })

df_flows = pd.DataFrame(trade_flows)

print(f"Film-level bilateral flows: {len(df_flows):,}")
print(f"Skipped - no filming location: {skipped_no_filming:,}")
print(f"Skipped - no valid origin: {skipped_no_origin:,}")
print(f"Skipped - domestic (origin=filming): {skipped_domestic:,}")

if len(df_flows) == 0:
    print("ERROR: No trade flows generated. Check data.")
    exit(1)

print(f"\nFlows with budget data: {df_flows['budget_usd'].notna().sum():,} "
      f"({100*df_flows['budget_usd'].notna().sum()/len(df_flows):.1f}%)")
print(f"Unique country pairs: {df_flows.groupby(['importer', 'exporter']).ngroups}")
print(f"Unique importers: {df_flows['importer'].nunique()}")
print(f"Unique exporters: {df_flows['exporter'].nunique()}")

# =============================================================================
# PHASE 8: AGGREGATE TO BILATERAL PAIR-YEAR
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 8: AGGREGATE TO PAIR-YEAR LEVEL")
print("=" * 70)

df_bilateral = df_flows.groupby(['year', 'importer', 'exporter']).agg(
    num_films=('film_count', 'sum'),
    total_budget=('budget_usd', 'sum'),        # Sum of budgets (NaN-safe)
    films_with_budget=('budget_usd', 'count'),  # Count of non-null budgets
).reset_index()

# Create log variables
df_bilateral['log_num_films'] = np.log(df_bilateral['num_films'])
df_bilateral['log_budget'] = np.where(
    df_bilateral['total_budget'] > 0,
    np.log(df_bilateral['total_budget']),
    np.nan
)

print(f"Bilateral pair-year observations: {len(df_bilateral):,}")
print(f"Year range: {df_bilateral['year'].min()} - {df_bilateral['year'].max()}")
print(f"Unique importers: {df_bilateral['importer'].nunique()}")
print(f"Unique exporters: {df_bilateral['exporter'].nunique()}")
print(f"Observations with budget data: {df_bilateral['log_budget'].notna().sum():,} "
      f"({100*df_bilateral['log_budget'].notna().sum()/len(df_bilateral):.1f}%)")

print(f"\nFilm count distribution:")
print(df_bilateral['num_films'].describe())

print(f"\nTop 15 bilateral corridors (by total films):")
top_corridors = df_bilateral.groupby(['importer', 'exporter'])['num_films'].sum().sort_values(ascending=False).head(15)
for (imp, exp), count in top_corridors.items():
    print(f"  {imp} → {exp}: {count} films")

# =============================================================================
# PHASE 9: SAVE OUTPUT
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 9: SAVE OUTPUT")
print("=" * 70)

# Save bilateral aggregated flows
bilateral_file = f'{OUTPUT_DIR}\\imdb_filming_trade_flows.csv'
df_bilateral.to_csv(bilateral_file, index=False)
print(f"Saved bilateral flows: {bilateral_file}")

# Save film-level flows
film_flows_file = f'{OUTPUT_DIR}\\imdb_filming_flows_by_film.csv'
df_flows.to_csv(film_flows_file, index=False)
print(f"Saved film-level flows: {film_flows_file}")

# Save cleaned combined dataset (for reference)
# Keep only the columns we need to keep file size manageable
keep_cols = ['Title', 'Year', 'countries_origin', 'filming_locations',
             'filming_country', 'filming_iso2', 'origin_iso2', 'n_origin_countries',
             'budget', 'budget_usd', 'production_company', 'Languages', 'genres']
combined_file = f'{OUTPUT_DIR}\\imdb_combined_cleaned.csv'
df[[c for c in keep_cols if c in df.columns]].to_csv(combined_file, index=False)
print(f"Saved cleaned combined data: {combined_file}")

# Save year-level coverage statistics
stats_file = f'{OUTPUT_DIR}\\imdb_year_coverage_stats.csv'
stats_df.to_csv(stats_file, index=False)
print(f"Saved coverage stats: {stats_file}")

print("\n" + "=" * 70)
print("DATA FETCH AND CLEAN COMPLETE")
print("=" * 70)
print(f"  Total films processed: {len(df):,}")
print(f"  Bilateral pair-year observations: {len(df_bilateral):,}")
print(f"  Film-level flows: {len(df_flows):,}")
print(f"  Observations with budget: {df_bilateral['log_budget'].notna().sum():,}")