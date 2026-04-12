"""
data_fetch_clean.py
===================
Unified data fetching, cleaning, and preparation pipeline for the
Filming Location Gravity Model.

Consolidates all data acquisition and cleaning into a single script,
organised in sequential phases:

  PHASE 1:  Fetch World Bank GDP and CPI data via API
  PHASE 2:  Fetch and process IMDb filming location data via Kaggle
  PHASE 3:  Pre-filter BaTIS SK1/SK2 audiovisual services trade data
  PHASE 4:  Clean CEPII gravity/distance variables
  PHASE 5:  Clean GDP data (country names → ISO2, log transform)
  PHASE 6:  Clean Economic Freedom of the World (EFW) data
  PHASE 7:  Prepare BaTIS SK1 gravity analysis dataset (merge all variables)

Each phase can be toggled on/off via the RUN_PHASE flags below.

Outputs:
  Phase 1: data/raw/world_bank_gdp.csv, data/raw/world_bank_cpi.csv
  Phase 2: data/processed/imdb/*.csv (stages 1-5 + final bilateral flows)
  Phase 3: data/raw/base/batis_sk1_all_years.csv, batis_sk2_all_years.csv
  Phase 4: data/processed/gravity_vars_cepii.csv
  Phase 5: data/processed/gdp_cleaned.csv
  Phase 6: data/processed/efw_cleaned.csv
  Phase 7: data/processed/batis_sk1_gravity_analysis.csv,
           data/processed/batis_sk1_gravity_merged.csv
"""

import os
import re
import ast
import numpy as np
import pandas as pd
import pycountry
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = r'C:\Users\kurtl\PycharmProjects\gravity_model'

# Toggle individual phases (set False to skip)
RUN_PHASE_1_FETCH_WB = True       # Fetch World Bank GDP + CPI
RUN_PHASE_2_IMDB = True           # Fetch & clean IMDb data
RUN_PHASE_3_BATIS_PREFILTER = True  # Pre-filter BaTIS bulk CSV
RUN_PHASE_4_CEPII = True          # Clean CEPII distance/gravity data
RUN_PHASE_5_GDP = True            # Clean GDP data
RUN_PHASE_6_EFW = True            # Clean EFW data
RUN_PHASE_7_BATIS_PREP = True     # Merge BaTIS with all variables

# Year ranges
YEAR_MIN = 1990
YEAR_MAX_IMDB = 2023
YEAR_MAX_BATIS = 2023
CPI_BASE_YEAR_IMDB = 2010
CPI_BASE_YEAR_BATIS = 2018


# =============================================================================
# SHARED UTILITIES
# =============================================================================

# Manual ISO3 → ISO2 mappings for old/non-standard codes (used by CEPII, EFW)
MANUAL_ISO3_MAPPINGS = {
    'ROM': 'RO',   # Romania (old code)
    'ZAR': 'CD',   # Zaire → DR Congo
    'TMP': 'TL',   # East Timor (old code)
    'PAL': 'PS',   # Palestine
    'YUG': 'RS',   # Yugoslavia → Serbia
    'ANT': 'CW',   # Netherlands Antilles → Curaçao
    'SCG': 'RS',   # Serbia and Montenegro → Serbia
}

# Manual country name → ISO2 mappings (used by GDP/World Bank)
MANUAL_NAME_MAPPINGS = {
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

# World Bank regional aggregates to exclude
WB_AGGREGATES = [
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
    'Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)',
]


def iso3_to_iso2(iso3_code):
    """Convert ISO3 country code to ISO2, with manual overrides."""
    if pd.isna(iso3_code):
        return None
    if iso3_code in MANUAL_ISO3_MAPPINGS:
        return MANUAL_ISO3_MAPPINGS[iso3_code]
    try:
        country = pycountry.countries.get(alpha_3=iso3_code)
        if country:
            return country.alpha_2
        return None
    except:
        return None


def country_name_to_iso2_wb(country_name):
    """Convert World Bank country name to ISO2, with manual overrides."""
    if country_name in MANUAL_NAME_MAPPINGS:
        return MANUAL_NAME_MAPPINGS[country_name]
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_2
    except:
        return None


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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1: FETCH WORLD BANK GDP AND CPI DATA                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def phase_1_fetch_world_bank():
    """Download World Bank GDP and CPI data via wbdata API."""
    import wbdata
    from datetime import datetime

    print("\n" + "=" * 70)
    print("PHASE 1: FETCH WORLD BANK GDP AND CPI DATA")
    print("=" * 70)

    # --- GDP ---
    gdp_filepath = f"{PROJECT_DIR}/data/raw/world_bank_gdp.csv"
    if os.path.exists(gdp_filepath):
        print("GDP data file already exists, skipping download")
    else:
        print("Downloading GDP data...")
        start_date = datetime(1990, 1, 1)
        end_date = datetime(2023, 12, 31)
        indicators = {"NY.GDP.MKTP.CD": "gdp"}
        df = wbdata.get_dataframe(indicators, date=(start_date, end_date))
        df = df.reset_index()
        df.to_csv(gdp_filepath, index=False)
        print(f"GDP data saved to {gdp_filepath}")

    # --- CPI ---
    cpi_filepath = f"{PROJECT_DIR}/data/raw/world_bank_cpi.csv"
    if os.path.exists(cpi_filepath):
        print("CPI data file already exists, skipping download")
    else:
        print("Downloading CPI data...")
        start_date = datetime(1990, 1, 1)
        end_date = datetime(2023, 12, 31)
        indicators = {"FP.CPI.TOTL": "cpi"}
        countries = ["USA"]
        df_cpi = wbdata.get_dataframe(indicators, country=countries, date=(start_date, end_date))
        df_cpi = df_cpi.reset_index()
        df_cpi.to_csv(cpi_filepath, index=False)
        print(f"CPI data saved to {cpi_filepath}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2: FETCH AND CLEAN IMDb FILMING LOCATION DATA                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def phase_2_imdb():
    """
    Download IMDb dataset via Kaggle, combine yearly files (1990-2023),
    extract filming countries, harmonise to ISO2, parse budgets, deflate,
    and build bilateral trade flows.
    """

    print("\n" + "=" * 70)
    print("PHASE 2: FETCH AND CLEAN IMDb FILMING LOCATION DATA")
    print("=" * 70)

    OUTPUT_DIR = f'{PROJECT_DIR}\\data\\processed\\imdb'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2a: Locate / download dataset ---
    print("\n--- 2a: Locate IMDb dataset ---")

    data_path = r'C:\Users\kurtl\.cache\kagglehub\datasets\raedaddala\imdb-movies-from-1960-to-2023\versions\5'
    if os.path.exists(data_path):
        print("IMDb dataset already downloaded, skipping download")
    else:
        print("Downloading dataset...")
        import kagglehub
        data_path = kagglehub.dataset_download("raedaddala/imdb-movies-from-1960-to-2023")
        print(f"Downloaded to: {data_path}")

    # Build year → filepath mapping
    year_file_map = {}
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.startswith('merged_movies_data_') and f.endswith('.csv'):
                try:
                    year_str = f.replace('merged_movies_data_', '').replace('.csv', '')
                    year_int = int(year_str)
                    year_file_map[year_int] = os.path.join(root, f)
                except ValueError:
                    pass

    if not year_file_map:
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
        raise FileNotFoundError("Cannot locate merged_movies_data_YYYY.csv files")

    print(f"Found {len(year_file_map)} yearly CSV files (years {min(year_file_map)}-{max(year_file_map)})")

    # --- 2b: Combine yearly files ---
    print("\n--- 2b: Combine yearly files ---")

    frames = []
    year_stats = []

    for year in range(YEAR_MIN, YEAR_MAX_IMDB + 1):
        if year not in year_file_map:
            print(f"  WARNING: year {year} not found — skipping")
            continue

        df_year = pd.read_csv(year_file_map[year])
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
        print(f"  {year}: {n_total} films, {n_filming} filming loc ({100*n_filming/n_total:.0f}%), "
              f"{n_budget} budget ({100*n_budget/n_total:.0f}%)")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nCombined dataset: {len(df):,} films across {len(frames)} years")

    stats_df = pd.DataFrame(year_stats)
    print(f"Filming location coverage: {stats_df['filming_pct'].mean():.1f}% average")
    print(f"Budget coverage: {stats_df['budget_pct'].mean():.1f}% average")

    # Save stage 1
    df.to_csv(f'{OUTPUT_DIR}\\stage1_combined_raw.csv', index=False)
    stats_df.to_csv(f'{OUTPUT_DIR}\\imdb_year_coverage_stats.csv', index=False)

    # --- 2c: Parse list fields ---
    print("\n--- 2c: Parse list fields ---")

    df['origin_countries'] = df['countries_origin'].apply(safe_parse_list)
    df['languages_list'] = df['Languages'].apply(safe_parse_list)
    df['n_origin_countries'] = df['origin_countries'].apply(len)

    print(f"Films with valid countries_origin: {(df['n_origin_countries'] > 0).sum():,}")
    print(f"Films with no countries_origin: {(df['n_origin_countries'] == 0).sum():,}")

    # Save stage 2
    stage2 = df[['Title', 'Year', 'countries_origin', 'filming_locations', 'budget',
                 'production_company', 'Languages', 'genres']].copy()
    stage2['origin_countries_parsed'] = df['origin_countries'].apply(str)
    stage2['n_origin_countries'] = df['n_origin_countries']
    stage2.to_csv(f'{OUTPUT_DIR}\\stage2_parsed_lists.csv', index=False)

    # --- 2d: Extract filming country ---
    print("\n--- 2d: Extract filming country ---")

    filming_map_file = f'{PROJECT_DIR}\\data\\raw\\base\\imdb_filming_location_mappings.csv'
    filming_map_df = pd.read_csv(filming_map_file, keep_default_na=False)
    FILMING_LOC_TO_STANDARD = {}
    for _, row in filming_map_df.iterrows():
        raw = row['raw_string']
        std = row['standardised_name'] if row['standardised_name'] != '' else None
        FILMING_LOC_TO_STANDARD[raw] = std
    print(f"Loaded {len(FILMING_LOC_TO_STANDARD)} filming location mappings")

    def extract_filming_country(loc_str):
        if pd.isna(loc_str):
            return None
        try:
            locs = ast.literal_eval(loc_str)
        except (ValueError, SyntaxError):
            return None
        for loc in locs:
            loc_clean = re.sub(r'\(.*?\)', '', loc).strip()
            parts = [p.strip() for p in loc_clean.split(',')]
            if not parts:
                continue
            raw_country = parts[-1]
            if raw_country in FILMING_LOC_TO_STANDARD:
                return FILMING_LOC_TO_STANDARD[raw_country]
            else:
                return raw_country
        return None

    df['filming_country'] = df['filming_locations'].apply(extract_filming_country)
    n_with_filming = df['filming_country'].notna().sum()
    print(f"Films with extracted filming country: {n_with_filming:,} / {len(df):,} "
          f"({100*n_with_filming/len(df):.1f}%)")

    filming_counts = df['filming_country'].value_counts().head(25)
    print(f"\nTop 25 filming countries:")
    for country, count in filming_counts.items():
        print(f"  {country}: {count}")

    df['filming_abroad'] = df.apply(
        lambda row: (row['filming_country'] is not None and
                     row['filming_country'] not in row['origin_countries']),
        axis=1
    )
    n_abroad = df['filming_abroad'].sum()
    n_with_both = df[(df['filming_country'].notna()) & (df['n_origin_countries'] > 0)].shape[0]
    print(f"\nFilms filmed abroad: {n_abroad:,} / {n_with_both:,} ({100*n_abroad/n_with_both:.1f}%)")

    # Save stage 3
    stage3 = df[['Title', 'Year', 'countries_origin', 'filming_locations',
                 'filming_country', 'filming_abroad', 'n_origin_countries', 'budget']].copy()
    stage3['origin_countries_parsed'] = df['origin_countries'].apply(str)
    stage3.to_csv(f'{OUTPUT_DIR}\\stage3_filming_country.csv', index=False)

    # --- 2e: Harmonise to ISO2 ---
    print("\n--- 2e: Harmonise country names to ISO2 ---")

    iso_map_file = f'{PROJECT_DIR}\\data\\raw\\base\\imdb_country_name_to_iso2.csv'
    iso_map_df = pd.read_csv(iso_map_file)
    COUNTRY_TO_ISO2 = dict(zip(iso_map_df['country_name'], iso_map_df['iso2']))
    print(f"Loaded {len(COUNTRY_TO_ISO2)} country-to-ISO2 mappings")

    def country_name_to_iso2_imdb(name):
        if pd.isna(name) or name is None:
            return None
        name = name.strip()
        return COUNTRY_TO_ISO2.get(name, None)

    def origins_to_iso2(country_list):
        codes = []
        for name in country_list:
            code = country_name_to_iso2_imdb(name)
            if code:
                codes.append(code)
        return codes

    df['origin_iso2'] = df['origin_countries'].apply(origins_to_iso2)
    df['filming_iso2'] = df['filming_country'].apply(country_name_to_iso2_imdb)

    all_origin_names = set()
    for lst in df['origin_countries']:
        all_origin_names.update(lst)

    unmapped_origins = {n for n in all_origin_names if country_name_to_iso2_imdb(n) is None}
    unmapped_filming = set(
        df[df['filming_country'].notna() & df['filming_iso2'].isna()]['filming_country'].unique()
    )

    print(f"Origin countries: {len(all_origin_names)} unique, {len(unmapped_origins)} unmapped")
    if unmapped_origins:
        print(f"  Unmapped origins: {sorted(unmapped_origins)}")
    print(f"Filming countries: {df['filming_country'].nunique()} unique, {len(unmapped_filming)} unmapped")
    if unmapped_filming:
        print(f"  Unmapped filming: {sorted(unmapped_filming)}")

    # Save stage 4
    stage4 = df[['Title', 'Year', 'countries_origin', 'filming_locations',
                 'filming_country', 'filming_iso2', 'n_origin_countries', 'budget']].copy()
    stage4['origin_iso2'] = df['origin_iso2'].apply(str)
    stage4.to_csv(f'{OUTPUT_DIR}\\stage4_iso2_mapped.csv', index=False)

    # --- 2f: Parse budget (USD only) ---
    print("\n--- 2f: Parse budget ---")

    def parse_budget_usd(budget_str):
        if pd.isna(budget_str):
            return None
        s = str(budget_str).strip()
        if not s.startswith('$'):
            return None
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

    # Save stage 5
    stage5 = df[['Title', 'Year', 'countries_origin', 'filming_locations',
                 'filming_country', 'filming_iso2', 'n_origin_countries',
                 'budget', 'budget_usd']].copy()
    stage5['origin_iso2'] = df['origin_iso2'].apply(str)
    stage5.to_csv(f'{OUTPUT_DIR}\\stage5_with_budgets.csv', index=False)

    # --- 2g: Deflate budgets to constant 2010 USD ---
    print("\n--- 2g: Deflate budgets (US CPI, base year 2010) ---")

    CPI_FILE = f'{PROJECT_DIR}\\data\\raw\\us_cpi.csv'
    try:
        cpi = pd.read_csv(CPI_FILE)
        print(f"Loaded CPI data: {len(cpi)} rows")

        if 'date' in cpi.columns:
            cpi = cpi.rename(columns={'date': 'year'})
        if 'value' in cpi.columns:
            cpi = cpi.rename(columns={'value': 'cpi'})

        cpi_dict = dict(zip(cpi['year'].astype(int), cpi['cpi']))
        cpi_base = cpi_dict.get(CPI_BASE_YEAR_IMDB, None)

        if cpi_base is None:
            print(f"  WARNING: No CPI value for base year {CPI_BASE_YEAR_IMDB}")
            print(f"  Skipping deflation — budgets remain nominal")
        else:
            print(f"  CPI base year: {CPI_BASE_YEAR_IMDB} (CPI = {cpi_base:.2f})")
            df['cpi'] = df['Year'].map(cpi_dict)
            n_missing_cpi = df[df['budget_usd'].notna() & df['cpi'].isna()].shape[0]
            if n_missing_cpi > 0:
                print(f"  WARNING: {n_missing_cpi} films with budget but no CPI")

            df['budget_usd_real'] = np.where(
                df['budget_usd'].notna() & df['cpi'].notna(),
                df['budget_usd'] * (cpi_base / df['cpi']),
                np.nan
            )

            nominal_median = df['budget_usd'].median()
            real_median = df['budget_usd_real'].median()
            print(f"  Median nominal budget: ${nominal_median:,.0f}")
            print(f"  Median real budget (2010 USD): ${real_median:,.0f}")

            df['budget_usd_nominal'] = df['budget_usd']
            df['budget_usd'] = df['budget_usd_real']
            print(f"  Budget column now contains constant {CPI_BASE_YEAR_IMDB} USD values")

    except FileNotFoundError:
        print(f"  WARNING: CPI file not found at {CPI_FILE}")
        print(f"  Skipping deflation — budgets remain nominal")

    # --- 2h: Build bilateral trade flows ---
    print("\n--- 2h: Build bilateral trade flows ---")

    trade_flows = []
    skipped_domestic = 0
    skipped_no_filming = 0
    skipped_no_origin = 0

    for idx, row in df.iterrows():
        filming = row['filming_iso2']
        origins = row['origin_iso2']
        year = row['Year']
        budget = row['budget_usd']

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
                'importer': origin,
                'exporter': filming,
                'film_count': 1,
                'budget_usd': budget,
                'title': row.get('Title', '')
            })

    df_flows = pd.DataFrame(trade_flows)

    print(f"Film-level bilateral flows: {len(df_flows):,}")
    print(f"Skipped - no filming location: {skipped_no_filming:,}")
    print(f"Skipped - no valid origin: {skipped_no_origin:,}")
    print(f"Skipped - domestic (origin=filming): {skipped_domestic:,}")

    if len(df_flows) == 0:
        print("ERROR: No trade flows generated. Check data.")
        return

    print(f"Flows with budget: {df_flows['budget_usd'].notna().sum():,} "
          f"({100*df_flows['budget_usd'].notna().sum()/len(df_flows):.1f}%)")

    # --- 2i: Aggregate to pair-year ---
    print("\n--- 2i: Aggregate to pair-year level ---")

    df_bilateral = df_flows.groupby(['year', 'importer', 'exporter']).agg(
        num_films=('film_count', 'sum'),
        total_budget=('budget_usd', 'sum'),
        films_with_budget=('budget_usd', 'count'),
    ).reset_index()

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

    print(f"\nTop 15 bilateral corridors (by total films):")
    top_corridors = df_bilateral.groupby(['importer', 'exporter'])['num_films'].sum() \
        .sort_values(ascending=False).head(15)
    for (imp, exp), count in top_corridors.items():
        print(f"  {imp} -> {exp}: {count} films")

    # --- Save final IMDb outputs ---
    print("\n--- Saving IMDb outputs ---")
    df_bilateral.to_csv(f'{OUTPUT_DIR}\\imdb_filming_trade_flows.csv', index=False)
    df_flows.to_csv(f'{OUTPUT_DIR}\\imdb_filming_flows_by_film.csv', index=False)

    keep_cols = ['Title', 'Year', 'countries_origin', 'filming_locations',
                 'filming_country', 'filming_iso2', 'origin_iso2', 'n_origin_countries',
                 'budget', 'budget_usd', 'budget_usd_nominal', 'production_company',
                 'Languages', 'genres']
    df[[c for c in keep_cols if c in df.columns]].to_csv(
        f'{OUTPUT_DIR}\\imdb_combined_cleaned.csv', index=False
    )

    print(f"  Total films processed: {len(df):,}")
    print(f"  Bilateral pair-year observations: {len(df_bilateral):,}")
    print(f"  Film-level flows: {len(df_flows):,}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3: PRE-FILTER BaTIS SK1/SK2                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def phase_3_batis_prefilter():
    """Extract SK1 and SK2 audiovisual services from full BaTIS bulk CSV."""

    print("\n" + "=" * 70)
    print("PHASE 3: PRE-FILTER BaTIS SK1/SK2")
    print("=" * 70)

    batis_bulk_file = (
        r'C:\Users\kurtl\Downloads\OECD-WTO_BATIS_data_BPM6-1'
        r'\OECD-WTO_BATIS_BPM6_December2025_bulk.csv'
    )
    chunk_size = 1_000_000

    for item_code, label in [('SK1', 'Audiovisual and related services'),
                              ('SK2', 'Other personal, cultural, recreational services')]:
        print(f"\n--- {item_code}: {label} ---")
        chunks = []
        for chunk in pd.read_csv(batis_bulk_file, chunksize=chunk_size):
            filtered = chunk[
                (chunk['Item_code'] == item_code) &
                (chunk['Year'] >= 2005)
            ]
            if len(filtered) > 0:
                chunks.append(filtered)
            print(f"  {item_code}: Processed chunk, found {len(filtered)} rows")

        data = pd.concat(chunks, ignore_index=True)
        exports = data[data['Flow'] == 'X'].copy()
        exports = exports[['Reporter', 'Partner', 'Year', 'Balanced_value']]

        output_file = f'{PROJECT_DIR}\\data\\raw\\base\\batis_{item_code.lower()}_all_years.csv'
        exports.to_csv(output_file, index=False)
        print(f"  {item_code} exports: {len(exports):,} rows, "
              f"years {exports['Year'].min()}-{exports['Year'].max()}")
        print(f"  Saved {output_file}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4: CLEAN CEPII DISTANCE/GRAVITY DATA                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def phase_4_cepii():
    """Clean CEPII Gravity dataset: filter, convert ISO codes, derive variables."""

    print("\n" + "=" * 70)
    print("PHASE 4: CLEAN CEPII DISTANCE/GRAVITY DATA")
    print("=" * 70)

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
        f"{PROJECT_DIR}/data/raw/base/Gravity_V202211.csv",
        usecols=cols_to_keep
    )
    print(f"Loaded: {len(df):,} observations")

    # Filter to 1990-2018
    df = df[(df['year'] >= 1990) & (df['year'] <= 2018)]
    print(f"After filtering to 1990-2018: {len(df):,} observations")

    # Convert ISO3 → ISO2
    print("Converting ISO3 to ISO2 codes...")
    df['iso_o'] = df['iso3_o'].apply(iso3_to_iso2)
    df['iso_d'] = df['iso3_d'].apply(iso3_to_iso2)

    matched = df['iso_o'].notna().sum()
    total = len(df)
    print(f"Origin conversion: {matched:,} / {total:,} ({matched/total*100:.1f}%)")

    failed_o = df[df['iso_o'].isna()]['iso3_o'].unique()
    if len(failed_o) > 0:
        print(f"Failed origin codes ({len(failed_o)}): {failed_o[:20].tolist()}")

    df = df.dropna(subset=['iso_o', 'iso_d'])
    print(f"After dropping failed conversions: {len(df):,} observations")

    # Derived variables
    df['log_dist'] = np.log(df['dist'])
    df['rta'] = (df['fta_wto'] == 1).astype(int)
    df['both_eu'] = ((df['eu_o'] == 1) & (df['eu_d'] == 1)).astype(int)

    # Select final columns
    df_final = df[[
        'year', 'iso_o', 'iso_d',
        'dist', 'distcap', 'log_dist',
        'contig', 'comlang_off', 'comlang_ethno',
        'col45', 'col_dep_ever', 'comrelig',
        'rta', 'both_eu'
    ]].copy()

    # Deduplicate
    print(f"\nBefore deduplication: {len(df_final):,} observations")
    df_final['completeness'] = df_final[['dist', 'contig', 'comlang_off', 'rta']].notna().sum(axis=1)
    df_final = df_final.sort_values('completeness', ascending=False)
    df_final = df_final.drop_duplicates(subset=['iso_o', 'iso_d', 'year'], keep='first')
    df_final = df_final.drop(columns=['completeness'])
    print(f"After deduplication: {len(df_final):,} observations")

    # Remove self-pairs
    before_count = len(df_final)
    df_final = df_final[df_final['iso_o'] != df_final['iso_d']]
    print(f"Removed {before_count - len(df_final):,} self-pair observations")

    # Summary
    print(f"\n=== Final CEPII Dataset ===")
    print(f"Observations: {len(df_final):,}")
    print(f"Unique origin countries: {df_final['iso_o'].nunique()}")
    print(f"Unique destination countries: {df_final['iso_d'].nunique()}")
    print(f"Year range: {df_final['year'].min()} - {df_final['year'].max()}")

    for col in ['dist', 'contig', 'comlang_off', 'rta', 'comrelig']:
        non_missing = df_final[col].notna().sum()
        print(f"  {col}: {non_missing:,} non-missing ({non_missing/len(df_final)*100:.1f}%)")

    # Save
    output_file = f"{PROJECT_DIR}/data/processed/gravity_vars_cepii.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 5: CLEAN GDP DATA                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def phase_5_gdp():
    """Clean World Bank GDP: remove aggregates, convert names to ISO2, log transform."""

    print("\n" + "=" * 70)
    print("PHASE 5: CLEAN GDP DATA")
    print("=" * 70)

    df = pd.read_csv(f"{PROJECT_DIR}/data/raw/world_bank_gdp.csv")
    print(f"Loaded: {len(df)} observations")
    print(f"Missing GDP: {df['gdp'].isnull().sum()}, Zero/negative: {(df['gdp'] <= 0).sum()}")

    # Remove regional aggregates
    df = df[~df['country'].isin(WB_AGGREGATES)]
    print(f"After removing aggregates: {len(df)} rows, {df['country'].nunique()} countries")

    # Drop missing GDP
    df = df.dropna(subset=['gdp'])
    print(f"After dropping missing GDP: {len(df)} rows")

    # Convert country names to ISO2
    df['iso2'] = df['country'].apply(country_name_to_iso2_wb)

    matched = df['iso2'].notna().sum()
    unmatched = df['iso2'].isna().sum()
    print(f"ISO conversion: {matched} matched, {unmatched} unmatched")

    if unmatched > 0:
        unmatched_countries = df[df['iso2'].isna()]['country'].unique()
        print(f"Unmatched ({len(unmatched_countries)}):")
        for c in unmatched_countries:
            print(f"  - {c}")

    df['country'] = df['iso2']
    df = df.drop(columns=['iso2'])
    df = df.dropna(subset=['country'])

    # Deduplicate
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['country', 'date'], keep='first')
    if before_dedup > len(df):
        print(f"Dropped {before_dedup - len(df)} duplicate country-year observations")

    # Log GDP
    df['log_gdp'] = np.log(df['gdp'])

    # Summary
    print(f"\n=== Final GDP Dataset ===")
    print(f"Observations: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Year range: {df['date'].min()} - {df['date'].max()}")

    # Save
    output_file = f"{PROJECT_DIR}/data/processed/gdp_cleaned.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 6: CLEAN ECONOMIC FREEDOM OF THE WORLD (EFW) DATA               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def phase_6_efw():
    """Clean EFW data: convert to ISO2, interpolate missing years."""

    print("\n" + "=" * 70)
    print("PHASE 6: CLEAN ECONOMIC FREEDOM OF THE WORLD DATA")
    print("=" * 70)

    df = pd.read_excel(
        f"{PROJECT_DIR}/data/raw/base/efotw-2025-master-index-data-for-researchers-iso.xlsx",
        header=3
    )
    print(f"Loaded: {len(df)} observations, years {df['Year'].min()}-{df['Year'].max()}")

    # Select columns
    df = df[['Year', 'ISO_Code', 'Countries', 'ECONOMIC FREEDOM ALL AREAS']].copy()
    df.columns = ['year', 'iso3', 'country_name', 'efw']

    # Filter to 1990-2018
    df = df[(df['year'] >= 1990) & (df['year'] <= 2018)]
    print(f"After filtering to 1990-2018: {len(df)} observations")

    # Convert ISO3 → ISO2
    df['iso2'] = df['iso3'].apply(iso3_to_iso2)

    matched = df['iso2'].notna().sum()
    unmatched = df['iso2'].isna().sum()
    print(f"ISO conversion: {matched} matched, {unmatched} unmatched")

    if unmatched > 0:
        failed_codes = df[df['iso2'].isna()][['iso3', 'country_name']].drop_duplicates()
        print(f"Failed codes:")
        print(failed_codes)

    df = df.dropna(subset=['iso2'])

    # Drop missing EFW
    print(f"Missing EFW values: {df['efw'].isna().sum()}")
    df = df.dropna(subset=['efw'])

    # Year coverage before interpolation
    print(f"\nYear coverage before interpolation:")
    print(df.groupby('year').size())

    # Interpolate missing years (EFW published every 5 years pre-2000)
    print("\nInterpolating missing years...")

    df_final = df[['year', 'iso2', 'efw']].copy()
    df_final.columns = ['year', 'country', 'efw']

    countries = df_final['country'].unique()
    all_years = list(range(1990, 2019))

    full_grid = pd.DataFrame([
        {'country': c, 'year': y}
        for c in countries
        for y in all_years
    ])

    df_full = full_grid.merge(df_final, on=['country', 'year'], how='left')
    df_full = df_full.sort_values(['country', 'year'])
    df_full['efw'] = df_full.groupby('country')['efw'].transform(
        lambda x: x.interpolate(method='linear')
    )

    before_interp = df_final['efw'].notna().sum()
    after_interp = df_full['efw'].notna().sum()
    print(f"EFW before interpolation: {before_interp}")
    print(f"EFW after interpolation: {after_interp}")
    print(f"Interpolated values added: {after_interp - before_interp}")

    df_final = df_full

    # Summary
    print(f"\n=== Final EFW Dataset ===")
    print(f"Observations: {len(df_final)}")
    print(f"Unique countries: {df_final['country'].nunique()}")
    print(f"Year range: {df_final['year'].min()} - {df_final['year'].max()}")
    print(f"\nEFW summary statistics:")
    print(df_final['efw'].describe())

    # Save
    output_file = f"{PROJECT_DIR}/data/processed/efw_cleaned.csv"
    df_final.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Show countries with remaining missing values
    missing = df_final[df_final['efw'].isna()]
    if len(missing) > 0:
        print(f"\nCountries with remaining missing EFW:")
        print(missing.groupby('country')['year'].agg(['min', 'max', 'count']))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 7: PREPARE BaTIS SK1 GRAVITY ANALYSIS DATASET                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def phase_7_batis_prep():
    """
    Merge BaTIS SK1 audiovisual trade data with CEPII, GDP, EFW, incentives,
    and remoteness to create analysis-ready datasets.
    """

    print("\n" + "=" * 70)
    print("PHASE 7: PREPARE BaTIS SK1 GRAVITY ANALYSIS DATASET")
    print("=" * 70)

    # File paths
    SK1_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\batis_sk1_all_years.csv'
    CEPII_FILE = f'{PROJECT_DIR}\\data\\processed\\gravity_vars_cepii.csv'
    GDP_FILE = f'{PROJECT_DIR}\\data\\processed\\gdp_cleaned.csv'
    EFW_FILE = f'{PROJECT_DIR}\\data\\processed\\efw_cleaned.csv'
    INCENTIVE_FILE = f'{PROJECT_DIR}\\data\\raw\\base\\film_incentive_intro_dates.csv'
    CPI_FILE = f'{PROJECT_DIR}\\data\\raw\\world_bank_cpi.csv'
    ANALYSIS_FILE = f'{PROJECT_DIR}\\data\\processed\\batis_sk1_gravity_analysis.csv'
    MERGED_FILE = f'{PROJECT_DIR}\\data\\processed\\batis_sk1_gravity_merged.csv'

    # --- 7a: Load shared data ---
    print("\n--- 7a: Load shared data ---")

    # CEPII
    cepii = pd.read_csv(CEPII_FILE)
    print(f"CEPII: {len(cepii):,} rows, years {cepii['year'].min()}-{cepii['year'].max()}")

    cepii_max = cepii['year'].max()
    if YEAR_MAX_BATIS > cepii_max:
        print(f"Forward-filling CEPII from {cepii_max} to {YEAR_MAX_BATIS}...")
        cepii_latest = cepii[cepii['year'] == cepii_max].copy()
        new_rows = [cepii_latest.assign(year=y) for y in range(cepii_max + 1, YEAR_MAX_BATIS + 1)]
        cepii = pd.concat([cepii] + new_rows, ignore_index=True)

    # GDP
    gdp = pd.read_csv(GDP_FILE)
    print(f"GDP: {len(gdp):,} rows, years {gdp['date'].min()}-{gdp['date'].max()}")

    # CPI
    cpi = pd.read_csv(CPI_FILE)
    cpi = cpi.rename(columns={'date': 'year'}) if 'date' in cpi.columns else cpi
    cpi['year'] = cpi['year'].astype(int)
    print(f"CPI: {len(cpi)} years, range {cpi['year'].min()}-{cpi['year'].max()}")

    # EFW
    efw = pd.read_csv(EFW_FILE)
    print(f"EFW: {len(efw):,} rows, years {efw['year'].min()}-{efw['year'].max()}")
    efw_max = efw['year'].max()
    if efw_max < YEAR_MAX_BATIS:
        print(f"Forward-filling EFW from {efw_max} to {YEAR_MAX_BATIS}...")
        efw_latest = efw[efw['year'] == efw_max].copy()
        new_efw = [efw_latest.assign(year=y) for y in range(efw_max + 1, YEAR_MAX_BATIS + 1)]
        efw = pd.concat([efw] + new_efw, ignore_index=True)

    # Incentives
    incentives = pd.read_csv(INCENTIVE_FILE)
    print(f"Incentives: {len(incentives)} countries")

    incentive_dict = dict(zip(
        incentives['country_iso2'].astype(str),
        incentives['incentive_intro_year']
    ))

    TYPE_DUMMIES = ['is_refundable_credit', 'is_transferable_credit',
                    'is_standard_credit', 'is_cash_rebate']
    type_lookups = {}
    for dtype in TYPE_DUMMIES:
        if dtype in incentives.columns:
            type_lookups[dtype] = dict(zip(
                incentives['country_iso2'].astype(str),
                incentives[dtype].fillna(0).astype(int)
            ))
        else:
            type_lookups[dtype] = {}

    has_generosity = 'headline_rate_pct' in incentives.columns
    generosity_dict = {}
    if has_generosity:
        generosity_dict = dict(zip(
            incentives['country_iso2'].astype(str),
            incentives['headline_rate_pct'].fillna(0) / 100
        ))
        print(f"  Loaded headline_rate_pct for generosity analysis")

    # --- 7b: Load and clean BaTIS SK1 ---
    print("\n--- 7b: Load and clean BaTIS SK1 ---")

    batis = pd.read_csv(SK1_FILE)
    print(f"Loaded: {len(batis):,} rows, years {batis['Year'].min()}-{batis['Year'].max()}")

    batis['Reporter'] = batis['Reporter'].astype(str)
    batis['Partner'] = batis['Partner'].astype(str)

    # Drop aggregates and zero/negative values
    batis = batis[
        (batis['Reporter'] != '888') &
        (batis['Partner'] != '888') &
        (batis['Partner'] != 'WL')
    ]
    batis = batis[batis['Balanced_value'] > 0].copy()

    batis = batis.rename(columns={
        'Reporter': 'exporter',
        'Partner': 'importer',
        'Year': 'year',
        'Balanced_value': 'trade_value_nominal'
    })

    # Deflate to constant 2018 USD
    base_cpi = cpi[cpi['year'] == CPI_BASE_YEAR_BATIS]['cpi'].values[0]
    cpi_adj = cpi[['year', 'cpi']].copy()
    cpi_adj['adjustment_factor'] = base_cpi / cpi_adj['cpi']

    batis = batis.merge(cpi_adj[['year', 'adjustment_factor']], on='year', how='left')

    if batis['adjustment_factor'].isna().any():
        latest_factor = cpi_adj.loc[cpi_adj['year'] == cpi_adj['year'].max(), 'adjustment_factor'].values[0]
        n_filled = batis['adjustment_factor'].isna().sum()
        batis['adjustment_factor'] = batis['adjustment_factor'].fillna(latest_factor)
        print(f"  Filled {n_filled} obs with latest CPI factor")

    batis['trade_value'] = batis['trade_value_nominal'] * batis['adjustment_factor']
    batis['log_trade'] = np.log(batis['trade_value'])
    batis = batis.drop(columns=['adjustment_factor'])

    print(f"After cleaning: {len(batis):,} rows")
    print(f"Exporters: {batis['exporter'].nunique()}, Importers: {batis['importer'].nunique()}")
    print(f"Years: {batis['year'].min()}-{batis['year'].max()}")
    print(f"Deflated to {CPI_BASE_YEAR_BATIS} USD")

    # --- 7c: Merge with CEPII ---
    print("\n--- 7c: Merge with CEPII ---")

    merged = batis.merge(
        cepii, left_on=['exporter', 'importer', 'year'],
        right_on=['iso_o', 'iso_d', 'year'], how='left'
    )
    merged = merged.drop(columns=['iso_o', 'iso_d'], errors='ignore')
    merged = merged[merged['year'] <= YEAR_MAX_BATIS].copy()

    print(f"After CEPII merge: {len(merged):,} rows")
    print(f"Missing distance: {merged['log_dist'].isna().sum()} ({merged['log_dist'].isna().mean()*100:.1f}%)")

    # --- 7d: Merge with GDP ---
    print("\n--- 7d: Merge with GDP ---")

    merged = merged.merge(
        gdp[['country', 'date', 'log_gdp']].rename(
            columns={'country': 'importer', 'date': 'year', 'log_gdp': 'log_gdp_importer'}),
        on=['importer', 'year'], how='left'
    )
    merged = merged.merge(
        gdp[['country', 'date', 'log_gdp']].rename(
            columns={'country': 'exporter', 'date': 'year', 'log_gdp': 'log_gdp_exporter'}),
        on=['exporter', 'year'], how='left'
    )

    print(f"Missing importer GDP: {merged['log_gdp_importer'].isna().sum()} "
          f"({merged['log_gdp_importer'].isna().mean()*100:.1f}%)")
    print(f"Missing exporter GDP: {merged['log_gdp_exporter'].isna().sum()} "
          f"({merged['log_gdp_exporter'].isna().mean()*100:.1f}%)")

    # --- 7e: Merge with EFW ---
    print("\n--- 7e: Merge with EFW ---")

    merged = merged.merge(
        efw.rename(columns={'country': 'importer', 'efw': 'efw_importer'}),
        on=['importer', 'year'], how='left'
    )
    merged = merged.merge(
        efw.rename(columns={'country': 'exporter', 'efw': 'efw_exporter'}),
        on=['exporter', 'year'], how='left'
    )

    print(f"Missing importer EFW: {merged['efw_importer'].isna().sum()} "
          f"({merged['efw_importer'].isna().mean()*100:.1f}%)")
    print(f"Missing exporter EFW: {merged['efw_exporter'].isna().sum()} "
          f"({merged['efw_exporter'].isna().mean()*100:.1f}%)")

    # --- 7f: Create incentive variables ---
    print("\n--- 7f: Create incentive variables ---")

    merged['incentive_exporter'] = merged.apply(
        lambda row: 1 if row['exporter'] in incentive_dict
                         and pd.notna(incentive_dict[row['exporter']])
                         and row['year'] >= incentive_dict[row['exporter']] else 0,
        axis=1
    )
    merged['incentive_importer'] = merged.apply(
        lambda row: 1 if row['importer'] in incentive_dict
                         and pd.notna(incentive_dict[row['importer']])
                         and row['year'] >= incentive_dict[row['importer']] else 0,
        axis=1
    )

    # Incentive type dummies
    for dtype in TYPE_DUMMIES:
        col_exp = f'{dtype}_exp'
        merged[col_exp] = merged.apply(
            lambda row, dt=dtype: (
                1 if (row['exporter'] in incentive_dict
                      and pd.notna(incentive_dict[row['exporter']])
                      and row['year'] >= incentive_dict[row['exporter']]
                      and type_lookups[dt].get(row['exporter'], 0) == 1)
                else 0
            ), axis=1
        )

    # Generosity
    if has_generosity:
        merged['generosity_exp'] = merged.apply(
            lambda row: (
                generosity_dict.get(row['exporter'], 0)
                if (row['exporter'] in incentive_dict
                    and pd.notna(incentive_dict[row['exporter']])
                    and row['year'] >= incentive_dict[row['exporter']])
                else 0
            ), axis=1
        )
    else:
        merged['generosity_exp'] = 0

    print(f"Exporter incentive active: {merged['incentive_exporter'].sum()} "
          f"({merged['incentive_exporter'].mean()*100:.1f}%)")
    print(f"Importer incentive active: {merged['incentive_importer'].sum()} "
          f"({merged['incentive_importer'].mean()*100:.1f}%)")

    for dtype in TYPE_DUMMIES:
        col = f'{dtype}_exp'
        print(f"  {dtype}: {merged[col].sum()} ({merged[col].mean()*100:.1f}%)")

    if has_generosity:
        active_gen = merged[merged['generosity_exp'] > 0]['generosity_exp']
        if len(active_gen) > 0:
            print(f"\nGenerosity (active): mean={active_gen.mean()*100:.1f}%, "
                  f"min={active_gen.min()*100:.1f}%, max={active_gen.max()*100:.1f}%")

    # --- 7g: Calculate remoteness ---
    print("\n--- 7g: Calculate remoteness ---")

    gdp_levels = gdp[['country', 'date', 'gdp']].rename(columns={'date': 'year'}).copy()
    gdp_levels = gdp_levels.dropna(subset=['gdp'])
    world_gdp = gdp_levels.groupby('year')['gdp'].sum().reset_index()
    world_gdp.columns = ['year', 'gdp_world']

    dist_year = cepii[cepii['year'] <= 2020]['year'].max()
    dist_data = cepii[cepii['year'] == dist_year][['iso_o', 'iso_d', 'dist']].copy()
    dist_data = dist_data.dropna(subset=['dist'])
    dist_data = dist_data[(dist_data['dist'] > 0) & (dist_data['iso_o'] != dist_data['iso_d'])]

    all_countries = sorted(
        (set(dist_data['iso_o'].dropna().unique()) | set(dist_data['iso_d'].dropna().unique())) - {np.nan}
    )
    all_years = [y for y in sorted(gdp_levels['year'].unique()) if 2005 <= y <= YEAR_MAX_BATIS]

    remoteness_records = []
    for yr in all_years:
        gdp_yr = gdp_levels[gdp_levels['year'] == yr].set_index('country')['gdp'].to_dict()
        gdp_world_yr = world_gdp[world_gdp['year'] == yr]['gdp_world'].values
        if len(gdp_world_yr) == 0:
            continue
        gdp_world_yr = gdp_world_yr[0]

        for country in all_countries:
            dists = dist_data[dist_data['iso_o'] == country][['iso_d', 'dist']]
            if len(dists) == 0:
                continue
            weighted_sum = 0
            for _, row in dists.iterrows():
                partner = row['iso_d']
                if partner == country:
                    continue
                partner_gdp = gdp_yr.get(partner, None)
                if partner_gdp and partner_gdp > 0:
                    weighted_sum += (partner_gdp / gdp_world_yr) / row['dist']
            if weighted_sum > 0:
                remoteness_records.append({
                    'country': country, 'year': yr,
                    'remoteness': np.log(1 / weighted_sum)
                })

        if yr % 5 == 0:
            print(f"  Processed remoteness for {yr}")

    remoteness_df = pd.DataFrame(remoteness_records)
    print(f"Remoteness calculated: {len(remoteness_df):,} country-year observations")

    merged = merged.merge(
        remoteness_df.rename(columns={'country': 'importer', 'remoteness': 'remoteness_importer'}),
        on=['importer', 'year'], how='left'
    )
    merged = merged.merge(
        remoteness_df.rename(columns={'country': 'exporter', 'remoteness': 'remoteness_exporter'}),
        on=['exporter', 'year'], how='left'
    )

    print(f"Missing importer remoteness: {merged['remoteness_importer'].isna().sum()}")
    print(f"Missing exporter remoteness: {merged['remoteness_exporter'].isna().sum()}")

    # --- 7h: Prepare and save analysis datasets ---
    print("\n--- 7h: Prepare and save analysis datasets ---")

    est_vars = [
        'log_trade', 'log_gdp_importer', 'log_gdp_exporter',
        'log_dist', 'contig', 'comlang_off', 'col45',
        'rta', 'remoteness_importer', 'remoteness_exporter',
        'efw_importer', 'efw_exporter',
        'incentive_exporter', 'incentive_importer'
    ]

    print(f"Total observations: {len(merged):,}")
    print(f"\nMissing values:")
    for var in est_vars:
        n_miss = merged[var].isna().sum()
        if n_miss > 0:
            print(f"  {var}: {n_miss} ({n_miss/len(merged)*100:.1f}%)")

    # Complete cases (with EFW)
    df = merged.dropna(subset=est_vars).copy()
    print(f"\nComplete cases (with EFW): {len(df):,}")
    print(f"Exporters: {df['exporter'].nunique()}, Importers: {df['importer'].nunique()}")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")

    # Save both datasets
    merged.to_csv(MERGED_FILE, index=False)
    df.to_csv(ANALYSIS_FILE, index=False)

    print(f"\nSaved:")
    print(f"  Merged (all obs):    {MERGED_FILE} ({len(merged):,} rows)")
    print(f"  Analysis (complete): {ANALYSIS_FILE} ({len(df):,} rows)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN EXECUTION                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    print("=" * 70)
    print("UNIFIED DATA FETCH AND CLEAN PIPELINE")
    print("=" * 70)

    if RUN_PHASE_1_FETCH_WB:
        phase_1_fetch_world_bank()

    if RUN_PHASE_2_IMDB:
        phase_2_imdb()

    if RUN_PHASE_3_BATIS_PREFILTER:
        phase_3_batis_prefilter()

    if RUN_PHASE_4_CEPII:
        phase_4_cepii()

    if RUN_PHASE_5_GDP:
        phase_5_gdp()

    if RUN_PHASE_6_EFW:
        phase_6_efw()

    if RUN_PHASE_7_BATIS_PREP:
        phase_7_batis_prep()

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)