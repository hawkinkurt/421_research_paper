"""film_data_cleaning.py"""
import pandas as pd
import ast

# =============================================================================
# LOAD DATA
# =============================================================================

# Load movies data
df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/movie_data_1.csv")
print(f"Total films loaded: {len(df)}")

# Load studio mappings
high_confidence = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/studio_mappings_high_confidence.csv")
low_confidence = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/studio_mappings_low_confidence.csv")

# Create dictionaries for mapping
high_conf_dict = dict(zip(high_confidence['company_name'], high_confidence['home_country']))
low_conf_dict = dict(zip(low_confidence['company_name'], low_confidence['home_country']))

print(f"High confidence mappings loaded: {len(high_conf_dict)}")
print(f"Low confidence mappings loaded: {len(low_conf_dict)}")

# =============================================================================
# CLEAN AND FILTER DATA
# =============================================================================

# Convert budget to numeric
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

# Filter for usable data
df_clean = df[
    (df['budget'] > 0) &
    (df['production_companies'].notna()) &
    (df['production_countries'].notna())
].copy()
print(f"Films with usable data: {len(df_clean)}")

# Extract year from release_date
df_clean['year'] = pd.to_datetime(df_clean['release_date'], errors='coerce').dt.year

# Filter to year range (1990-2023)
df_clean = df_clean[(df_clean['year'] >= 1990) & (df_clean['year'] <= 2023)]
print(f"Films in 1990-2023: {len(df_clean)}")

# =============================================================================
# PARSE PRODUCTION DATA
# =============================================================================

# Function to safely parse string representations of lists
def parse_list_column(value):
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value)
    except:
        return []

# Parse production_companies and production_countries
df_clean['production_companies_parsed'] = df_clean['production_companies'].apply(parse_list_column)
df_clean['production_countries_parsed'] = df_clean['production_countries'].apply(parse_list_column)

# Extract country ISO codes from production_countries
def get_country_codes(parsed_list):
    return [item['iso_3166_1'] for item in parsed_list if 'iso_3166_1' in item]

df_clean['country_codes'] = df_clean['production_countries_parsed'].apply(get_country_codes)

# Count number of production countries per film
df_clean['num_countries'] = df_clean['country_codes'].apply(len)

# Filter to films with 2 or more production countries
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
# ASSIGN HOME COUNTRY WITH CONFIDENCE LEVELS
# =============================================================================

# First try high confidence mappings
df_multi['home_country'] = df_multi['lead_company'].map(high_conf_dict)
df_multi['mapping_confidence'] = df_multi['home_country'].apply(
    lambda x: 'high' if pd.notna(x) else None
)

# Then try low confidence mappings for remaining unmapped films
unmapped_mask = df_multi['home_country'].isna()
df_multi.loc[unmapped_mask, 'home_country'] = df_multi.loc[unmapped_mask, 'lead_company'].map(low_conf_dict)
df_multi.loc[unmapped_mask & df_multi['home_country'].notna(), 'mapping_confidence'] = 'low'

# Filter out films with empty production companies
df_multi = df_multi[df_multi['lead_company'].notna()]
df_multi = df_multi[df_multi['lead_company'].notna()].copy()
df_multi = df_multi[df_multi['home_country'].notna()].copy()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n=== MAPPING RESULTS ===")
total = len(df_multi)
high_mapped = (df_multi['mapping_confidence'] == 'high').sum()
low_mapped = (df_multi['mapping_confidence'] == 'low').sum()
unmapped = df_multi['home_country'].isna().sum()

print(f"Total films: {total}")
print(f"High confidence mappings: {high_mapped} ({high_mapped/total*100:.1f}%)")
print(f"Low confidence mappings: {low_mapped} ({low_mapped/total*100:.1f}%)")
print(f"Total mapped: {high_mapped + low_mapped} ({(high_mapped + low_mapped)/total*100:.1f}%)")
print(f"Still unmapped: {unmapped} ({unmapped/total*100:.1f}%)")

print("\n=== HOME COUNTRY DISTRIBUTION ===")
print(df_multi['home_country'].value_counts().head(15))

print("\n=== SAMPLE DATA ===")
print(df_multi[['title', 'year', 'budget', 'home_country', 'mapping_confidence', 'country_codes']].head(10))

# =============================================================================
# SAVE FINAL DATASET
# =============================================================================

# Save all films (including unmapped for reference)
df_multi.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_with_home_country_all.csv", index=False)

# Save only mapped films (for analysis)
df_mapped = df_multi[df_multi['home_country'].notna()].copy()
df_mapped.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_with_home_country_mapped.csv", index=False)

# Save high confidence only (for conservative analysis)
df_high_conf = df_multi[df_multi['mapping_confidence'] == 'high'].copy()
df_high_conf.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_with_home_country_high_confidence.csv", index=False)

print(f"\n=== FILES SAVED ===")
print(f"All films: {len(df_multi)} (including {unmapped} unmapped)")
print(f"All mapped films: {len(df_mapped)}")
print(f"High confidence only: {len(df_high_conf)}")

# Check which companies are still unmapped
unmapped = df_multi[df_multi['home_country'].isna()]['lead_company'].value_counts()
print(f"Unmapped companies: {len(unmapped)}")
print(unmapped)

# Check films with missing home_country
unmapped_films = df_multi[df_multi['home_country'].isna()]
print(f"Total unmapped films: {len(unmapped_films)}")

# Check if lead_company is null for any
print(f"Films with null lead_company: {unmapped_films['lead_company'].isna().sum()}")

# Look at some unmapped films
print("\n=== Sample Unmapped Films ===")
print(unmapped_films[['title', 'lead_company', 'production_companies_parsed']].head(20))

# Check year distribution in original films data
df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_with_home_country_mapped.csv")

print("=== Year Distribution ===")
print(df['year'].value_counts().sort_index())

print(f"\nYear range: {df['year'].min()} to {df['year'].max()}")

# =============================================================================
# BUDGET CORRECTIONS
# =============================================================================

print("\n=== BUDGET CORRECTIONS ===")

# Load budget corrections from CSV
df_corrections = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/budget_corrections.csv")

print(f"Films before corrections: {len(df_multi)}")

# Apply corrections where corrected_budget is not null
corrections_to_apply = df_corrections[df_corrections['corrected_budget'].notna()]
corrections_made = 0

for _, row in corrections_to_apply.iterrows():
    mask = (df_multi['title'] == row['title']) & (df_multi['year'] == row['year'])
    if mask.any():
        old_budget = df_multi.loc[mask, 'budget'].values[0]
        df_multi.loc[mask, 'budget'] = row['corrected_budget']
        print(f"Corrected: {row['title']} ({int(row['year'])}): ${old_budget:,.0f} -> ${row['corrected_budget']:,.0f}")
        corrections_made += 1

print(f"\nBudget corrections applied: {corrections_made}")

# Drop films where corrected_budget is null (unverifiable)
films_to_drop = df_corrections[df_corrections['corrected_budget'].isna()]
films_before_drop = len(df_multi)

for _, row in films_to_drop.iterrows():
    mask = (df_multi['title'] == row['title']) & (df_multi['year'] == row['year'])
    df_multi = df_multi[~mask]

films_dropped = films_before_drop - len(df_multi)
print(f"Films dropped (unverifiable budgets): {films_dropped}")
print(f"Films after corrections: {len(df_multi)}")

# =============================================================================
# SAVE CORRECTED DATA
# =============================================================================

df_multi.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_with_home_country_mapped_corrected.csv", index=False)

df_high_conf = df_multi[df_multi['mapping_confidence'] == 'high'].copy()
df_high_conf.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_with_home_country_high_confidence_corrected.csv", index=False)

print(f"\n=== CORRECTED FILES SAVED ===")
print(f"All mapped films (corrected): {len(df_multi)}")
print(f"High confidence only (corrected): {len(df_high_conf)}")

# Verify no suspicious budgets remain
suspicious = df_multi[df_multi['budget'] < 100000]
print(f"\nRemaining films with budget < $100,000: {len(suspicious)}")
if len(suspicious) > 0:
    print(suspicious[['title', 'year', 'budget']])