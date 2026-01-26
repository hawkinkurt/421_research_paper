"""calculate_remoteness.py"""
import pandas as pd
import numpy as np

# =============================================================================
# LOAD DATA
# =============================================================================

df_gdp = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gdp_cleaned.csv")
df_grav = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_vars_cepii.csv")

# Rename GDP columns for clarity
df_gdp = df_gdp.rename(columns={'date': 'year', 'country': 'iso2'})

print("=== Data Loaded ===")
print(f"GDP observations: {len(df_gdp)}")
print(f"Gravity observations: {len(df_grav)}")

# =============================================================================
# GET UNIQUE DISTANCE PAIRS (distance doesn't vary by year)
# =============================================================================

# Take one year's distance data (they're the same across years)
df_dist = df_grav[df_grav['year'] == 2000][['iso_o', 'iso_d', 'dist']].copy()
df_dist = df_dist.dropna(subset=['dist'])

print(f"Unique country pairs with distance: {len(df_dist)}")

# =============================================================================
# CALCULATE WORLD GDP BY YEAR
# =============================================================================

world_gdp = df_gdp.groupby('year')['gdp'].sum().reset_index()
world_gdp.columns = ['year', 'gdp_world']

print("\n=== World GDP (sample years) ===")
print(world_gdp[world_gdp['year'].isin([1990, 2000, 2010, 2018])])

# =============================================================================
# CALCULATE REMOTENESS FOR EACH COUNTRY-YEAR
# =============================================================================

print("\n=== Calculating Remoteness ===")

# Get all country-year combinations from GDP data
country_years = df_gdp[['iso2', 'year']].drop_duplicates()

# Filter to 1990-2018
country_years = country_years[(country_years['year'] >= 1990) & (country_years['year'] <= 2018)]

remoteness_list = []

# Get unique years
years = sorted(country_years['year'].unique())

for year in years:
    # Get GDP data for this year
    gdp_year = df_gdp[df_gdp['year'] == year][['iso2', 'gdp']].copy()

    # Get world GDP for this year
    gdp_world = world_gdp[world_gdp['year'] == year]['gdp_world'].values[0]

    # Calculate GDP share for each country
    gdp_year['gdp_share'] = gdp_year['gdp'] / gdp_world

    # Get countries with GDP data this year
    countries = gdp_year['iso2'].unique()

    for country_i in countries:
        # Get distances from country_i to all other countries
        dist_i = df_dist[df_dist['iso_o'] == country_i][['iso_d', 'dist']].copy()

        # Merge with GDP shares
        dist_i = dist_i.merge(
            gdp_year[['iso2', 'gdp_share']],
            left_on='iso_d',
            right_on='iso2',
            how='inner'
        )

        if len(dist_i) > 0 and dist_i['dist'].sum() > 0:
            # Calculate remoteness: sum of (gdp_share / distance)
            # Then take inverse and log
            weighted_sum = (dist_i['gdp_share'] / dist_i['dist']).sum()

            if weighted_sum > 0:
                remoteness = np.log(1 / weighted_sum)
            else:
                remoteness = np.nan
        else:
            remoteness = np.nan

        remoteness_list.append({
            'year': year,
            'country': country_i,
            'remoteness': remoteness
        })

    if year % 5 == 0:
        print(f"Processed year {year}")

df_remoteness = pd.DataFrame(remoteness_list)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n=== Remoteness Summary ===")
print(f"Total observations: {len(df_remoteness)}")
print(f"Missing values: {df_remoteness['remoteness'].isna().sum()}")
print(f"Unique countries: {df_remoteness['country'].nunique()}")

print("\n=== Remoteness Statistics ===")
print(df_remoteness['remoteness'].describe())

# Show most and least remote countries (using 2010 as example)
print("\n=== Most Remote Countries (2010) ===")
remote_2010 = df_remoteness[df_remoteness['year'] == 2010].sort_values('remoteness', ascending=False)
print(remote_2010.head(10))

print("\n=== Least Remote Countries (2010) ===")
print(remote_2010.tail(10))

# =============================================================================
# SAVE
# =============================================================================

df_remoteness.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/remoteness.csv",
    index=False
)
print("\n=== Saved to remoteness.csv ===")