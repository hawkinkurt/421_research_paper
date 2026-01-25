"""gdp_data_cleaning.py"""
import wbdata
import pandas as pd
from datetime import datetime

# Define the date range
start_date = datetime(2000, 1, 1)
end_date = datetime(2022,12,31)

# Indicators
indicators = {"NY.GDP.MKTP.CD": "gdp"}

# Fetch data
df = wbdata.get_dataframe(indicators, date=(start_date, end_date))

# Reset index to turn the multi-index into columns
df = df.reset_index()

# Save to raw data folder
df.to_csv("../data/raw/world_bank_gdp.csv", index=False)

print(df.head())
