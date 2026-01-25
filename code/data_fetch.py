"""data_fetch.py"""
import os
import wbdata
import kagglehub
import shutil
import pandas as pd
from datetime import datetime

# Download World Bank GDP data
filepath = "C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/world_bank_gdp.csv"

if os.path.exists(filepath):
    print("GDP data file already exists, skipping download")
    df = pd.read_csv(filepath)
else:
    print("Downloading data...")
    start_date = datetime(1990, 1, 1)
    end_date = datetime(2023, 12, 31)
    indicators = {"NY.GDP.MKTP.CD": "gdp"}
    df = wbdata.get_dataframe(indicators, date=(start_date, end_date))
    df = df.reset_index()
    df.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/world_bank_gdp.csv", index=False)
    print("Data saved")

# Download World Bank CPI data
filepath = "C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/world_bank_cpi.csv"

if os.path.exists(filepath):
    print("CPI data file already exists, skipping download")
    df = pd.read_csv(filepath)
else:
    print("Downloading data...")
    start_date = datetime(1990, 1, 1)
    end_date = datetime(2023, 12, 31)
    indicators = {"FP.CPI.TOTL": "cpi"}
    countries = ["USA"]
    df_cpi = wbdata.get_dataframe(indicators, country=countries, date=(start_date, end_date))
    df_cpi = df_cpi.reset_index()
    df_cpi.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/world_bank_cpi.csv", index=False)
    print("Data saved")

# Download movie data 1
filepath2 = "C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/movie_data_1.csv"

if os.path.exists(filepath2):
    print('Movie data file already exists, skipping download')
    df2 = pd.read_csv(filepath2)
else:
    print("Downloading data...")
    path = kagglehub.dataset_download("mehmetisik/movie-metadata")
    print(os.listdir(path))
    shutil.copy("C:/Users/kurtl/.cache/kagglehub/datasets/mehmetisik/movie-metadata/versions/1/movies_metadata.csv", filepath2)
    print("Data saved")