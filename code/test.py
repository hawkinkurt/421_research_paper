import pandas as pd

df = pd.read_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/gravity_dataset_full.csv")

print("=== Missing Importer GDP ===")
missing_imp = df[df['gdp_importer'].isna()][['importer', 'year']].drop_duplicates()
print(f"Countries: {missing_imp['importer'].unique().tolist()}")

print("\n=== Missing Exporter GDP ===")
missing_exp = df[df['gdp_exporter'].isna()][['exporter', 'year']].drop_duplicates()
print(f"Countries: {missing_exp['exporter'].unique().tolist()}")

print("\n=== Missing Distance ===")
missing_dist = df[df['dist'].isna()][['importer', 'exporter']].drop_duplicates()
print(f"Pairs ({len(missing_dist)}):")
print(missing_dist)