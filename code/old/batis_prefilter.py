"""batis_prefilter.py"""
""" Program to fetch SK1 and SK2 from the full BaTIS dataset (all available years) """

import pandas as pd

chunk_size = 1_000_000

# --- SK1: Audiovisual and related services ---
chunks_sk1 = []
for chunk in pd.read_csv(
    r'C:\Users\kurtl\Downloads\OECD-WTO_BATIS_data_BPM6-1\OECD-WTO_BATIS_BPM6_December2025_bulk.csv',
    chunksize=chunk_size
):
    filtered = chunk[
        (chunk['Item_code'] == 'SK1') &
        (chunk['Year'] >= 2005)
    ]
    if len(filtered) > 0:
        chunks_sk1.append(filtered)
    print(f"SK1: Processed chunk, found {len(filtered)} matching rows")

sk1_data = pd.concat(chunks_sk1, ignore_index=True)
exports_sk1 = sk1_data[sk1_data['Flow'] == 'X'].copy()
exports_sk1 = exports_sk1[['Reporter', 'Partner', 'Year', 'Balanced_value']]

print(f"\nSK1 exports: {len(exports_sk1):,} rows, years {exports_sk1['Year'].min()}-{exports_sk1['Year'].max()}")
exports_sk1.to_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\raw\base\batis_sk1_all_years.csv', index=False)
print("Saved batis_sk1_all_years.csv")

# --- SK2: Other personal, cultural, recreational services ---
chunks_sk2 = []
for chunk in pd.read_csv(
    r'C:\Users\kurtl\Downloads\OECD-WTO_BATIS_data_BPM6-1\OECD-WTO_BATIS_BPM6_December2025_bulk.csv',
    chunksize=chunk_size
):
    filtered = chunk[
        (chunk['Item_code'] == 'SK2') &
        (chunk['Year'] >= 2005)
    ]
    if len(filtered) > 0:
        chunks_sk2.append(filtered)
    print(f"SK2: Processed chunk, found {len(filtered)} matching rows")

sk2_data = pd.concat(chunks_sk2, ignore_index=True)
exports_sk2 = sk2_data[sk2_data['Flow'] == 'X'].copy()
exports_sk2 = exports_sk2[['Reporter', 'Partner', 'Year', 'Balanced_value']]

print(f"\nSK2 exports: {len(exports_sk2):,} rows, years {exports_sk2['Year'].min()}-{exports_sk2['Year'].max()}")
exports_sk2.to_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\raw\base\batis_sk2_all_years.csv', index=False)
print("Saved batis_sk2_all_years.csv")