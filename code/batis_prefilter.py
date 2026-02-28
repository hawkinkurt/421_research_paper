"""batis_prefilter.py"""
""" Program to fetch SK1 audiovisual services data from BaTIS (all available years) """

import pandas as pd

# Read in chunks and filter as we go
chunks = []
chunk_size = 1_000_000  # 1 million rows at a time

for chunk in pd.read_csv(
    r'C:\Users\kurtl\Downloads\OECD-WTO_BATIS_data_BPM6-1\OECD-WTO_BATIS_BPM6_December2025_bulk.csv',
    chunksize=chunk_size
):
    # Filter each chunk for SK1 (audiovisual services), all available years
    filtered = chunk[
        (chunk['Item_code'] == 'SK1') &
        (chunk['Year'] >= 2005)
    ]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"Processed chunk, found {len(filtered)} matching rows")

# Combine all filtered chunks
sk1_data = pd.concat(chunks, ignore_index=True)
print(f"\nTotal filtered rows: {len(sk1_data):,}")
print(f"Unique Flows: {sk1_data['Flow'].unique()}")
print(f"Year range: {sk1_data['Year'].min()} - {sk1_data['Year'].max()}")

# Keep only exports
exports = sk1_data[sk1_data['Flow'] == 'X'].copy()
exports = exports[['Reporter', 'Partner', 'Year', 'Balanced_value']]

print(f"\nExports only: {len(exports):,} rows")
print(f"Reporter codes sample: {list(exports['Reporter'].astype(str).unique()[:10])}")
print(f"Year range: {exports['Year'].min()} - {exports['Year'].max()}")

# Save
exports.to_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\raw\base\batis_sk1_all_years.csv', index=False)
print(f"\nSaved to data/raw/base/batis_sk1_all_years.csv")