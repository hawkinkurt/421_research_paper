"""robustness_checks_data.py"""
""" Program to fetch/clean robustness check data """

import pandas as pd

# Read in chunks and filter as we go
chunks = []
chunk_size = 1_000_000  # 1 million rows at a time

for chunk in pd.read_csv(
    r'C:\Users\kurtl\Downloads\OECD-WTO_BATIS_data_BPM6-1\OECD-WTO_BATIS_BPM6_December2025_bulk.csv',
    chunksize=chunk_size
):
    # Filter each chunk for SK1 (audiovisual services), years 2005-2018
    filtered = chunk[
        (chunk['Item_code'] == 'SK1') &
        (chunk['Year'] >= 2005) &
        (chunk['Year'] <= 2018)
    ]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"Processed chunk, found {len(filtered)} matching rows")

# Combine all filtered chunks
sk1_data = pd.concat(chunks, ignore_index=True)
print(f"\nTotal filtered rows: {len(sk1_data):,}")
print(f"Unique Flows: {sk1_data['Flow'].unique()}")

# Keep only exports (to match your film production proxy approach)
exports = sk1_data[sk1_data['Flow'] == 'X'].copy()
exports = exports[['Reporter', 'Partner', 'Year', 'Balanced_value']]

print(f"\nExports only: {len(exports):,} rows")
print(f"Reporter codes sample: {list(exports['Reporter'].astype(str).unique()[:10])}")
print(f"Year range: {exports['Year'].min()} - {exports['Year'].max()}")

# Save to raw/base folder
exports.to_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\raw\base\batis_sk1_exports.csv', index=False)
print(f"\nSaved to data/raw/base/batis_sk1_exports.csv")

for chunk in pd.read_csv(
    r'C:\Users\kurtl\Downloads\OECD-WTO_BATIS_data_BPM6-1\OECD-WTO_BATIS_BPM6_December2025_bulk.csv',
    chunksize=chunk_size
):
    # Filter each chunk for SK2 (other personal/cultural/recreational), years 2005-2018
    filtered = chunk[
        (chunk['Item_code'] == 'SK2') &
        (chunk['Year'] >= 2005) &
        (chunk['Year'] <= 2018)
    ]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"Processed chunk, found {len(filtered)} matching rows")

# Combine all filtered chunks
sk2_data = pd.concat(chunks, ignore_index=True)
print(f"\nTotal filtered rows: {len(sk2_data):,}")
print(f"Unique Flows: {sk2_data['Flow'].unique()}")

# Keep only exports (to match your film production proxy approach)
exports = sk2_data[sk2_data['Flow'] == 'X'].copy()
exports = exports[['Reporter', 'Partner', 'Year', 'Balanced_value']]

print(f"\nExports only: {len(exports):,} rows")
print(f"Reporter codes sample: {list(exports['Reporter'].astype(str).unique()[:10])}")
print(f"Year range: {exports['Year'].min()} - {exports['Year'].max()}")

# Save to raw/base folder
exports.to_csv(r'C:\Users\kurtl\PycharmProjects\gravity_model\data\raw\base\batis_sk2_exports.csv', index=False)
print(f"\nSaved to data/raw/base/batis_sk2_exports.csv")