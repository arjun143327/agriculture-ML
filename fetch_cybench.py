import requests
import json
import pandas as pd
import io

# Zenodo record ID for CY-Bench
record_id = '11502142'
url = f'https://zenodo.org/api/records/{record_id}'

print("Fetching Zenodo record...")
response = requests.get(url)
response.raise_for_status()
data = response.json()

print(f"Title: {data['metadata']['title']}")
print("Files available:")
csv_files = []
for f in data['files']:
    print(f"- {f['key']} (size: {f['size']} bytes)")
    if f['key'].endswith('.csv'):
        csv_files.append(f)
        
if not csv_files:
    print("No CSV files found. Looking for other tabular formats (parquet, etc.)")
    for f in data['files']:
        if f['key'].endswith('.parquet') or f['key'].endswith('.zip'):
            csv_files.append(f)

# Find a representative file, e.g., one crop, one region, or smallest subset
# Pick the smallest file for inspection if there are multiple.
if csv_files:
    # Sort by size to get smallest
    csv_files.sort(key=lambda x: x['size'])
    target_file = csv_files[0]
    print(f"\nSelecting target file for sample: {target_file['key']}")
    download_url = target_file['links']['self']
    
    # Download file
    print(f"Downloading from {download_url}...")
    r = requests.get(download_url)
    r.raise_for_status()
    
    # Check extension
    if target_file['key'].endswith('.csv'):
        df = pd.read_csv(io.BytesIO(r.content))
    elif target_file['key'].endswith('.parquet'):
        df = pd.read_parquet(io.BytesIO(r.content))
    else:
        print(f"Cannot read file type automatically for {target_file['key']}")
        exit(1)
        
    print("\n--- Data Sample Loaded ---")
    print("Shape:", df.shape)
    
    # Save a small sample (e.g. 500-1000 rows) as local CSV called cybench_sample.csv
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df.copy()
    sample_df.to_csv("cybench_sample.csv", index=False)
    print(f"\nSaved {len(sample_df)} rows to cybench_sample.csv")
else:
    print("No suitable files found to inspect.")
