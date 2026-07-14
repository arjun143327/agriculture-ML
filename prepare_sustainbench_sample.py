import zipfile
import numpy as np
import pandas as pd
import os

zip_path = r"D:\agriculture research paper\crop_yield\soybeans_updated.zip"

print(f"Reading {zip_path}")
try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Check files inside the zip
        files = z.namelist()
        print(f"Found {len(files)} files in zip.")
        
        # We need yields, years, keys for a specific country, e.g., 'usa'
        country = 'usa'
        prefix = f'soybeans/{country}/'
        
        # Load train split
        yields_path = f'{prefix}train_yields.npz'
        years_path = f'{prefix}train_years.npz'
        keys_path = f'{prefix}train_keys.npz'
        
        if yields_path in files:
            with z.open(yields_path) as f:
                yields = np.load(f)['data']
            with z.open(years_path) as f:
                years = np.load(f)['data']
            with z.open(keys_path) as f:
                keys = np.load(f)['data']
                
            df = pd.DataFrame({
                'key': keys,
                'year': years,
                'yield_t_ha': yields.flatten() if yields.ndim > 1 else yields
            })
            
            # The key is often region1_region2, e.g., loc1_loc2
            # Let's split it
            df[['region1', 'region2']] = df['key'].str.split('_', expand=True, n=1)
            
            print("\n--- SustainBench Crop Yield (USA Train Split) ---")
            print("Shape:", df.shape)
            print("\nColumns and dtypes:")
            print(df.info())
            
            print("\nFirst 10 rows:")
            print(df.head(10))
            
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            sample_df.to_csv(r"D:\agriculture research paper\sustainbench_sample.csv", index=False)
            print(f"\nSaved {len(sample_df)} rows to sustainbench_sample.csv")
        else:
            print(f"Could not find {yields_path} in the zip file.")
            print("Available files:")
            print(files[:20])
except Exception as e:
    print(f"Error: {e}")
