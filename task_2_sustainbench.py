import zipfile
import numpy as np
import pandas as pd
import os

def process_sustainbench():
    zip_path = r"D:\agriculture research paper\agriculture-ML\crop_yield\soybeans_updated.zip"
    print(f"Reading {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Load train split for USA
        prefix = 'soybeans/usa/'
        
        print("Loading yields, years, keys...")
        with z.open(f'{prefix}train_yields.npz') as f:
            yields = np.load(f)['data']
        with z.open(f'{prefix}train_years.npz') as f:
            years = np.load(f)['data']
        with z.open(f'{prefix}train_keys.npz') as f:
            keys = np.load(f)['data']
            
        print("Loading histograms (this might take a moment)...")
        with z.open(f'{prefix}train_hists.npz') as f:
            # Shape is expected to be (N, 32, 32, 9)
            hists = np.load(f)['data']
            
    print(f"Loaded {len(yields)} records. Histograms shape: {hists.shape}")
    
    # Compute summary statistics per band: mean, std, min, max
    # axis 1 and 2 are the 32x32 dimensions (assuming shape N, 32, 32, 9)
    # 9 bands total
    
    print("Computing summary statistics per band...")
    means = hists.mean(axis=(1, 2))
    stds = hists.std(axis=(1, 2))
    mins = hists.min(axis=(1, 2))
    maxes = hists.max(axis=(1, 2))
    
    # Create column names
    bands = range(9)
    mean_cols = [f'band_{b}_mean' for b in bands]
    std_cols = [f'band_{b}_std' for b in bands]
    min_cols = [f'band_{b}_min' for b in bands]
    max_cols = [f'band_{b}_max' for b in bands]
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'key': keys,
        'year': years,
        'yield_t_ha': yields.flatten() if yields.ndim > 1 else yields
    })
    
    df[['region1', 'region2']] = df['key'].str.split('_', expand=True, n=1)
    
    # Add stats
    df[mean_cols] = means
    df[std_cols] = stds
    df[min_cols] = mins
    df[max_cols] = maxes
    
    output_file = "sustainbench_full.csv"
    df.to_csv(output_file, index=False)
    
    print("\n--- TASK 2 DONE ---")
    print(f"Total Rows: {len(df)}")
    print(f"Unique Counties: {df['region1'].nunique()}")
    print(f"Total Columns: {len(df.columns)}")

if __name__ == '__main__':
    process_sustainbench()
