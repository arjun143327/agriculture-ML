import sys
import numpy as np
import pandas as pd

sys.path.append("d:/agriculture research paper/sustainbench")
from sustainbench.datasets.crop_yield_dataset import CropYieldDataset

print("Initializing SustainBench CropYieldDataset (this may download data)...")
dataset = CropYieldDataset(root_dir='d:/agriculture research paper/data', download=True)

print("Dataset initialized.")
print("Metadata fields:", dataset.metadata_fields)

# The crop yield dataset stores metadata in dataset.metadata, which is a pandas DataFrame
df = dataset.metadata.copy()

print("\n--- Data Sample Loaded ---")
print("Shape:", df.shape)
print("\nColumns and dtypes:")
print(df.info())

print("\nFirst 10 rows:")
print(df.head(10))

# Save a sample
sample_size = min(1000, len(df))
sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df.copy()
sample_df.to_csv("d:/agriculture research paper/sustainbench_sample.csv", index=False)
print(f"\nSaved {len(sample_df)} rows to sustainbench_sample.csv")

