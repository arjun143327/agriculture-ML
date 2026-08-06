import pandas as pd
import numpy as np
import glob

files = [
    ('CY-Bench Maize (Europe)', 'rolling_st_CY-Bench_Maize_Europe.csv'),
    ('CY-Bench Wheat (Europe)', 'rolling_st_CY-Bench_Wheat_Europe.csv'),
    ('CY-Bench Maize (Zambia)', 'rolling_st_CY-Bench_Maize_Zambia.csv'),
    ('SustainBench Soybean', 'rolling_st_SustainBench_Soybean.csv')
]

results = []

for ds_name, file in files:
    df = pd.read_csv(file)
    models = df['model'].unique()
    
    for m in models:
        sub = df[df['model'] == m].copy()
        
        # Fixed window is the last window chronologically
        # We find the row with the max window_start
        max_start = sub['window_start'].max()
        fixed_row = sub[sub['window_start'] == max_start].iloc[0]
        fixed_r2 = fixed_row['r2_mean']
        
        # Average across all windows
        avg_r2 = sub['r2_mean'].mean()
        std_r2 = sub['r2_mean'].std()
        
        # Absolute deviation
        abs_dev = abs(fixed_r2 - avg_r2)
        
        results.append({
            'Dataset': ds_name,
            'Model': m,
            'Fixed_Window': fixed_row['window_label'],
            'Fixed_Window_R2': fixed_r2,
            'Window_Avg_R2': avg_r2,
            'Window_R2_SD': std_r2,
            'Absolute_Deviation': abs_dev
        })

res_df = pd.DataFrame(results)

# Save to CSV
res_df.to_csv('spatiotemporal_comparison.csv', index=False)
print("Saved to spatiotemporal_comparison.csv")

print("\n--- RESULTS ---")
print(res_df.to_markdown(index=False, floatfmt=".3f"))
