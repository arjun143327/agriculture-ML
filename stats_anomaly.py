import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

files = [
    ('CY-Bench Maize (Europe)', 'rolling_st_CY-Bench_Maize_Europe.csv'),
    ('CY-Bench Wheat (Europe)', 'rolling_st_CY-Bench_Wheat_Europe.csv'),
    ('CY-Bench Maize (Zambia)', 'rolling_st_CY-Bench_Maize_Zambia.csv'),
    ('SustainBench Soybean', 'rolling_st_SustainBench_Soybean.csv')
]

anomalies = {
    'CY-Bench Maize (Europe)': [2018],
    'CY-Bench Wheat (Europe)': [2017],
    'CY-Bench Maize (Zambia)': [2012, 2014],
    'SustainBench Soybean': [2014, 2015, 2016] # 2014-2016 regime shift
}

data = []

for ds_name, file in files:
    df = pd.read_csv(file)
    # Check if 'r2_mean' exists, otherwise look for 'R2' or similar
    if 'r2_mean' not in df.columns:
        print(f"Warning: 'r2_mean' not found in {file}, columns are: {df.columns}")
        continue
        
    for _, row in df.iterrows():
        # Parse window
        try:
            w_str = str(row['window_label'])
            if '-' in w_str:
                start, end = map(int, w_str.split('-'))
                window_years = list(range(start, end + 1))
            else:
                window_years = [int(w_str)]
        except:
            window_years = []
            
        # Check for anomaly
        has_anomaly = 0
        ds_anoms = anomalies.get(ds_name, [])
        for a in ds_anoms:
            if a in window_years:
                has_anomaly = 1
                break
                
        data.append({
            'dataset': ds_name,
            'model': row['model'],
            'r2': row['r2_mean'],
            'anomaly_in_window': has_anomaly
        })

df_all = pd.DataFrame(data)

# 1. Point-Biserial Correlation
print("1. Point-Biserial Correlation (Anomaly-in-Window vs R2)\n")
print("| Dataset | Correlation (r) | p-value |")
print("| :--- | :--- | :--- |")

for ds in df_all['dataset'].unique():
    sub = df_all[df_all['dataset'] == ds]
    if sub['anomaly_in_window'].nunique() > 1:
        r, p = stats.pointbiserialr(sub['anomaly_in_window'], sub['r2'])
        print(f"| {ds} | {r:.3f} | {p:.4f} |")
    else:
        print(f"| {ds} | N/A (no variation in anomaly flag) | N/A |")

r_pool, p_pool = stats.pointbiserialr(df_all['anomaly_in_window'], df_all['r2'])
print(f"| **Pooled (All Datasets)** | **{r_pool:.3f}** | **{p_pool:.4f}** |")

# 2. Linear Regression (controlling for dataset and model)
print("\n2. Linear Regression (R2 ~ anomaly_in_window + dataset + model)\n")
# Using formula API for easy dummy creation
model_ols = smf.ols("r2 ~ anomaly_in_window + C(dataset) + C(model)", data=df_all).fit()
print(model_ols.summary().tables[1].as_text())
print(f"Overall Model R2: {model_ols.rsquared:.4f}")
print(f"Coefficient on anomaly_in_window: {model_ols.params['anomaly_in_window']:.4f}")
print(f"p-value: {model_ols.pvalues['anomaly_in_window']:.4e}")

