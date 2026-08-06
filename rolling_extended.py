import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
WINDOW_SIZES  = [2, 3, 5]
MIN_TRAIN_YRS = 5        
REGION_FRAC   = 0.20     
SEEDS         = [42, 43, 44]
MIN_TEST_OBS  = 10       

IGNORE_COLS = ['harvest_year', 'latitude', 'longitude', 'harvest_area',
               'production', 'country_code', 'adm_id',
               'crop_name_x', 'crop_name_y',  
               'year', 'key', 'region1', 'region2']  

ANOMALIES = {
    'CY-Bench Maize (Europe)':  [2018],
    'CY-Bench Wheat (Europe)':  [2017], # Using 2017 for wheat as per user prompt
    'CY-Bench Maize (Zambia)':  [2012, 2014],
    'SustainBench Soybean':     [2014, 2015, 2016], # 2014-2016 worst window
}

DATASETS = [
    {
        'name':      'CY-Bench Maize (Europe)',
        'csv':       'cybench_full.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
    },
    {
        'name':      'CY-Bench Wheat (Europe)',
        'csv':       'cybench_wheat_full.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
    },
    {
        'name':      'CY-Bench Maize (Zambia)',
        'csv':       'cybench_zambia_maize.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
    },
    {
        'name':      'SustainBench Soybean',
        'csv':       'sustainbench_full.csv',
        'yield_col': 'yield_t_ha',
        'year_col':  'year',
        'group_col': 'region1',
    },
]

MODELS = {
    'Ridge':         Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=1.0))]),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XGBoost':       XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1),
}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load(ds_cfg):
    df = pd.read_csv(ds_cfg['csv'])
    yc = ds_cfg['yield_col']
    df = df[df[yc] > 0].copy()
    numeric = df.select_dtypes(include='number').columns
    pred_cols = [c for c in numeric if c not in IGNORE_COLS and c != yc]
    df = df.dropna(subset=pred_cols)
    df['_year']  = df[ds_cfg['year_col']].astype(int)
    df['_group'] = df[ds_cfg['group_col']].astype(str)
    df['_yield'] = df[yc].values
    X = df[pred_cols].values
    return df, X, df['_yield'].values

def rolling_st(df, X, y, window_size, min_train_yrs, region_frac, seeds, models, min_obs):
    years          = df['_year'].values
    groups         = df['_group'].values
    unique_years   = np.sort(np.unique(years))
    unique_regions = np.unique(groups)

    results = []
    for wi, wy_start in enumerate(unique_years):
        test_years = unique_years[(unique_years >= wy_start) & (unique_years < wy_start + window_size)]
        if len(test_years) < window_size:
            continue                              
        train_years_avail = unique_years[unique_years < wy_start]
        if len(train_years_avail) < min_train_yrs:
            continue                              

        for seed in seeds:
            np.random.seed(seed)
            test_regions = np.random.choice(
                unique_regions,
                size=max(1, int(len(unique_regions) * region_frac)),
                replace=False
            )
            train_mask = (~np.isin(groups, test_regions)) & (~np.isin(years, test_years))
            test_mask  =  np.isin(groups, test_regions)  &  np.isin(years, test_years)
            tr_idx = np.where(train_mask)[0]
            te_idx = np.where(test_mask)[0]

            if len(te_idx) < min_obs or len(tr_idx) < 10:
                continue

            for mname, base_model in models.items():
                m = sklearn.base.clone(base_model)
                m.fit(X[tr_idx], y[tr_idx])
                preds = m.predict(X[te_idx])
                results.append({
                    'window_start': int(wy_start),
                    'window_end':   int(test_years[-1]),
                    'window_label': f"{wy_start}-{test_years[-1]}",
                    'model':        mname,
                    'seed':         seed,
                    'r2':           r2_score(y[te_idx], preds),
                })
    return pd.DataFrame(results)

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
final_rows = []

for ds in DATASETS:
    name = ds['name']
    df, X, y = load(ds)
    anomalous = ANOMALIES.get(name, [])
    
    # We only run 2 and 5, load 3 from existing CSVs to perfectly preserve them
    existing_3_file = f"rolling_st_{name.replace(' ','_').replace('(','').replace(')','')}.csv"
    if os.path.exists(existing_3_file):
        df_3 = pd.read_csv(existing_3_file)
        for _, r in df_3.iterrows():
            # Check anomaly
            w_str = str(r['window_label'])
            try:
                start, end = map(int, w_str.split('-'))
                w_years = list(range(start, end + 1))
            except:
                w_years = [int(w_str)]
            
            has_anomaly = any(ay in w_years for ay in anomalous)
            
            final_rows.append({
                'dataset': name,
                'window_length': 3,
                'window_label': r['window_label'],
                'model': r['model'],
                'r2_mean': r['r2_mean'],
                'has_anomaly': has_anomaly
            })
    
    for ws in [2, 5]:
        raw = rolling_st(df, X, y, ws, MIN_TRAIN_YRS, REGION_FRAC, SEEDS, MODELS, MIN_TEST_OBS)
        if raw.empty: continue
        summary = (raw.groupby(['window_label', 'window_start', 'model'])
                       .agg(r2_mean=('r2', 'mean')).reset_index())
        
        for _, r in summary.iterrows():
            w_str = str(r['window_label'])
            try:
                start, end = map(int, w_str.split('-'))
                w_years = list(range(start, end + 1))
            except:
                w_years = [int(w_str)]
            
            has_anomaly = any(ay in w_years for ay in anomalous)
            
            final_rows.append({
                'dataset': name,
                'window_length': ws,
                'window_label': r['window_label'],
                'model': r['model'],
                'r2_mean': r['r2_mean'],
                'has_anomaly': has_anomaly
            })

all_df = pd.DataFrame(final_rows)
all_df.to_csv('rolling_extended_results.csv', index=False)

# Format for output table
# We want dataset, window_length, window range, RF R2, Ridge R2, XGB R2, anomaly
pivot_df = all_df.pivot_table(index=['dataset', 'window_length', 'window_label', 'has_anomaly'], 
                              columns='model', values='r2_mean').reset_index()

pivot_df.to_csv('rolling_extended_results_pivot.csv', index=False)

# Print out
with open('extended_sweep_report.txt', 'w') as f:
    f.write("| Dataset | Win Length | Window | Anomaly? | Ridge R2 | RF R2 | XGBoost R2 |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for _, r in pivot_df.iterrows():
        anom = "Yes" if r['has_anomaly'] else "No"
        f.write(f"| {r['dataset']} | {r['window_length']} | {r['window_label']} | {anom} | {r['Ridge']:.3f} | {r['Random Forest']:.3f} | {r['XGBoost']:.3f} |\n")

print("Done")
