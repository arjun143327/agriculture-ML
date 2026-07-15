import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

def load_data(dataset_name):
    if dataset_name == 'CY-Bench':
        df = pd.read_csv('cybench_full.csv')
        df = df[df['yield'] > 0]
        ignore_cols = ['harvest_year', 'latitude', 'longitude', 'harvest_area', 'production', 'crop_name_x', 'crop_name_y', 'country_code', 'adm_id']
        numeric_cols = df.select_dtypes(include=['number']).columns
        predictor_cols = [c for c in numeric_cols if c not in ignore_cols and c != 'yield']
        df = df.dropna(subset=predictor_cols)
        X = df[predictor_cols].values
        y = df['yield'].values
        return df, X, y
        
    elif dataset_name == 'SustainBench':
        df = pd.read_csv('sustainbench_full.csv')
        df['harvest_year'] = df['year']
        df['adm_id'] = df['region1']
        ignore_cols = ['year', 'harvest_year', 'key', 'region1', 'region2', 'adm_id']
        numeric_cols = df.select_dtypes(include=['number']).columns
        predictor_cols = [c for c in numeric_cols if c not in ignore_cols and c != 'yield_t_ha' and not c.endswith('_min')]
        df = df.dropna(subset=predictor_cols)
        X = df[predictor_cols].values
        y = df['yield_t_ha'].values
        return df, X, y

print('=== PART 1: NULL BASELINE ===')
all_results = []
for ds_name in ['CY-Bench', 'SustainBench']:
    df, X, y = load_data(ds_name)
    groups = df['adm_id'].values
    years = df['harvest_year'].values
    unique_years = np.sort(np.unique(years))
    
    # SPATIAL NULL
    gkf = GroupKFold(n_splits=5)
    spatial_rmse, spatial_r2 = [], []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        train_mean = np.mean(y[train_idx])
        preds = np.full(len(test_idx), train_mean)
        spatial_rmse.append(root_mean_squared_error(y[test_idx], preds))
        spatial_r2.append(r2_score(y[test_idx], preds))
        
    all_results.append({
        'Dataset': ds_name,
        'Model': 'Null Baseline',
        'Split': 'Spatial (5-Fold GKF)',
        'RMSE': f"{np.mean(spatial_rmse):.3f} ± {np.std(spatial_rmse):.3f}",
        'R2': f"{np.mean(spatial_r2):.3f} ± {np.std(spatial_r2):.3f}"
    })
    
    # TEMPORAL NULL
    temporal_rmse, temporal_r2 = [], []
    for i in range(4, len(unique_years) - 1):
        train_years = unique_years[:i+1]
        test_year = unique_years[i+1]
        train_mask = np.isin(years, train_years)
        test_mask = (years == test_year)
        
        if not np.any(test_mask) or not np.any(train_mask):
            continue
            
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        train_mean = np.mean(y[train_idx])
        preds = np.full(len(test_idx), train_mean)
        temporal_rmse.append(root_mean_squared_error(y[test_idx], preds))
        temporal_r2.append(r2_score(y[test_idx], preds))
        
    all_results.append({
        'Dataset': ds_name,
        'Model': 'Null Baseline',
        'Split': 'Temporal (Walk-Forward)',
        'RMSE': f"{np.mean(temporal_rmse):.3f} ± {np.std(temporal_rmse):.3f}",
        'R2': f"{np.mean(temporal_r2):.3f} ± {np.std(temporal_r2):.3f}"
    })

print(pd.DataFrame(all_results).to_markdown(index=False))

print('\n=== PART 2: 2018 CY-BENCH ANOMALY ===')
df, _, _ = load_data('CY-Bench')
regions_2018 = df[df['harvest_year'] == 2018]['adm_id'].unique()

z_scores = []
means = []
stds = []
actuals = []

for r in regions_2018:
    hist_data = df[(df['adm_id'] == r) & (df['harvest_year'] >= 2001) & (df['harvest_year'] <= 2017)]['yield'].values
    val_2018 = df[(df['adm_id'] == r) & (df['harvest_year'] == 2018)]['yield'].values
    
    if len(hist_data) >= 5 and len(val_2018) == 1:
        hist_mean = np.mean(hist_data)
        hist_std = np.std(hist_data)
        if hist_std > 0:
            z = (val_2018[0] - hist_mean) / hist_std
            z_scores.append(z)
            means.append(hist_mean)
            stds.append(hist_std)
            actuals.append(val_2018[0])

print(f'Evaluated {len(z_scores)} regions for 2018 anomaly.')
print(f'Average historical (2001-2017) mean yield for these regions: {np.mean(means):.3f} t/ha')
print(f'Average historical std yield for these regions: {np.mean(stds):.3f} t/ha')
print(f'Average actual 2018 yield for these regions: {np.mean(actuals):.3f} t/ha')
print(f'Average Z-Score for 2018: {np.mean(z_scores):.3f} standard deviations (below historical mean)')
