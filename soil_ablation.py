import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

IGNORE_COLS = ['harvest_year', 'latitude', 'longitude', 'harvest_area',
               'production', 'country_code', 'adm_id',
               'crop_name_x', 'crop_name_y'] 

DATASETS = [
    {
        'name':      'CY-Bench Maize (Europe)',
        'csv':       'cybench_full.csv',
        'yield_col': 'yield',
        'group_col': 'adm_id',
    },
    {
        'name':      'CY-Bench Wheat (Europe)',
        'csv':       'cybench_wheat_full.csv',
        'yield_col': 'yield',
        'group_col': 'adm_id',
    },
    {
        'name':      'CY-Bench Maize (Zambia)',
        'csv':       'cybench_zambia_maize.csv',
        'yield_col': 'yield',
        'group_col': 'adm_id',
    },
]

SEEDS = [42, 43, 44]
STATIC_FEATURES = ['bulk_density', 'awc', 'drainage_class']
MODELS = {
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XGBoost':       XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1),
}

def load_data(ds_cfg, drop_static=False):
    df = pd.read_csv(ds_cfg['csv'])
    yc = ds_cfg['yield_col']
    df = df[df[yc] > 0].copy()
    numeric = df.select_dtypes(include='number').columns
    pred_cols = [c for c in numeric if c not in IGNORE_COLS and c != yc]
    
    if drop_static:
        pred_cols = [c for c in pred_cols if c not in STATIC_FEATURES]
        
    df = df.dropna(subset=pred_cols)
    return df, df[pred_cols].values, df[yc].values, df[ds_cfg['group_col']].values

results = []

for ds in DATASETS:
    for feat_set in ['Full', 'No-Soil']:
        drop_static = (feat_set == 'No-Soil')
        df, X, y, groups = load_data(ds, drop_static=drop_static)
        unique_regions = np.unique(groups)
        
        for mname, base_model in MODELS.items():
            for split_type in ['Random', 'Spatial']:
                rmses = []
                r2s = []
                for seed in SEEDS:
                    np.random.seed(seed)
                    if split_type == 'Random':
                        n = len(y)
                        indices = np.random.permutation(n)
                        train_idx = indices[:int(0.8*n)]
                        test_idx = indices[int(0.8*n):]
                    else:
                        test_regions = np.random.choice(unique_regions, size=max(1, int(len(unique_regions) * 0.2)), replace=False)
                        train_idx = np.where(~np.isin(groups, test_regions))[0]
                        test_idx = np.where(np.isin(groups, test_regions))[0]
                    
                    if len(test_idx) == 0 or len(train_idx) == 0:
                        continue
                    
                    model = sklearn.base.clone(base_model)
                    model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[test_idx])
                    
                    rmses.append(root_mean_squared_error(y[test_idx], preds))
                    r2s.append(r2_score(y[test_idx], preds))
                    
                if r2s:
                    results.append({
                        'dataset': ds['name'],
                        'model': mname,
                        'split': split_type,
                        'feature_set': feat_set,
                        'rmse_mean': np.mean(rmses),
                        'rmse_sd': np.std(rmses),
                        'r2_mean': np.mean(r2s),
                        'r2_sd': np.std(r2s)
                    })

res_df = pd.DataFrame(results)

# Calculate Random-vs-Spatial Gap
gap_data = []
for ds in DATASETS:
    for mname in MODELS.keys():
        for feat_set in ['Full', 'No-Soil']:
            sub = res_df[(res_df['dataset'] == ds['name']) & (res_df['model'] == mname) & (res_df['feature_set'] == feat_set)]
            
            rand_row = sub[sub['split'] == 'Random']
            spat_row = sub[sub['split'] == 'Spatial']
            
            if not rand_row.empty and not spat_row.empty:
                r2_rand = rand_row.iloc[0]['r2_mean']
                r2_spat = spat_row.iloc[0]['r2_mean']
                gap = r2_rand - r2_spat
                
                # Add to gap tracking
                gap_data.append({
                    'dataset': ds['name'],
                    'model': mname,
                    'feature_set': feat_set,
                    'gap': gap
                })

gap_df = pd.DataFrame(gap_data)

# Print output
with open('ablation_report.txt', 'w') as f:
    f.write("SOIL ABLATION EXPERIMENT\n")
    f.write("="*80 + "\n\n")
    
    f.write("| Dataset | Model | Feature_Set | Split | RMSE | R2 | Random-vs-Spatial Gap |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    
    for ds in DATASETS:
        for mname in MODELS.keys():
            for feat_set in ['Full', 'No-Soil']:
                # Get the gap
                gap_val = gap_df[(gap_df['dataset']==ds['name']) & (gap_df['model']==mname) & (gap_df['feature_set']==feat_set)]['gap'].values[0]
                
                for split in ['Random', 'Spatial']:
                    sub = res_df[(res_df['dataset']==ds['name']) & (res_df['model']==mname) & (res_df['feature_set']==feat_set) & (res_df['split']==split)]
                    r = sub.iloc[0]
                    f.write(f"| {r['dataset']} | {r['model']} | {r['feature_set']} | {r['split']} | {r['rmse_mean']:.3f} ± {r['rmse_sd']:.3f} | {r['r2_mean']:.3f} ± {r['r2_sd']:.3f} | {gap_val:.3f} |\n")

    f.write("\n\n--- KEY COMPARISONS ---\n")
    for ds in DATASETS:
        f.write(f"\n# {ds['name']}\n")
        for mname in MODELS.keys():
            gap_full = gap_df[(gap_df['dataset']==ds['name']) & (gap_df['model']==mname) & (gap_df['feature_set']=='Full')]['gap'].values[0]
            gap_nosoil = gap_df[(gap_df['dataset']==ds['name']) & (gap_df['model']==mname) & (gap_df['feature_set']=='No-Soil')]['gap'].values[0]
            
            rand_full = res_df[(res_df['dataset']==ds['name']) & (res_df['model']==mname) & (res_df['feature_set']=='Full') & (res_df['split']=='Random')]['r2_mean'].values[0]
            rand_nosoil = res_df[(res_df['dataset']==ds['name']) & (res_df['model']==mname) & (res_df['feature_set']=='No-Soil') & (res_df['split']=='Random')]['r2_mean'].values[0]
            
            spat_full = res_df[(res_df['dataset']==ds['name']) & (res_df['model']==mname) & (res_df['feature_set']=='Full') & (res_df['split']=='Spatial')]['r2_mean'].values[0]
            spat_nosoil = res_df[(res_df['dataset']==ds['name']) & (res_df['model']==mname) & (res_df['feature_set']=='No-Soil') & (res_df['split']=='Spatial')]['r2_mean'].values[0]
            
            f.write(f"{mname}:\n")
            f.write(f"  Gap: {gap_full:.3f} (Full) -> {gap_nosoil:.3f} (No-Soil)\n")
            f.write(f"  Random R2: {rand_full:.3f} -> {rand_nosoil:.3f}\n")
            f.write(f"  Spatial R2: {spat_full:.3f} -> {spat_nosoil:.3f}\n")

print("Done writing ablation_report.txt")
