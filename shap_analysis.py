import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor

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

def load_data(ds_cfg):
    df = pd.read_csv(ds_cfg['csv'])
    yc = ds_cfg['yield_col']
    df = df[df[yc] > 0].copy()
    numeric = df.select_dtypes(include='number').columns
    pred_cols = [c for c in numeric if c not in IGNORE_COLS and c != yc]
    df = df.dropna(subset=pred_cols)
    return df, df[pred_cols].values, df[yc].values, df[ds_cfg['group_col']].values, pred_cols

results = []

for ds in DATASETS:
    df, X, y, groups, features = load_data(ds)
    unique_regions = np.unique(groups)
    
    for split_type in ['Random', 'Spatial']:
        shap_matrix = []
        gain_matrix = []
        
        for seed in SEEDS:
            np.random.seed(seed)
            if split_type == 'Random':
                # 80/20 random split
                n = len(y)
                indices = np.random.permutation(n)
                train_idx = indices[:int(0.8*n)]
                test_idx = indices[int(0.8*n):]
            else:
                # 80/20 spatial split
                test_regions = np.random.choice(unique_regions, size=max(1, int(len(unique_regions) * 0.2)), replace=False)
                train_idx = np.where(~np.isin(groups, test_regions))[0]
                test_idx = np.where(np.isin(groups, test_regions))[0]
            
            model = XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X[train_idx], y[train_idx])
            
            # Gain importance
            gain_matrix.append(model.feature_importances_)
            
            # SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[test_idx])
            # mean absolute shap per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_matrix.append(mean_abs_shap)
            
        shap_matrix = np.array(shap_matrix)
        gain_matrix = np.array(gain_matrix)
        
        shap_mean = shap_matrix.mean(axis=0)
        shap_sd = shap_matrix.std(axis=0)
        gain_mean = gain_matrix.mean(axis=0)
        
        # Calculate static share
        static_idx = [i for i, f in enumerate(features) if f in STATIC_FEATURES]
        total_shap = shap_mean.sum()
        static_shap = shap_mean[static_idx].sum()
        static_share = (static_shap / total_shap) * 100 if total_shap > 0 else 0
        
        # Store for table
        for i, f in enumerate(features):
            results.append({
                'dataset': ds['name'],
                'split': split_type,
                'feature': f,
                'shap_mean': shap_mean[i],
                'shap_sd': shap_sd[i],
                'gain_mean': gain_mean[i],
                'static_share': static_share
            })

results_df = pd.DataFrame(results)

with open('shap_report.txt', 'w') as f:
    f.write("SHAP ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    # 2. Combined SHAP importance share
    f.write("--- STATIC SOIL VARIABLE SHARE (SHAP) ---\n")
    for ds in DATASETS:
        for split in ['Random', 'Spatial']:
            sub = results_df[(results_df['dataset'] == ds['name']) & (results_df['split'] == split)]
            if not sub.empty:
                share = sub['static_share'].iloc[0]
                f.write(f"{ds['name']} | {split} : {share:.2f}%\n")
    f.write("\n")
    
    # 4. Rank Differences
    f.write("--- MEANINGFUL RANK DIFFERENCES (SHAP vs GAIN) ---\n")
    for ds in DATASETS:
        for split in ['Random', 'Spatial']:
            sub = results_df[(results_df['dataset'] == ds['name']) & (results_df['split'] == split)].copy()
            sub['shap_rank'] = sub['shap_mean'].rank(ascending=False)
            sub['gain_rank'] = sub['gain_mean'].rank(ascending=False)
            
            # Find big differences (e.g., in top 5 vs not in top 5)
            # or rank diff > 3
            sub['rank_diff'] = np.abs(sub['shap_rank'] - sub['gain_rank'])
            diffs = sub[(sub['rank_diff'] >= 3) | 
                        ((sub['shap_rank'] <= 5) & (sub['gain_rank'] > 5)) |
                        ((sub['gain_rank'] <= 5) & (sub['shap_rank'] > 5))]
            
            if not diffs.empty:
                f.write(f"\nDifferences in {ds['name']} | {split}:\n")
                for _, row in diffs.iterrows():
                    f.write(f"  {row['feature']}: SHAP Rank {int(row['shap_rank'])} vs Gain Rank {int(row['gain_rank'])}\n")
    
    f.write("\n")
    f.write("--- RAW TABLE ---\n")
    f.write("| Dataset | Split | Feature | SHAP_mean | SHAP_SD | Gain_Importance |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for ds in DATASETS:
        for split in ['Random', 'Spatial']:
            sub = results_df[(results_df['dataset'] == ds['name']) & (results_df['split'] == split)].copy()
            sub = sub.sort_values(by='shap_mean', ascending=False)
            for _, r in sub.iterrows():
                f.write(f"| {r['dataset']} | {r['split']} | {r['feature']} | {r['shap_mean']:.4f} | {r['shap_sd']:.4f} | {r['gain_mean']:.4f} |\n")

print("Done writing shap_report.txt")
