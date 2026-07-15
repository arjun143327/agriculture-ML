import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
import sklearn
warnings.filterwarnings('ignore')

def get_splits(df, protocol, seed, test_size=0.2):
    np.random.seed(seed)
    if protocol == 'random':
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed)
        return train_idx, test_idx
        
    elif protocol == 'spatial':
        regions = df['adm_id'].unique()
        test_regions = np.random.choice(regions, size=int(len(regions) * test_size), replace=False)
        test_mask = df['adm_id'].isin(test_regions)
        return np.where(~test_mask)[0], np.where(test_mask)[0]
        
    elif protocol == 'temporal':
        years = df['harvest_year'].unique()
        test_years = np.random.choice(years, size=int(len(years) * test_size), replace=False)
        test_mask = df['harvest_year'].isin(test_years)
        return np.where(~test_mask)[0], np.where(test_mask)[0]
        
    elif protocol == 'spatiotemporal':
        regions = df['adm_id'].unique()
        years = df['harvest_year'].unique()
        
        test_regions = np.random.choice(regions, size=int(len(regions) * test_size), replace=False)
        test_years = np.random.choice(years, size=int(len(years) * test_size), replace=False)
        
        test_mask = df['adm_id'].isin(test_regions) & df['harvest_year'].isin(test_years)
        train_mask = (~df['adm_id'].isin(test_regions)) & (~df['harvest_year'].isin(test_years))
        return np.where(train_mask)[0], np.where(test_mask)[0]

def load_data(dataset_name):
    if dataset_name == 'CY-Bench':
        df = pd.read_csv("cybench_full.csv")
        df = df[df['yield'] > 0]
        ignore_cols = ['harvest_year', 'latitude', 'longitude', 'harvest_area', 'production', 'crop_name_x', 'crop_name_y', 'country_code', 'adm_id']
        numeric_cols = df.select_dtypes(include=['number']).columns
        predictor_cols = [c for c in numeric_cols if c not in ignore_cols and c != 'yield']
        
        df = df.dropna(subset=predictor_cols)
        X = df[predictor_cols].values
        y = df['yield'].values
        return df, X, y
        
    elif dataset_name == 'SustainBench':
        df = pd.read_csv("sustainbench_full.csv")
        df['harvest_year'] = df['year']
        df['adm_id'] = df['region1']
        ignore_cols = ['year', 'harvest_year', 'key', 'region1', 'region2', 'adm_id']
        numeric_cols = df.select_dtypes(include=['number']).columns
        predictor_cols = [c for c in numeric_cols if c not in ignore_cols and c != 'yield_t_ha' and not c.endswith('_min')]
        
        df = df.dropna(subset=predictor_cols)
        X = df[predictor_cols].values
        y = df['yield_t_ha'].values
        return df, X, y

def main():
    datasets = ['CY-Bench', 'SustainBench']
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    models = {
        'Ridge (alpha=1.0)': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Ridge (alpha=10.0)': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    protocols = ['random', 'spatial', 'temporal', 'spatiotemporal']
    seeds = [42, 100, 999]
    
    all_results = []
    
    for ds_name in datasets:
        print(f"Running on {ds_name}...")
        df, X, y = load_data(ds_name)
        
        for model_name, base_model in models.items():
            results = defaultdict(list)
            
            for protocol in protocols:
                for seed in seeds:
                    train_idx, test_idx = get_splits(df, protocol, seed)
                    
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]
                    
                    model = sklearn.base.clone(base_model)
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    rmse = root_mean_squared_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    
                    results[protocol].append({'rmse': rmse, 'r2': r2})
            
            summary = {}
            for p in protocols:
                rmse_mean = np.mean([r['rmse'] for r in results[p]])
                rmse_std = np.std([r['rmse'] for r in results[p]])
                r2_mean = np.mean([r['r2'] for r in results[p]])
                r2_std = np.std([r['r2'] for r in results[p]])
                
                summary[p] = {'rmse': rmse_mean, 'r2': r2_mean, 'rmse_std': rmse_std, 'r2_std': r2_std}
                
            for p in protocols:
                if p == 'random':
                    rmse_gap = 0.0
                    r2_gap = 0.0
                else:
                    rmse_gap = summary[p]['rmse'] - summary['random']['rmse']
                    r2_gap = summary['random']['r2'] - summary[p]['r2']
                
                row = {
                    'Dataset': ds_name,
                    'Model': model_name,
                    'Split': p.capitalize(),
                    'RMSE': f"{summary[p]['rmse']:.3f} ± {summary[p]['rmse_std']:.3f}",
                    'R2': f"{summary[p]['r2']:.3f} ± {summary[p]['r2_std']:.3f}",
                    'Gap (RMSE)': f"+{rmse_gap:.3f}" if p != 'random' else "-",
                    'Gap (R2)': f"-{r2_gap:.3f}" if p != 'random' else "-"
                }
                all_results.append(row)
                
    out_df = pd.DataFrame(all_results)
    out_df.to_csv('pilot_results_summary.csv', index=False)
    
if __name__ == '__main__':
    main()
