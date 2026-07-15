import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
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

def main():
    print("Loading sustainbench data...")
    df = pd.read_csv("sustainbench_full.csv")
    
    # Target and groups
    target_col = 'yield_t_ha'
    df['harvest_year'] = df['year']  # match function names
    df['adm_id'] = df['region1']     # use county as region for split
    
    ignore_cols = ['year', 'harvest_year', 'key', 'region1', 'region2', 'adm_id']
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Filter out min columns which are constant 0
    predictor_cols = [c for c in numeric_cols if c not in ignore_cols and c != target_col and not c.endswith('_min')]
    
    df = df.dropna(subset=predictor_cols)
    X = df[predictor_cols].values
    y = df[target_col].values
    
    protocols = ['random', 'spatial', 'temporal', 'spatiotemporal']
    seeds = [42, 100, 999]
    
    results = defaultdict(list)
    
    for protocol in protocols:
        for seed in seeds:
            train_idx, test_idx = get_splits(df, protocol, seed)
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            
            rmse = root_mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            results[protocol].append({'rmse': rmse, 'r2': r2})
            
    print("\n--- PILOT EXPERIMENT RESULTS ---")
    summary = {}
    for p in protocols:
        rmse_mean = np.mean([r['rmse'] for r in results[p]])
        rmse_std = np.std([r['rmse'] for r in results[p]])
        r2_mean = np.mean([r['r2'] for r in results[p]])
        r2_std = np.std([r['r2'] for r in results[p]])
        
        summary[p] = {'rmse': rmse_mean, 'r2': r2_mean}
        
        print(f"{p.upper()}:")
        print(f"  RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}")
        print(f"  R²:   {r2_mean:.3f} ± {r2_std:.3f}")
        
    print("\n--- PERFORMANCE GAP (Naive Random - Honest Split) ---")
    for p in ['spatial', 'temporal', 'spatiotemporal']:
        rmse_gap = summary[p]['rmse'] - summary['random']['rmse']  # positive means worse error
        r2_gap = summary['random']['r2'] - summary[p]['r2']      # positive means worse R2
        
        print(f"{p.upper()} Gap:")
        print(f"  RMSE Penalty: +{rmse_gap:.3f}")
        print(f"  R² Drop:      -{r2_gap:.3f}")

if __name__ == '__main__':
    main()
