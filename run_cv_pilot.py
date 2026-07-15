import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import sklearn
import warnings
warnings.filterwarnings('ignore')

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
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    all_results = []
    
    for ds_name in datasets:
        print(f"Running on {ds_name}...")
        df, X, y = load_data(ds_name)
        groups_spatial = df['adm_id'].values
        years = df['harvest_year'].values
        
        unique_years = np.sort(np.unique(years))
        
        for model_name, base_model in models.items():
            
            # --- SPATIAL (GroupKFold 5-splits) ---
            gkf = GroupKFold(n_splits=5)
            spatial_rmse = []
            spatial_r2 = []
            for train_idx, test_idx in gkf.split(X, y, groups=groups_spatial):
                model = sklearn.base.clone(base_model)
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[test_idx])
                spatial_rmse.append(root_mean_squared_error(y[test_idx], preds))
                spatial_r2.append(r2_score(y[test_idx], preds))
                
            all_results.append({
                'Dataset': ds_name,
                'Model': model_name,
                'Split': 'Spatial (5-Fold GKF)',
                'RMSE': f"{np.mean(spatial_rmse):.3f} ± {np.std(spatial_rmse):.3f}",
                'R2': f"{np.mean(spatial_r2):.3f} ± {np.std(spatial_r2):.3f}"
            })
            
            # --- TEMPORAL (Expanding Window Walk-Forward) ---
            temporal_rmse = []
            temporal_r2 = []
            
            # Start testing from the 5th available year to ensure enough training data
            for i in range(4, len(unique_years) - 1):
                train_years = unique_years[:i+1]
                test_year = unique_years[i+1]
                
                train_mask = np.isin(years, train_years)
                test_mask = (years == test_year)
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue
                    
                model = sklearn.base.clone(base_model)
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[test_idx])
                temporal_rmse.append(root_mean_squared_error(y[test_idx], preds))
                temporal_r2.append(r2_score(y[test_idx], preds))
                
            all_results.append({
                'Dataset': ds_name,
                'Model': model_name,
                'Split': 'Temporal (Walk-Forward)',
                'RMSE': f"{np.mean(temporal_rmse):.3f} ± {np.std(temporal_rmse):.3f}",
                'R2': f"{np.mean(temporal_r2):.3f} ± {np.std(temporal_r2):.3f}"
            })

    out_df = pd.DataFrame(all_results)
    out_df.to_csv('cv_results_summary.csv', index=False)
    print("\n" + out_df.to_markdown(index=False))

if __name__ == '__main__':
    main()
