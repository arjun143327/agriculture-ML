import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv("cybench_zambia_maize.csv")
    df = df[df['yield'] > 0]
    ignore_cols = ['harvest_year', 'latitude', 'longitude', 'harvest_area', 'production', 'crop_name_x', 'crop_name_y', 'country_code', 'adm_id']
    numeric_cols = df.select_dtypes(include=['number']).columns
    predictor_cols = [c for c in numeric_cols if c not in ignore_cols and c != 'yield']
    df = df.dropna(subset=predictor_cols)
    X = df[predictor_cols].values
    y = df['yield'].values
    return df, X, y

def main():
    df, X, y = load_data()
    groups_spatial = df['adm_id'].values
    years = df['harvest_year'].values
    unique_years = np.sort(np.unique(years))
    
    models = {
        'Null Baseline': 'null',
        'Ridge': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    all_results = []
    temporal_yearly_breakdown = {}
    
    for model_name, base_model in models.items():
        print(f"Evaluating {model_name}...")
        temporal_yearly_breakdown[model_name] = []
        
        def fit_predict(train_idx, test_idx):
            if model_name == 'Null Baseline':
                pred = np.full(len(test_idx), np.mean(y[train_idx]))
                return pred
            model = sklearn.base.clone(base_model)
            model.fit(X[train_idx], y[train_idx])
            return model.predict(X[test_idx])

        # --- RANDOM SPLIT (3 seeds) ---
        rand_rmse = []
        rand_r2 = []
        for seed in [42, 43, 44]:
            train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
            preds = fit_predict(train_idx, test_idx)
            rand_rmse.append(root_mean_squared_error(y[test_idx], preds))
            rand_r2.append(r2_score(y[test_idx], preds))
            
        all_results.append({
            'Dataset': 'CY-Bench Zambia Maize',
            'Model': model_name,
            'Split': 'Random',
            'RMSE': f"{np.mean(rand_rmse):.3f} ± {np.std(rand_rmse):.3f}",
            'R2': f"{np.mean(rand_r2):.3f} ± {np.std(rand_r2):.3f}"
        })
        
        # --- SPATIAL SPLIT (5-Fold GKF) ---
        gkf = GroupKFold(n_splits=5)
        spatial_rmse = []
        spatial_r2 = []
        for train_idx, test_idx in gkf.split(X, y, groups=groups_spatial):
            preds = fit_predict(train_idx, test_idx)
            spatial_rmse.append(root_mean_squared_error(y[test_idx], preds))
            spatial_r2.append(r2_score(y[test_idx], preds))
            
        all_results.append({
            'Dataset': 'CY-Bench Zambia Maize',
            'Model': model_name,
            'Split': 'Spatial',
            'RMSE': f"{np.mean(spatial_rmse):.3f} ± {np.std(spatial_rmse):.3f}",
            'R2': f"{np.mean(spatial_r2):.3f} ± {np.std(spatial_r2):.3f}"
        })
        
        # --- TEMPORAL SPLIT (Walk-Forward) ---
        temporal_rmse = []
        temporal_r2 = []
        for i in range(4, len(unique_years) - 1):
            train_years = unique_years[:i+1]
            test_year = unique_years[i+1]
            train_mask = np.isin(years, train_years)
            test_mask = (years == test_year)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(test_idx) == 0 or len(train_idx) == 0: continue
            
            # Need at least a few samples to compute valid R2
            if len(test_idx) < 5: continue
            
            preds = fit_predict(train_idx, test_idx)
            test_rmse = root_mean_squared_error(y[test_idx], preds)
            test_r2 = r2_score(y[test_idx], preds)
            
            temporal_rmse.append(test_rmse)
            temporal_r2.append(test_r2)
            
            temporal_yearly_breakdown[model_name].append({
                'Year': test_year,
                'RMSE': test_rmse,
                'R2': test_r2,
                'Samples': len(test_idx)
            })
            
        if len(temporal_rmse) > 0:
            all_results.append({
                'Dataset': 'CY-Bench Zambia Maize',
                'Model': model_name,
                'Split': 'Temporal',
                'RMSE': f"{np.mean(temporal_rmse):.3f} ± {np.std(temporal_rmse):.3f}",
                'R2': f"{np.mean(temporal_r2):.3f} ± {np.std(temporal_r2):.3f}"
            })
            
        # --- SPATIOTEMPORAL SPLIT (3 seeds) ---
        st_rmse = []
        st_r2 = []
        for seed in [42, 43, 44]:
            np.random.seed(seed)
            unique_regions = np.unique(groups_spatial)
            test_regions = np.random.choice(unique_regions, size=int(len(unique_regions)*0.2), replace=False)
            test_years = unique_years[-int(len(unique_years)*0.2):]
            
            train_mask = (~np.isin(groups_spatial, test_regions)) & (~np.isin(years, test_years))
            test_mask = np.isin(groups_spatial, test_regions) & np.isin(years, test_years)
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(test_idx) == 0 or len(train_idx) == 0: continue
            
            preds = fit_predict(train_idx, test_idx)
            st_rmse.append(root_mean_squared_error(y[test_idx], preds))
            st_r2.append(r2_score(y[test_idx], preds))
            
        if len(st_rmse) > 0:
            all_results.append({
                'Dataset': 'CY-Bench Zambia Maize',
                'Model': model_name,
                'Split': 'Spatiotemporal',
                'RMSE': f"{np.mean(st_rmse):.3f} ± {np.std(st_rmse):.3f}",
                'R2': f"{np.mean(st_r2):.3f} ± {np.std(st_r2):.3f}"
            })

    out_df = pd.DataFrame(all_results)
    out_df.to_csv('zambia_results.csv', index=False)
    
    print("\n" + "="*50)
    print("TEMPORAL WALK-FORWARD YEARLY BREAKDOWN (Random Forest)")
    print("="*50)
    rf_breakdown = pd.DataFrame(temporal_yearly_breakdown['Random Forest'])
    print(rf_breakdown.to_string(index=False))
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY TABLE")
    print("="*50)
    print(out_df.to_markdown(index=False))

if __name__ == '__main__':
    main()
