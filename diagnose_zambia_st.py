import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv("cybench_zambia_maize.csv")
    df = df[df['yield'] > 0]
    ignore_cols = ['harvest_year', 'latitude', 'longitude', 'harvest_area', 'production',
                   'crop_name_x', 'crop_name_y', 'country_code', 'adm_id']
    numeric_cols = df.select_dtypes(include=['number']).columns
    predictor_cols = [c for c in numeric_cols if c not in ignore_cols and c != 'yield']
    df = df.dropna(subset=predictor_cols)
    X = df[predictor_cols].values
    y = df['yield'].values
    return df, X, y, predictor_cols

df, X, y, predictor_cols = load_data()
groups_spatial = df['adm_id'].values
years = df['harvest_year'].values
unique_years = np.sort(np.unique(years))
unique_regions = np.unique(groups_spatial)

anomalous_years = [2012, 2014]

models = {
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
}

print("=" * 65)
print("SPATIOTEMPORAL SPLIT DIAGNOSTIC — ZAMBIA MAIZE")
print("=" * 65)

for model_name, base_model in models.items():
    print(f"\n--- Model: {model_name} ---")
    
    for seed in [42, 43, 44]:
        np.random.seed(seed)
        
        # Replicate exact split logic from run_cv_zambia.py
        test_regions = np.random.choice(
            unique_regions,
            size=int(len(unique_regions) * 0.2),
            replace=False
        )
        test_years = unique_years[-int(len(unique_years) * 0.2):]
        
        train_mask = (~np.isin(groups_spatial, test_regions)) & (~np.isin(years, test_years))
        test_mask  = np.isin(groups_spatial, test_regions) & np.isin(years, test_years)
        
        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]
        
        if len(test_idx) == 0 or len(train_idx) == 0:
            print(f"  Seed {seed}: SKIPPED (empty split)")
            continue
        
        # Fit and predict
        model = sklearn.base.clone(base_model)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        fold_rmse = root_mean_squared_error(y[test_idx], preds)
        fold_r2   = r2_score(y[test_idx], preds)
        
        # Composition report
        held_years   = sorted(np.unique(years[test_idx]))
        held_regions = sorted(np.unique(groups_spatial[test_idx]))
        anomaly_present = [yr for yr in anomalous_years if yr in held_years]
        
        print(f"\n  Seed {seed}:")
        print(f"    Test years  ({len(held_years)}): {held_years}")
        print(f"    Test regions({len(held_regions)}): {held_regions[:5]}{'...' if len(held_regions)>5 else ''} [total {len(held_regions)}]")
        print(f"    Anomalous years 2012/2014 in test? {anomaly_present if anomaly_present else 'NO'}")
        print(f"    Test samples: {len(test_idx)}")
        print(f"    RMSE: {fold_rmse:.3f}  |  R²: {fold_r2:.3f}")

# Also print per-seed summary across all models in a table
print("\n" + "=" * 65)
print("PER-FOLD R² SUMMARY TABLE")
print("=" * 65)
header = f"{'Seed':>6}  {'RF R²':>8}  {'XGB R²':>8}  {'Ridge R²':>8}  {'Anomaly in Test?':>20}"
print(header)
print("-" * 65)

for seed in [42, 43, 44]:
    np.random.seed(seed)
    test_regions = np.random.choice(unique_regions, size=int(len(unique_regions)*0.2), replace=False)
    test_years   = unique_years[-int(len(unique_years)*0.2):]
    train_mask = (~np.isin(groups_spatial, test_regions)) & (~np.isin(years, test_years))
    test_mask  = np.isin(groups_spatial, test_regions) & np.isin(years, test_years)
    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(test_mask)[0]
    
    held_years = sorted(np.unique(years[test_idx]))
    anomaly_present = any(yr in held_years for yr in anomalous_years)
    
    fold_r2s = {}
    for model_name, base_model in models.items():
        model = sklearn.base.clone(base_model)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        fold_r2s[model_name] = r2_score(y[test_idx], preds)
    
    print(f"{seed:>6}  {fold_r2s['Random Forest']:>8.3f}  {fold_r2s['XGBoost']:>8.3f}  {fold_r2s['Ridge']:>8.3f}  {'YES' if anomaly_present else 'NO':>20}")
