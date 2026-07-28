import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import sklearn
import warnings
warnings.filterwarnings('ignore')

ANOMALOUS_YEAR_MAIZE = 2018   # known European maize anomaly
ANOMALOUS_YEAR_WHEAT = None   # no pre-identified anomaly for wheat

IGNORE_COLS = ['harvest_year', 'latitude', 'longitude', 'harvest_area',
               'production', 'crop_name_x', 'crop_name_y', 'country_code', 'adm_id']

def load_dataset(path, yield_col='yield'):
    df = pd.read_csv(path)
    df = df[df[yield_col] > 0]
    numeric_cols = df.select_dtypes(include=['number']).columns
    predictor_cols = [c for c in numeric_cols if c not in IGNORE_COLS and c != yield_col]
    df = df.dropna(subset=predictor_cols)
    X = df[predictor_cols].values
    y = df[yield_col].values
    return df, X, y

models = {
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XGBoost':       XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'Ridge':         Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
}

for dataset_name, csv_path, anomalous_year in [
    ('CY-Bench European Maize', 'cybench_full.csv',       ANOMALOUS_YEAR_MAIZE),
    ('CY-Bench European Wheat', 'cybench_wheat_full.csv', ANOMALOUS_YEAR_WHEAT),
]:
    print(f"\n{'='*65}")
    print(f"SPATIOTEMPORAL SPLIT DIAGNOSTIC — {dataset_name}")
    print(f"{'='*65}")

    df, X, y = load_dataset(csv_path)
    groups_spatial = df['adm_id'].values
    years          = df['harvest_year'].values
    unique_years   = np.sort(np.unique(years))
    unique_regions = np.unique(groups_spatial)

    n_test_years   = int(len(unique_years) * 0.2)
    held_out_years_pool = unique_years[-n_test_years:]

    print(f"\nYear range in data : {unique_years[0]} - {unique_years[-1]}  ({len(unique_years)} years)")
    print(f"Regions            : {len(unique_regions)}")
    print(f"20% tail years pool: {list(held_out_years_pool)}")
    if anomalous_year:
        in_pool = anomalous_year in held_out_years_pool
        print(f"Anomalous year {anomalous_year} is in the tail pool? {'YES <<' if in_pool else 'NO  (always in training)'}")

    print(f"\n{'Seed':>6}  {'Test years':^30}  {'Regions':>7}  {'Samples':>7}  {'Anomaly?':>12}  {'RF R2':>7}  {'XGB R2':>7}  {'Ridge R2':>7}")
    print("-" * 100)

    for seed in [42, 43, 44]:
        np.random.seed(seed)
        test_regions = np.random.choice(unique_regions,
                                        size=int(len(unique_regions) * 0.2),
                                        replace=False)
        test_years_actual = held_out_years_pool   # last 20% years (fixed)

        train_mask = (~np.isin(groups_spatial, test_regions)) & (~np.isin(years, test_years_actual))
        test_mask  =  np.isin(groups_spatial, test_regions)  &  np.isin(years, test_years_actual)

        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]

        held_years   = sorted(np.unique(years[test_idx]))
        n_regions_test = len(np.unique(groups_spatial[test_idx]))

        if anomalous_year:
            anomaly_tag = 'YES <<' if anomalous_year in held_years else 'NO'
        else:
            anomaly_tag = 'N/A'

        fold_r2s = {}
        for model_name, base_model in models.items():
            model = sklearn.base.clone(base_model)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            fold_r2s[model_name] = r2_score(y[test_idx], preds)

        print(f"{seed:>6}  {str(held_years):^30}  {n_regions_test:>7}  {len(test_idx):>7}  "
              f"{anomaly_tag:>12}  {fold_r2s['Random Forest']:>7.3f}  "
              f"{fold_r2s['XGBoost']:>7.3f}  {fold_r2s['Ridge']:>7.3f}")

print()
