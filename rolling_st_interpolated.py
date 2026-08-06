import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

WINDOW_SIZE   = 3
MIN_TRAIN_YRS = 5
REGION_FRAC   = 0.20
SEEDS         = [42, 43, 44]
MIN_TEST_OBS  = 10

IGNORE_COLS = ['harvest_year', 'latitude', 'longitude', 'harvest_area',
               'production', 'country_code', 'adm_id',
               'crop_name_x', 'crop_name_y'] 

DATASETS = [
    {
        'name':      'CY-Bench Maize (Europe)',
        'csv':       'cybench_full.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
        'anomalies': [2018]
    },
    {
        'name':      'CY-Bench Wheat (Europe)',
        'csv':       'cybench_wheat_full.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
        'anomalies': [2017]
    },
    {
        'name':      'CY-Bench Maize (Zambia)',
        'csv':       'cybench_zambia_maize.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
        'anomalies': [2012, 2014]
    },
]

MODELS = {
    'Ridge':         Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=1.0))]),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XGBoost':       XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1),
}

def load_and_interpolate(ds_cfg):
    df = pd.read_csv(ds_cfg['csv'])
    yc = ds_cfg['yield_col']
    df = df[df[yc] > 0].copy()
    numeric = df.select_dtypes(include='number').columns
    pred_cols = [c for c in numeric if c not in IGNORE_COLS and c != yc]
    df = df.dropna(subset=pred_cols)
    
    df['_year']  = df[ds_cfg['year_col']].astype(int)
    df['_group'] = df[ds_cfg['group_col']].astype(str)
    df['_yield_orig'] = df[yc].values
    
    # Sort by year so interpolation works correctly temporally
    df = df.sort_values(by=['_group', '_year']).reset_index(drop=True)
    
    # Set anomalies to NaN for yield only
    df['_yield_interp'] = df['_yield_orig'].copy()
    df.loc[df['_year'].isin(ds_cfg['anomalies']), '_yield_interp'] = np.nan
    
    # Linear interpolate per region, fallback to bfill/ffill if on edges
    df['_yield_interp'] = df.groupby('_group')['_yield_interp'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # Ensure no NaNs left
    assert not df['_yield_interp'].isna().any()
    
    X = df[pred_cols].values
    return df, X, pred_cols

def rolling_st(df, X, y_col, window_size, min_train_yrs, region_frac, seeds, models, min_obs):
    years          = df['_year'].values
    groups         = df['_group'].values
    y              = df[y_col].values
    unique_years   = np.sort(np.unique(years))
    unique_regions = np.unique(groups)
    results = []

    for wi, wy_start in enumerate(unique_years):
        test_years = unique_years[(unique_years >= wy_start) & (unique_years < wy_start + window_size)]
        if len(test_years) < window_size: continue
        train_years_avail = unique_years[unique_years < wy_start]
        if len(train_years_avail) < min_train_yrs: continue

        for seed in seeds:
            np.random.seed(seed)
            test_regions = np.random.choice(unique_regions, size=max(1, int(len(unique_regions) * region_frac)), replace=False)
            train_mask = (~np.isin(groups, test_regions)) & (~np.isin(years, test_years))
            test_mask  =  np.isin(groups, test_regions)  &  np.isin(years, test_years)
            tr_idx = np.where(train_mask)[0]
            te_idx = np.where(test_mask)[0]

            if len(te_idx) < min_obs or len(tr_idx) < 10: continue

            for mname, base_model in models.items():
                m = sklearn.base.clone(base_model)
                m.fit(X[tr_idx], y[tr_idx])
                preds = m.predict(X[te_idx])
                results.append({
                    'window_label': f"{wy_start}-{test_years[-1]}",
                    'model':        mname,
                    'seed':         seed,
                    'n_train':      len(tr_idx),
                    'n_test':       len(te_idx),
                    'r2':           r2_score(y[te_idx], preds),
                })
    return pd.DataFrame(results)

print("DATASET | WINDOW | MODEL | R2_ORIGINAL | R2_INTERPOLATED | N_TRAIN | N_TEST")

with open("interpolated_results.txt", "w") as f:
    f.write("DATASET | WINDOW | MODEL | R2_ORIGINAL | R2_INTERPOLATED | N_TRAIN_ORIG | N_TRAIN_INTERP | N_TEST_ORIG | N_TEST_INTERP\n")

for ds in DATASETS:
    df, X, pred_cols = load_and_interpolate(ds)
    
    # Run original
    res_orig = rolling_st(df, X, '_yield_orig', WINDOW_SIZE, MIN_TRAIN_YRS, REGION_FRAC, SEEDS, MODELS, MIN_TEST_OBS)
    summ_orig = res_orig.groupby(['window_label', 'model']).agg(r2_orig=('r2', 'mean'), n_tr_orig=('n_train', 'mean'), n_te_orig=('n_test', 'mean')).reset_index()
    
    # Run interpolated
    res_interp = rolling_st(df, X, '_yield_interp', WINDOW_SIZE, MIN_TRAIN_YRS, REGION_FRAC, SEEDS, MODELS, MIN_TEST_OBS)
    summ_interp = res_interp.groupby(['window_label', 'model']).agg(r2_interp=('r2', 'mean'), n_tr_interp=('n_train', 'mean'), n_te_interp=('n_test', 'mean')).reset_index()
    
    merged = pd.merge(summ_orig, summ_interp, on=['window_label', 'model'])
    
    with open("interpolated_results.txt", "a") as f:
        for _, row in merged.iterrows():
            line = f"{ds['name']} | {row['window_label']} | {row['model']} | {row['r2_orig']:.3f} | {row['r2_interp']:.3f} | {row['n_tr_orig']:.1f} | {row['n_tr_interp']:.1f} | {row['n_te_orig']:.1f} | {row['n_te_interp']:.1f}"
            print(line)
            f.write(line + "\n")
