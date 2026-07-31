"""
Rolling spatiotemporal evaluation.

For each dataset, a 3-year temporal window is slid across the full timeline
(requiring at least 5 years of prior training data).  Each window position is
combined with a 20% random-region holdout (averaged over seeds 42/43/44).

Output: one CSV per dataset + a combined figure showing R² vs window position.
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
WINDOW_SIZE   = 3        # years in each temporal test window
MIN_TRAIN_YRS = 5        # minimum years before window starts
REGION_FRAC   = 0.20     # fraction of regions held out spatially
SEEDS         = [42, 43, 44]
MIN_TEST_OBS  = 10       # skip window-seed combos with too few test points

IGNORE_COLS = ['harvest_year', 'latitude', 'longitude', 'harvest_area',
               'production', 'country_code', 'adm_id',
               'crop_name_x', 'crop_name_y',  # European maize/wheat
               'year', 'key', 'region1', 'region2']  # SustainBench

# Known anomalous years per dataset (for annotation)
ANOMALIES = {
    'CY-Bench Maize (Europe)':  [2018],
    'CY-Bench Wheat (Europe)':  [],
    'CY-Bench Maize (Zambia)':  [2012, 2014],
    'SustainBench Soybean':     [],
}

DATASETS = [
    {
        'name':      'CY-Bench Maize (Europe)',
        'csv':       'cybench_full.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
    },
    {
        'name':      'CY-Bench Wheat (Europe)',
        'csv':       'cybench_wheat_full.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
    },
    {
        'name':      'CY-Bench Maize (Zambia)',
        'csv':       'cybench_zambia_maize.csv',
        'yield_col': 'yield',
        'year_col':  'harvest_year',
        'group_col': 'adm_id',
    },
    {
        'name':      'SustainBench Soybean',
        'csv':       'sustainbench_full.csv',
        'yield_col': 'yield_t_ha',
        'year_col':  'year',
        'group_col': 'region1',
    },
]

MODELS = {
    'Ridge':         Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=1.0))]),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XGBoost':       XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1),
}

C = {'Ridge': '#2ca02c', 'Random Forest': '#1f77b4', 'XGBoost': '#ff7f0e'}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load(ds_cfg):
    df = pd.read_csv(ds_cfg['csv'])
    yc = ds_cfg['yield_col']
    df = df[df[yc] > 0].copy()
    numeric = df.select_dtypes(include='number').columns
    pred_cols = [c for c in numeric if c not in IGNORE_COLS and c != yc]
    df = df.dropna(subset=pred_cols)
    # harmonise column names
    df['_year']  = df[ds_cfg['year_col']].astype(int)
    df['_group'] = df[ds_cfg['group_col']].astype(str)
    df['_yield'] = df[yc].values
    X = df[pred_cols].values
    return df, X, df['_yield'].values, pred_cols


def rolling_st(df, X, y, window_size, min_train_yrs, region_frac, seeds,
               models, min_obs):
    """
    Returns a list of dicts, one per (window_start, model_name) combination.
    """
    years          = df['_year'].values
    groups         = df['_group'].values
    unique_years   = np.sort(np.unique(years))
    unique_regions = np.unique(groups)

    results = []

    # Slide window: test = [wy, wy+1, ..., wy+window_size-1]
    # Require at least min_train_yrs before window start
    for wi, wy_start in enumerate(unique_years):
        test_years = unique_years[
            (unique_years >= wy_start) & (unique_years < wy_start + window_size)
        ]
        if len(test_years) < window_size:
            continue                              # can't fill window at tail
        train_years_avail = unique_years[unique_years < wy_start]
        if len(train_years_avail) < min_train_yrs:
            continue                              # not enough history

        for seed in seeds:
            np.random.seed(seed)
            test_regions = np.random.choice(
                unique_regions,
                size=max(1, int(len(unique_regions) * region_frac)),
                replace=False
            )

            train_mask = (~np.isin(groups, test_regions)) & (~np.isin(years, test_years))
            test_mask  =  np.isin(groups, test_regions)  &  np.isin(years, test_years)

            tr_idx = np.where(train_mask)[0]
            te_idx = np.where(test_mask)[0]

            if len(te_idx) < min_obs or len(tr_idx) < 10:
                continue

            for mname, base_model in models.items():
                m = sklearn.base.clone(base_model)
                m.fit(X[tr_idx], y[tr_idx])
                preds = m.predict(X[te_idx])
                results.append({
                    'window_start': int(wy_start),
                    'window_end':   int(test_years[-1]),
                    'window_label': f"{wy_start}-{test_years[-1]}",
                    'model':        mname,
                    'seed':         seed,
                    'n_train':      len(tr_idx),
                    'n_test':       len(te_idx),
                    'rmse':         root_mean_squared_error(y[te_idx], preds),
                    'r2':           r2_score(y[te_idx], preds),
                })

    return pd.DataFrame(results)


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
all_summaries = {}

for ds in DATASETS:
    name = ds['name']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    df, X, y, preds = load(ds)
    print(f"  Rows={len(df)}  Regions={df['_group'].nunique()}  "
          f"Years={df['_year'].min()}-{df['_year'].max()}")

    raw = rolling_st(df, X, y, WINDOW_SIZE, MIN_TRAIN_YRS,
                     REGION_FRAC, SEEDS, MODELS, MIN_TEST_OBS)

    # Average over seeds for each (window, model)
    summary = (raw.groupby(['window_label', 'window_start', 'model'])
                   .agg(r2_mean=('r2', 'mean'),
                        r2_std=('r2', 'std'),
                        rmse_mean=('rmse', 'mean'),
                        n_test=('n_test', 'mean'))
                   .reset_index()
                   .sort_values(['window_start', 'model']))

    out_csv = f"rolling_st_{name.replace(' ','_').replace('(','').replace(')','')}.csv"
    summary.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")

    # Print table
    pivot = summary.pivot_table(index='window_label', columns='model',
                                values='r2_mean').round(3)
    print(pivot.to_string())

    all_summaries[name] = summary


# -------------------------------------------------------------------
# Figure: rolling spatiotemporal R² for all 4 datasets
# -------------------------------------------------------------------
plt.rcParams.update({'font.size': 10})

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

for ax, (name, summary) in zip(axes, all_summaries.items()):
    anomalous = ANOMALIES.get(name, [])
    windows   = sorted(summary['window_start'].unique())

    for mname, color in C.items():
        sub = summary[summary['model'] == mname].sort_values('window_start')
        ax.plot(sub['window_start'], sub['r2_mean'], 'o-',
                color=color, label=mname, linewidth=1.6, markersize=5)
        ax.fill_between(sub['window_start'],
                        sub['r2_mean'] - sub['r2_std'],
                        sub['r2_mean'] + sub['r2_std'],
                        alpha=0.15, color=color)

    # Shade windows that contain an anomalous year
    for wy in windows:
        test_end = wy + WINDOW_SIZE - 1
        if any(wy <= ay <= test_end for ay in anomalous):
            ax.axvspan(wy - 0.4, wy + 0.4, color='red', alpha=0.12, zorder=0)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(name)
    ax.set_xlabel(f'Window start year  (window = {WINDOW_SIZE} yrs)')
    ax.set_ylabel('R² (mean over seeds)')

    if anomalous:
        ax.legend(loc='lower left', fontsize=8,
                  title=f'Red shade = anomaly yr(s) {anomalous} in window')
    else:
        ax.legend(loc='lower left', fontsize=8)

fig.suptitle('Rolling Spatiotemporal R²  (3-yr window × 20% region holdout)',
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig('figures/figure_rolling_spatiotemporal.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved figures/figure_rolling_spatiotemporal.png")
