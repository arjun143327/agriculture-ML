import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('sustainbench_full.csv')
df = df[df['yield_t_ha'] > 0].copy()
df['_year']  = df['year'].astype(int)
df['_group'] = df['region1'].astype(str)

IGNORE = ['harvest_year','latitude','longitude','harvest_area','production',
          'country_code','adm_id','crop_name_x','crop_name_y',
          'year','key','region1','region2']
numeric   = df.select_dtypes(include='number').columns
pred_cols = [c for c in numeric if c not in IGNORE and c != 'yield_t_ha']
df        = df.dropna(subset=pred_cols)
X  = df[pred_cols].values
y  = df['yield_t_ha'].values

unique_years   = np.sort(np.unique(df['_year'].values))
unique_regions = np.unique(df['_group'].values)
years  = df['_year'].values
groups = df['_group'].values

print("All years in SustainBench:", list(unique_years))
print(f"Total rows: {len(df)}, regions: {len(unique_regions)}")
print()

print("Rows per year:")
for yr in unique_years:
    print(f"  {yr}: {(years == yr).sum()} rows")
print()

WINDOW_SIZE   = 3
MIN_TRAIN_YRS = 5
SEEDS = [42, 43, 44]
REGION_FRAC = 0.20
model_proto = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

header = ("Window      | Test years         | Train years                   | "
          "2012 in train? | Train rows | Test rows |  RF R2")
print(header)
print("-" * len(header))

for wy_start in unique_years:
    test_years = unique_years[
        (unique_years >= wy_start) & (unique_years < wy_start + WINDOW_SIZE)
    ]
    if len(test_years) < WINDOW_SIZE:
        continue
    train_years_avail = unique_years[unique_years < wy_start]
    if len(train_years_avail) < MIN_TRAIN_YRS:
        continue

    seed_r2s, tr_sizes, te_sizes = [], [], []
    for seed in SEEDS:
        np.random.seed(seed)
        test_regions = np.random.choice(unique_regions,
                                        size=max(1, int(len(unique_regions)*REGION_FRAC)),
                                        replace=False)
        train_mask = (~np.isin(groups, test_regions)) & (~np.isin(years, test_years))
        test_mask  =  np.isin(groups, test_regions)  &  np.isin(years, test_years)
        tr = np.where(train_mask)[0]
        te = np.where(test_mask)[0]
        if len(te) < 10: continue
        m = sklearn.base.clone(model_proto)
        m.fit(X[tr], y[tr])
        seed_r2s.append(r2_score(y[te], m.predict(X[te])))
        tr_sizes.append(len(tr))
        te_sizes.append(len(te))

    if not seed_r2s: continue

    label       = f"{wy_start}-{test_years[-1]}"
    train_yr_list = list(train_years_avail)
    drought_in_train = 2012 in train_yr_list

    print(f"{label:<12}| {str(list(test_years)):<19}| {str(train_yr_list):<30}| "
          f"{str(drought_in_train):<15}| {int(np.mean(tr_sizes)):>10} | "
          f"{int(np.mean(te_sizes)):>9} | {np.mean(seed_r2s):>7.3f}")
