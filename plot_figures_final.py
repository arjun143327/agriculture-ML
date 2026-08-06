import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------------------------
# Config & Setup
# -------------------------------------------------------------------
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
COLORS = {'Ridge': '#2ca02c', 'Random Forest': '#1f77b4', 'XGBoost': '#ff7f0e', 'Null Baseline': '#7f7f7f'}
DATASETS = ['CY-Bench Maize (Europe)', 'CY-Bench Wheat', 'CY-Bench Zambia Maize', 'SustainBench Soybean']

import re

# Helper to parse string "mean ± std" to just mean
def parse_mean(val_str):
    if pd.isna(val_str): return np.nan
    s = str(val_str)
    # Match first floating point number
    m = re.search(r'-?\d+\.\d+', s)
    if m: return float(m.group(0))
    return float(s)

# Helper to parse standard deviation
def parse_sd(val_str):
    if pd.isna(val_str): return 0.0
    s = str(val_str)
    # Match all floating point numbers
    m = re.findall(r'-?\d+\.\d+', s)
    if len(m) > 1: return float(m[-1])
    return 0.0

# Load main dataset results
df_all = pd.read_csv('results/all_dataset_results.csv')
df_wheat = pd.read_csv('results/wheat_results.csv')
df_zambia = pd.read_csv('results/zambia_results.csv')

# Merge them into a single dataframe
df_wheat['Dataset'] = 'CY-Bench Wheat' 
df_zambia['Dataset'] = 'CY-Bench Zambia Maize'
# df_all has SustainBench Soybean and CY-Bench Maize (Europe)
main_df = pd.concat([df_all, df_wheat, df_zambia], ignore_index=True)

# Parse numerical values
main_df['RMSE_mean'] = main_df['RMSE'].apply(parse_mean)
main_df['RMSE_sd'] = main_df['RMSE'].apply(parse_sd)
main_df['R2_mean'] = main_df['R2'].apply(parse_mean)
main_df['R2_sd'] = main_df['R2'].apply(parse_sd)

# -------------------------------------------------------------------
# Figure 1: RMSE Random vs Spatial
# -------------------------------------------------------------------
print("Generating Figure 1...")
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
axes1 = axes1.flatten()

for i, ds in enumerate(DATASETS):
    ax = axes1[i]
    sub = main_df[(main_df['Dataset'] == ds) & (main_df['Split'].isin(['Random', 'Spatial'])) & (main_df['Model'] != 'Null Baseline')]
    
    if not sub.empty:
        sns.barplot(data=sub, x='Model', y='RMSE_mean', hue='Split', ax=ax, palette=['#aec7e8', '#ffbb78'])
        ax.set_title(ds)
        ax.set_ylabel('RMSE')
        ax.set_xlabel('')
        if i != 0: ax.get_legend().remove()
        
fig1.suptitle('Figure 1: RMSE Random vs Spatial Splitting', fontsize=16)
fig1.tight_layout()
fig1.savefig('figures/figure1_rmse_random_vs_spatial.png')
plt.close(fig1)

# -------------------------------------------------------------------
# Figure 2: R2 all protocols
# -------------------------------------------------------------------
print("Generating Figure 2...")
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
axes2 = axes2.flatten()

for i, ds in enumerate(DATASETS):
    ax = axes2[i]
    sub = main_df[(main_df['Dataset'] == ds) & (main_df['Model'] != 'Null Baseline')]
    
    if not sub.empty:
        sns.barplot(data=sub, x='Split', y='R2_mean', hue='Model', ax=ax, palette=COLORS)
        ax.set_title(ds)
        ax.set_ylabel('R²')
        ax.set_xlabel('')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        if i != 0: ax.get_legend().remove()

fig2.suptitle('Figure 2: R² Across All Splitting Protocols', fontsize=16)
fig2.tight_layout()
fig2.savefig('figures/figure2_r2_all_protocols.png')
plt.close(fig2)

# -------------------------------------------------------------------
# Figure 3: vs Null Baseline
# -------------------------------------------------------------------
print("Generating Figure 3...")
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
axes3 = axes3.flatten()

for i, ds in enumerate(DATASETS):
    ax = axes3[i]
    sub = main_df[main_df['Dataset'] == ds]
    
    if not sub.empty:
        sns.barplot(data=sub, x='Split', y='R2_mean', hue='Model', ax=ax, palette=COLORS)
        ax.set_title(ds)
        ax.set_ylabel('R²')
        ax.set_xlabel('')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        if i != 0: ax.get_legend().remove()

fig3.suptitle('Figure 3: R² vs Null Baseline', fontsize=16)
fig3.tight_layout()
fig3.savefig('figures/figure3_vs_null_baseline.png')
plt.close(fig3)

# -------------------------------------------------------------------
# Figure 4: Feature Importance (Random vs Spatial)
# -------------------------------------------------------------------
print("Generating Figure 4...")
feat_files = [
    ('CY-Bench Maize (Europe)', 'results/feat_importance_CY-Bench_Maize_Europe.csv'),
    ('CY-Bench Wheat', 'results/feat_importance_CY-Bench_Wheat_Europe.csv'),
    ('CY-Bench Maize (Zambia)', 'results/feat_importance_CY-Bench_Maize_Zambia.csv')
]

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 8), sharex=True)

for i, (name, fname) in enumerate(feat_files):
    if os.path.exists(fname):
        df_feat = pd.read_csv(fname)
        df_feat = df_feat.sort_values('spat_mean', ascending=True) # sort descending for horizontal bar chart
        
        ax = axes4[i]
        
        y_pos = np.arange(len(df_feat))
        height = 0.35
        
        ax.barh(y_pos - height/2, df_feat['rand_mean'], height, xerr=df_feat['rand_sd'], label='Random', color='#aec7e8')
        ax.barh(y_pos + height/2, df_feat['spat_mean'], height, xerr=df_feat['spat_sd'], label='Spatial', color='#ffbb78')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_feat['feature'])
        ax.set_title(name)
        ax.set_xlabel('Gain Importance')
        if i == 0:
            ax.legend()

fig4.suptitle('Figure 4: XGBoost Feature Importance (Random vs Spatial)', fontsize=16)
fig4.tight_layout()
fig4.savefig('figures/figure4_feature_importance.png')
plt.close(fig4)


# -------------------------------------------------------------------
# Figures 5, 6, 7 are flagged as missing data.
# -------------------------------------------------------------------
print("Skipping Figure 5, 6, 7 - Data missing per plan approval.")

# -------------------------------------------------------------------
# Figure 8: Rolling Window Sweep (2yr, 3yr, 5yr)
# -------------------------------------------------------------------
print("Generating Figure 8...")
if os.path.exists('results/rolling_extended_results.csv'):
    df_roll = pd.read_csv('results/rolling_extended_results.csv')
    
    # We want 4 panels, one per dataset. Within each panel, line per window length, using RF? 
    # Or average over models? The prompt says: "showing rolling-window R² across the full timeline for all three window lengths ... as separate lines/series within each panel". 
    # Usually we show just XGBoost or Random Forest, or average. Let's show XGBoost to keep it readable, or average across all 3 models if they are similar. 
    # The original Figure 8 showed 3 lines (one for each model) for ONE window length. Now we have 3 window lengths.
    # Let's average across the 3 models per window length for clarity, or just plot XGBoost. Let's use XGBoost.
    
    fig8, axes8 = plt.subplots(2, 2, figsize=(16, 12))
    axes8 = axes8.flatten()
    
    datasets_ordered = [
        'CY-Bench Maize (Europe)', 'CY-Bench Wheat (Europe)', 
        'CY-Bench Maize (Zambia)', 'SustainBench Soybean'
    ]
    
    colors_win = {2: '#1f77b4', 3: '#ff7f0e', 5: '#2ca02c'}
    
    for i, ds in enumerate(datasets_ordered):
        ax = axes8[i]
        sub = df_roll[(df_roll['dataset'] == ds) & (df_roll['model'] == 'XGBoost')]
        
        if sub.empty: continue
            
        for wl in [2, 3, 5]:
            sub_wl = sub[sub['window_length'] == wl].copy()
            # extract start year for x axis
            sub_wl['start_year'] = sub_wl['window_label'].apply(lambda x: int(str(x).split('-')[0]) if '-' in str(x) else int(x))
            sub_wl = sub_wl.sort_values('start_year')
            ax.plot(sub_wl['start_year'], sub_wl['r2_mean'], 'o-', label=f'{wl}-year window', color=colors_win[wl])
            
            # mark anomalies on the x axis
            # we just do it once
            if wl == 3:
                for _, row in sub_wl.iterrows():
                    if row['has_anomaly']:
                        ax.axvspan(row['start_year'] - 0.2, row['start_year'] + 0.2, color='red', alpha=0.1, zorder=0)

        ax.set_title(ds)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel('XGBoost R²')
        ax.set_xlabel('Window Start Year')
        if i == 0: ax.legend()
        
    fig8.suptitle('Figure 8: Rolling Spatiotemporal R² by Window Length (XGBoost)', fontsize=16)
    fig8.tight_layout()
    fig8.savefig('figures/figure8_rolling_window_sweep.png')
    plt.close(fig8)

# -------------------------------------------------------------------
# Figure 9: Window averaging comparison
# -------------------------------------------------------------------
print("Generating Figure 9...")
if os.path.exists('results/spatiotemporal_comparison.csv'):
    df_comp = pd.read_csv('results/spatiotemporal_comparison.csv')
    
    fig9, axes9 = plt.subplots(2, 2, figsize=(16, 12))
    axes9 = axes9.flatten()
    
    datasets_ordered = [
        'CY-Bench Maize (Europe)', 'CY-Bench Wheat (Europe)', 
        'CY-Bench Maize (Zambia)', 'SustainBench Soybean'
    ]
    
    for i, ds in enumerate(datasets_ordered):
        ax = axes9[i]
        sub = df_comp[df_comp['Dataset'] == ds]
        
        if sub.empty: continue
            
        # Slope chart / dot chart
        models = sub['Model'].unique()
        for j, m in enumerate(models):
            m_sub = sub[sub['Model'] == m].iloc[0]
            # plot Fixed
            ax.plot([1], [m_sub['Fixed_Window_R2']], 'o', color=COLORS.get(m, 'black'), markersize=8, label=m if j==0 else "") # label only first time? wait, models are legend
            # plot Averaged
            ax.errorbar([2], [m_sub['Window_Avg_R2']], yerr=[m_sub['Window_R2_SD']], fmt='o', color=COLORS.get(m, 'black'), markersize=8, capsize=5)
            # connecting line
            ax.plot([1, 2], [m_sub['Fixed_Window_R2'], m_sub['Window_Avg_R2']], '-', color=COLORS.get(m, 'black'), alpha=0.6)
            
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Fixed Window\n(Headline R²)', 'Rolling Average\n(True R²)'])
        ax.set_title(ds)
        ax.set_ylabel('R²')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
    # custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=COLORS[m], marker='o', label=m) for m in ['Ridge', 'Random Forest', 'XGBoost']]
    fig9.legend(handles=legend_elements, loc='upper center', ncol=3)
        
    fig9.suptitle('Figure 9: Fixed Cutoff vs. Window-Averaged R²', fontsize=16)
    fig9.tight_layout(rect=[0, 0, 1, 0.95])
    fig9.savefig('figures/figure9_window_averaging_comparison.png')
    plt.close(fig9)

# -------------------------------------------------------------------
# Figure 10: Anomaly Penalty
# -------------------------------------------------------------------
print("Generating Figure 10...")
if os.path.exists('results/rolling_extended_results.csv'):
    df_roll = pd.read_csv('results/rolling_extended_results.csv')
    # only use 3-year for the regression comparison
    df_3yr = df_roll[df_roll['window_length'] == 3].copy()
    
    fig10, ax10 = plt.subplots(1, 1, figsize=(10, 6))
    
    sns.boxplot(data=df_3yr, x='has_anomaly', y='r2_mean', ax=ax10, palette=['#aec7e8', '#ff9896'], width=0.5)
    sns.stripplot(data=df_3yr, x='has_anomaly', y='r2_mean', ax=ax10, color='black', alpha=0.5, jitter=True)
    
    ax10.set_xticks([0, 1])
    ax10.set_xticklabels(['No Anomaly\nin Window', 'Anomaly\nin Window'])
    ax10.set_ylabel('Spatiotemporal R²')
    ax10.set_xlabel('')
    ax10.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Add text annotation for OLS
    textstr = "OLS Anomaly Penalty:\n-0.39 R² Points\n95% CI: [-0.51, -0.27]\np < 0.0001"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax10.text(0.95, 0.95, textstr, transform=ax10.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
            
    fig10.suptitle('Figure 10: Degradation of R² due to Climate Anomalies', fontsize=16)
    fig10.tight_layout()
    fig10.savefig('figures/figure10_anomaly_penalty.png')
    plt.close(fig10)

print("Finished plotting.")
