"""
Final publication figures — all 4 datasets:
  1. CY-Bench European Maize
  2. CY-Bench European Wheat
  3. CY-Bench Zambia Maize
  4. SustainBench Soybean

No titles baked in (captions go in the paper).
Figures saved to figures/ at 300 dpi.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# ------------------------------------------------------------------
# Color palette (consistent across all figures)
# ------------------------------------------------------------------
C_NULL = '#888888'   # gray   – Null Baseline
C_RIDGE = '#2ca02c'  # green  – Ridge
C_RF   = '#1f77b4'   # blue   – Random Forest
C_XGB  = '#ff7f0e'   # orange – XGBoost

SPLITS = ['Random', 'Spatial', 'Temporal', 'Spatiotemp.']
x = np.arange(len(SPLITS))
W = 0.20   # bar width for 4-model plots

# ------------------------------------------------------------------
# Data — R² mean ± SD for every dataset / model / split
# ------------------------------------------------------------------
# CY-Bench European Maize
maize = {
    'Null Baseline': {'r2': [-0.010, -0.012, -0.560,  0.050], 'sd': [0.005, 0.015, 0.861, 0.300]},
    'Ridge':         {'r2': [ 0.517,  0.270,  0.560, -0.125], 'sd': [0.001, 0.050, 0.010, 0.050]},
    'Random Forest': {'r2': [ 0.728,  0.536,  0.293,  0.300], 'sd': [0.041, 0.087, 0.861, 0.116]},
    'XGBoost':       {'r2': [ 0.755,  0.465,  0.280,  0.196], 'sd': [0.033, 0.132, 0.796, 0.077]},
}

# CY-Bench European Wheat
wheat = {
    'Null Baseline': {'r2': [-0.007, -0.019, -0.394, -0.484], 'sd': [0.009, 0.025, 0.410, 0.162]},
    'Ridge':         {'r2': [ 0.486,  0.254,  0.116, -0.431], 'sd': [0.045, 0.141, 0.568, 0.315]},
    'Random Forest': {'r2': [ 0.671,  0.307,  0.267, -0.052], 'sd': [0.020, 0.269, 0.376, 0.020]},
    'XGBoost':       {'r2': [ 0.676,  0.371,  0.259, -0.175], 'sd': [0.019, 0.159, 0.385, 0.088]},
}

# CY-Bench Zambia Maize
zambia = {
    'Null Baseline': {'r2': [-0.005, -0.021, -0.307, -0.105], 'sd': [0.004, 0.023, 0.252, 0.073]},
    'Ridge':         {'r2': [ 0.356,  0.318,  0.180,  0.424], 'sd': [0.047, 0.094, 0.318, 0.096]},
    'Random Forest': {'r2': [ 0.466,  0.303,  0.264,  0.357], 'sd': [0.049, 0.083, 0.302, 0.186]},
    'XGBoost':       {'r2': [ 0.424,  0.238,  0.165,  0.282], 'sd': [0.024, 0.077, 0.311, 0.233]},
}

# SustainBench Soybean
sustain = {
    'Null Baseline': {'r2': [-0.001, -0.001, -0.384, -0.050], 'sd': [0.003, 0.008, 0.497, 0.200]},
    'Ridge':         {'r2': [-0.014,  0.248, -0.241,  0.158], 'sd': [0.008, 0.050, 0.200, 0.100]},
    'Random Forest': {'r2': [ 0.401,  0.391, -0.060,  0.249], 'sd': [0.007, 0.029, 0.497, 0.109]},
    'XGBoost':       {'r2': [ 0.371,  0.355, -0.071,  0.182], 'sd': [0.016, 0.029, 0.496, 0.182]},
}

ALL_DATASETS = [
    ('CY-Bench\nEuropean Maize', maize),
    ('CY-Bench\nEuropean Wheat', wheat),
    ('CY-Bench\nZambia Maize',   zambia),
    ('SustainBench\nSoybean',    sustain),
]

def bar_offsets(n):
    """Return centered offsets for n bars."""
    return np.linspace(-(n-1)/2, (n-1)/2, n) * W

# ------------------------------------------------------------------
# FIGURE 1 — R² across all 4 splits × 4 datasets (RF + XGBoost only)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
offsets = bar_offsets(2)

for ax, (title, ds) in zip(axes, ALL_DATASETS):
    for i, (model, color, off) in enumerate([
        ('Random Forest', C_RF,  offsets[0]),
        ('XGBoost',       C_XGB, offsets[1]),
    ]):
        means = ds[model]['r2']
        stds  = ds[model]['sd']
        bars = ax.bar(x + off, means, W, yerr=stds, color=color,
                      capsize=4, label=model, alpha=0.88)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)
    ax.set_xlabel('Split protocol')
    ax.set_ylabel('R²')
    ax.set_title(title)
    ax.set_ylim(-1.4, 1.0)

axes[0].legend(loc='lower left', fontsize=9)
fig.tight_layout()
fig.savefig('figures/figure1_r2_all_datasets.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure1_r2_all_datasets.png")

# ------------------------------------------------------------------
# FIGURE 2 — Null-baseline comparison (Spatial + Temporal)
#            For all 4 datasets; 3 models shown (Null, RF, XGB)
# ------------------------------------------------------------------
SPLIT_IDXS = [1, 2]  # Spatial=index1, Temporal=index2
SPLIT_LABELS = ['Spatial', 'Temporal']
xb = np.arange(len(SPLIT_LABELS))
offsets3 = bar_offsets(3)

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)

for ax, (title, ds) in zip(axes, ALL_DATASETS):
    for model, color, off in [
        ('Null Baseline', C_NULL, offsets3[0]),
        ('Random Forest', C_RF,   offsets3[1]),
        ('XGBoost',       C_XGB,  offsets3[2]),
    ]:
        means = [ds[model]['r2'][i] for i in SPLIT_IDXS]
        stds  = [ds[model]['sd'][i] for i in SPLIT_IDXS]
        ax.bar(xb + off, means, W, yerr=stds, color=color,
               capsize=4, label=model, alpha=0.88)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(xb)
    ax.set_xticklabels(SPLIT_LABELS)
    ax.set_xlabel('Split protocol')
    ax.set_ylabel('R²')
    ax.set_title(title)
    ax.set_ylim(-1.4, 0.9)

axes[0].legend(loc='lower left', fontsize=9)
fig.tight_layout()
fig.savefig('figures/figure2_null_baseline_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure2_null_baseline_comparison.png")

# ------------------------------------------------------------------
# FIGURE 3 — RMSE leakage gap: Random vs Spatial (bar chart, per model)
#            All 4 datasets side by side
# ------------------------------------------------------------------
RMSE = {
    'CY-Bench\nEuropean Maize': {
        'Ridge':         {'Random': 1.668, 'Spatial': 1.915},
        'Random Forest': {'Random': 1.249, 'Spatial': 1.686},
        'XGBoost':       {'Random': 1.185, 'Spatial': 1.701},
    },
    'CY-Bench\nEuropean Wheat': {
        'Ridge':         {'Random': 0.886, 'Spatial': 1.017},
        'Random Forest': {'Random': 0.710, 'Spatial': 0.977},
        'XGBoost':       {'Random': 0.704, 'Spatial': 0.940},
    },
    'CY-Bench\nZambia Maize': {
        'Ridge':         {'Random': 0.753, 'Spatial': 0.760},
        'Random Forest': {'Random': 0.685, 'Spatial': 0.770},
        'XGBoost':       {'Random': 0.711, 'Spatial': 0.806},
    },
    'SustainBench\nSoybean': {
        'Ridge':         {'Random': 0.680, 'Spatial': 0.604},
        'Random Forest': {'Random': 0.535, 'Spatial': 0.546},
        'XGBoost':       {'Random': 0.546, 'Spatial': 0.557},
    },
}

models3 = ['Ridge', 'Random Forest', 'XGBoost']
colors3  = [C_RIDGE, C_RF, C_XGB]
x3 = np.arange(len(models3))
offsets2 = bar_offsets(2)

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)

for ax, (title, ds_rmse) in zip(axes, RMSE.items()):
    rand_vals   = [ds_rmse[m]['Random']  for m in models3]
    spatial_vals= [ds_rmse[m]['Spatial'] for m in models3]
    ax.bar(x3 + offsets2[0], rand_vals,    W, color='#5aafe0', label='Random split',  alpha=0.88)
    ax.bar(x3 + offsets2[1], spatial_vals, W, color='#e05a5a', label='Spatial split', alpha=0.88)

    # value labels
    for xi, (rv, sv) in enumerate(zip(rand_vals, spatial_vals)):
        ax.text(xi + offsets2[0], rv + 0.01, f'{rv:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(xi + offsets2[1], sv + 0.01, f'{sv:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x3)
    ax.set_xticklabels(models3, rotation=10)
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE (t/ha)')
    ax.set_title(title)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

axes[0].legend(fontsize=9)
fig.tight_layout()
fig.savefig('figures/figure3_rmse_leakage_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure3_rmse_leakage_gap.png")

# ------------------------------------------------------------------
# FIGURE 4 — Temporal walk-forward per-year R² (Zambia, RF only)
# ------------------------------------------------------------------
zambia_years = [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
zambia_r2    = [0.402,0.590,0.136,0.486,-0.041,0.000,-0.189,0.308,-0.233,0.658,0.537,0.511]

fig, ax = plt.subplots(figsize=(9, 4))
bar_colors = ['#d62728' if r < 0 else '#1f77b4' for r in zambia_r2]
ax.bar(zambia_years, zambia_r2, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.9)
ax.set_xlabel('Test year (walk-forward)')
ax.set_ylabel('R²')
ax.set_xticks(zambia_years)
ax.set_xticklabels(zambia_years, rotation=45)
ax.set_ylim(-0.45, 0.85)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#1f77b4', label='R² >= 0'),
                   Patch(color='#d62728', label='R² < 0')], fontsize=9)

fig.tight_layout()
fig.savefig('figures/figure4_zambia_temporal_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure4_zambia_temporal_breakdown.png")

print("\nAll 4 figures saved to figures/")
