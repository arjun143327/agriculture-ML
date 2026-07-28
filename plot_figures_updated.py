import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# Consistent colors
COLOR_RF = '#1f77b4'  # blue
COLOR_XGB = '#ff7f0e' # orange
COLOR_NULL = '#7f7f7f' # gray
COLOR_RAW = '#d62728' # red
COLOR_DET = '#2ca02c' # green

plt.rcParams.update({'font.size': 12})

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f'figures/{name}', dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# FIGURE 1: Leakage gap overview (3 Datasets)
# ---------------------------------------------------------
labels = ['Random', 'Spatial', 'Temporal', 'Spatiotemp.']
x = np.arange(len(labels))
width = 0.35

# CY-Bench Maize
cy_rf_means = [0.728, 0.536, 0.293, 0.300]
cy_rf_stds = [0.041, 0.087, 0.861, 0.116]
cy_xgb_means = [0.755, 0.465, 0.280, 0.196]
cy_xgb_stds = [0.033, 0.132, 0.796, 0.077]

# CY-Bench Wheat
wh_rf_means = [0.671, 0.307, 0.267, -0.052]
wh_rf_stds = [0.020, 0.269, 0.376, 0.020]
wh_xgb_means = [0.676, 0.371, 0.259, -0.175]
wh_xgb_stds = [0.019, 0.159, 0.385, 0.088]

# SustainBench
sb_rf_means = [0.401, 0.391, -0.060, 0.249]
sb_rf_stds = [0.007, 0.029, 0.497, 0.109]
sb_xgb_means = [0.371, 0.355, -0.071, 0.182]
sb_xgb_stds = [0.016, 0.029, 0.496, 0.182]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Maize
ax1.bar(x - width/2, cy_rf_means, width, yerr=cy_rf_stds, label='Random Forest', color=COLOR_RF, capsize=4)
ax1.bar(x + width/2, cy_xgb_means, width, yerr=cy_xgb_stds, label='XGBoost', color=COLOR_XGB, capsize=4)
ax1.set_title('CY-Bench Maize')
ax1.set_ylabel('R² Score')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(-1.5, 1.0)
ax1.axhline(0, color='black', linewidth=0.8)

# Wheat
ax2.bar(x - width/2, wh_rf_means, width, yerr=wh_rf_stds, color=COLOR_RF, capsize=4)
ax2.bar(x + width/2, wh_xgb_means, width, yerr=wh_xgb_stds, color=COLOR_XGB, capsize=4)
ax2.set_title('CY-Bench Wheat')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylim(-1.5, 1.0)
ax2.axhline(0, color='black', linewidth=0.8)

# SustainBench
ax3.bar(x - width/2, sb_rf_means, width, yerr=sb_rf_stds, color=COLOR_RF, capsize=4)
ax3.bar(x + width/2, sb_xgb_means, width, yerr=sb_xgb_stds, color=COLOR_XGB, capsize=4)
ax3.set_title('SustainBench')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.set_ylim(-1.5, 1.0)
ax3.axhline(0, color='black', linewidth=0.8)

fig.legend(['Random Forest', 'XGBoost'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
save_fig('figure1_leakage_gap.png')


# ---------------------------------------------------------
# FIGURE 3: Null baseline comparison
# ---------------------------------------------------------
labels = ['Spatial Split', 'Temporal Split']
x = np.arange(len(labels))
width = 0.25

# Maize
cy_null = [-0.012, -0.560]
cy_rf = [0.536, 0.293]
cy_xgb = [0.465, 0.280]

# Wheat
wh_null = [-0.019, -0.394]
wh_rf = [0.307, 0.267]
wh_xgb = [0.371, 0.259]

# SustainBench
sb_null = [-0.001, -0.384]
sb_rf = [0.391, -0.060]
sb_xgb = [0.355, -0.071]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Maize
ax1.bar(x - width, cy_null, width, label='Null Baseline', color=COLOR_NULL)
ax1.bar(x, cy_rf, width, label='Random Forest', color=COLOR_RF)
ax1.bar(x + width, cy_xgb, width, label='XGBoost', color=COLOR_XGB)
ax1.set_title('CY-Bench Maize')
ax1.set_ylabel('R² Score')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_ylim(-0.8, 0.8)

# Wheat
ax2.bar(x - width, wh_null, width, color=COLOR_NULL)
ax2.bar(x, wh_rf, width, color=COLOR_RF)
ax2.bar(x + width, wh_xgb, width, color=COLOR_XGB)
ax2.set_title('CY-Bench Wheat')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylim(-0.8, 0.8)

# SustainBench
ax3.bar(x - width, sb_null, width, color=COLOR_NULL)
ax3.bar(x, sb_rf, width, color=COLOR_RF)
ax3.bar(x + width, sb_xgb, width, color=COLOR_XGB)
ax3.set_title('SustainBench')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_ylim(-0.8, 0.8)

fig.legend(['Null Baseline', 'Random Forest', 'XGBoost'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)

save_fig('figure3_null_baseline.png')

print("Figures updated to include CY-Bench Wheat.")
