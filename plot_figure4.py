import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def get_color(feature_name):
    static_features = ['bulk_density', 'awc', 'drainage_class']
    return '#ff7f0e' if feature_name in static_features else '#1f77b4'

datasets = [
    ('CY-Bench Maize (Europe)', 'feat_importance_CY-Bench_Maize_Europe.csv'),
    ('CY-Bench Wheat (Europe)', 'feat_importance_CY-Bench_Wheat_Europe.csv'),
    ('CY-Bench Zambia Maize', 'feat_importance_CY-Bench_Maize_Zambia.csv')
]

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle('Feature Importance Under Random vs. Spatial Split, XGBoost\n(Mean ± SD Across 3 Seeds), by Dataset', fontsize=16, y=0.95)

for i, (title, csv_file) in enumerate(datasets):
    # Load and get top 8 features based on Rand_mean (to keep order consistent)
    try:
        df = pd.read_csv(csv_file)
        # Ensure column names are stripped of whitespace
        df.columns = df.columns.str.strip()
        # Fallback for lowercase/uppercase
        if 'rand_mean' in df.columns:
            df = df.rename(columns={'feature':'Feature', 'rand_mean':'Rand_mean', 'rand_sd':'Rand_SD', 'spat_mean':'Spat_mean', 'spat_sd':'Spat_SD'})
        
        df = df.sort_values(by='Rand_mean', ascending=True).tail(8) # ascending=True because horizontal bar puts first at bottom
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        continue

    features = df['Feature'].tolist()
    
    # Colors for each feature
    colors = [get_color(f) for f in features]
    
    # Plot Random Split (Left)
    ax_rand = axes[i, 0]
    bars_rand = ax_rand.barh(features, df['Rand_mean'], xerr=df['Rand_SD'], color=colors, capsize=4)
    ax_rand.set_title(f'{title} - Random Split')
    ax_rand.set_xlabel('Importance')
    ax_rand.set_xlim(0, max(df['Rand_mean']) + 0.1)
    
    # Add value labels
    for bar in bars_rand:
        width = bar.get_width()
        ax_rand.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)

    # Plot Spatial Split (Right)
    ax_spat = axes[i, 1]
    # Keep the same y-axis order as random split for easy comparison
    bars_spat = ax_spat.barh(features, df['Spat_mean'], xerr=df['Spat_SD'], color=colors, capsize=4)
    ax_spat.set_title(f'{title} - Spatial Split')
    ax_spat.set_xlabel('Importance')
    ax_spat.set_xlim(0, max(df['Spat_mean']) + 0.1)

    # Add value labels
    for bar in bars_spat:
        width = bar.get_width()
        ax_spat.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)

# Add a custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', label='Dynamic/Weather Feature'),
    Patch(facecolor='#ff7f0e', label='Static/Soil Feature')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=2, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig('figures/figure4_feature_importance_combined.png', dpi=300, bbox_inches='tight')
print("Successfully generated figure4_feature_importance_combined.png")
