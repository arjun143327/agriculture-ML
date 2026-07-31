import pandas as pd

with open('report.md', 'w') as f:
    f.write("### 2. Feature Importance\n\n")
    for dataset, filename in [('CY-Bench Wheat', 'feat_importance_CY-Bench_Wheat_Europe.csv'), 
                              ('CY-Bench Zambia Maize', 'feat_importance_CY-Bench_Maize_Zambia.csv')]:
        f.write(f"**{dataset}**\n")
        df = pd.read_csv(filename).head(8)
        f.write("Feature                 Rand mean  Rand SD  Spat mean  Spat SD\n")
        f.write("------------------------------------------------------------\n")
        for _, row in df.iterrows():
            f.write(f"{row['feature']:<23} {row['rand_mean']:>8.4f} {row['rand_sd']:>8.4f} {row['spat_mean']:>10.4f} {row['spat_sd']:>8.4f}\n")
        f.write("\n")

    f.write("### 3. Full Rolling-Window Spatiotemporal Data Table\n\n")
    files = {
        'CY-Bench Maize': 'rolling_st_CY-Bench_Maize_Europe.csv',
        'CY-Bench Wheat': 'rolling_st_CY-Bench_Wheat_Europe.csv',
        'CY-Bench Zambia Maize': 'rolling_st_CY-Bench_Maize_Zambia.csv',
        'SustainBench Soybean': 'rolling_st_SustainBench_Soybean.csv'
    }
    
    anomaly_mapping = {
        'CY-Bench Maize': [2018],
        'CY-Bench Wheat': [2018],
        'CY-Bench Zambia Maize': [2012, 2014],
        'SustainBench Soybean': [2012]
    }

    f.write("| Dataset | Window (start-end year) | RF R² | Ridge R² | XGB R² | Anomaly year(s) in window? |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    
    for name, filename in files.items():
        df = pd.read_csv(filename)
        pivoted = df.pivot_table(index=['window_label'], columns='model', values='r2_mean').reset_index()
        
        for _, row in pivoted.iterrows():
            window = row['window_label']
            rf = row.get('Random Forest', float('nan'))
            ridge = row.get('Ridge', float('nan'))
            xgb = row.get('XGBoost', float('nan'))
            
            # Check for anomalies
            window_years = []
            if '-' in str(window):
                try:
                    start, end = map(int, str(window).split('-'))
                    window_years = list(range(start, end + 1))
                except:
                    pass
                
            anomalies = [str(y) for y in anomaly_mapping[name] if y in window_years]
            anomaly_str = ", ".join(anomalies) if anomalies else "No"
            
            f.write(f"| {name} | {window} | {rf:.3f} | {ridge:.3f} | {xgb:.3f} | {anomaly_str} |\n")
