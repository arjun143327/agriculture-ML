import pandas as pd

def validate_dataset(df, dataset_name, region_col, year_col, target_col, ignore_cols):
    print(f"\n{'='*50}")
    print(f"VALIDATION REPORT: {dataset_name}")
    print(f"{'='*50}")
    
    # 1. Duplicate (region, year) rows
    duplicates = df.duplicated(subset=[region_col, year_col]).sum()
    print(f"1. Duplicate ({region_col}, {year_col}) rows: {duplicates}")
    
    # 2. Missing values per column
    print(f"\n2. Missing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    if missing.sum() == 0:
        print("   None")
        
    # 3. Summary statistics of the target
    print(f"\n3. Summary statistics of target ({target_col}):")
    print(df[target_col].describe())
    
    # 4. Numeric predictor columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    predictor_cols = [c for c in numeric_cols if c not in ignore_cols and c != target_col]
    print(f"\n4. Number of numeric predictor columns: {len(predictor_cols)}")
    
    # 5. Unique years
    unique_years = sorted(df[year_col].unique())
    print(f"\n5. Unique years: {len(unique_years)} ({min(unique_years)} - {max(unique_years)})")
    
    # 6. Unique regions
    unique_regions = df[region_col].nunique()
    print(f"\n6. Unique regions: {unique_regions}")
    
    # 7. Constant predictor columns
    print("\n7. Constant predictor columns:")
    constant_cols = [c for c in predictor_cols if df[c].nunique() <= 1]
    if len(constant_cols) > 0:
        print(f"   {constant_cols}")
    else:
        print("   None")
        
    return {
        'duplicates': duplicates,
        'missing': missing,
        'constant_cols': constant_cols
    }

def main():
    print("Loading cybench_full.csv...")
    cy_df = pd.read_csv("cybench_full.csv")
    validate_dataset(
        cy_df, 
        "CY-Bench", 
        region_col="adm_id", 
        year_col="harvest_year", 
        target_col="yield", 
        ignore_cols=["harvest_year", "latitude", "longitude", "harvest_area", "production"]
    )
    
    print("\nLoading sustainbench_full.csv...")
    sb_df = pd.read_csv("sustainbench_full.csv")
    validate_dataset(
        sb_df, 
        "SustainBench", 
        region_col="key", 
        year_col="year", 
        target_col="yield_t_ha", 
        ignore_cols=["year"]
    )

if __name__ == '__main__':
    main()
