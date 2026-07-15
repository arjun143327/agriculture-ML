import requests
import zipfile
import io
import pandas as pd
import numpy as np

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        r = requests.head(url, allow_redirects=True)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch HEAD, status {r.status_code}")
        self.length = int(r.headers.get('Content-Length', 0))
        self.pos = 0
        self.session = requests.Session()

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1:
            size = self.length - self.pos
        if size == 0:
            return b""
        end = self.pos + size - 1
        headers = {'Range': f'bytes={self.pos}-{end}'}
        r = self.session.get(self.url, headers=headers)
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data

    def seekable(self): return True
    def readable(self): return True

def main():
    url = "https://zenodo.org/api/records/17279151/files/cybench-data.zip/content"
    print("Opening remote zip file...")
    f = HttpFile(url)
    z = zipfile.ZipFile(f)
    
    csv_files = [info.filename for info in z.infolist() if info.filename.endswith('.csv')]
    
    # Identify countries that have maize data
    maize_files = [f for f in csv_files if 'cybench-data/maize/' in f]
    countries = list(set([f.split('/')[2] for f in maize_files]))
    print(f"Available countries for Maize: {countries}")
    
    # Pick 6 countries
    target_countries = ['HR', 'HU', 'RS', 'BG', 'RO', 'AT']
    # fallback to others if missing
    target_countries = [c for c in target_countries if c in countries]
    if len(target_countries) < 5:
        target_countries = countries[:6]
    
    print(f"Selected countries: {target_countries}")
    
    all_data = []
    
    for country in target_countries:
        print(f"Processing {country}...")
        
        # Load yield
        yield_file = f'cybench-data/maize/{country}/yield_maize_{country}.csv'
        loc_file = f'cybench-data/maize/{country}/location_maize_{country}.csv'
        meteo_file = f'cybench-data/maize/{country}/meteo_maize_{country}.csv'
        soil_file = f'cybench-data/maize/{country}/soil_maize_{country}.csv'
        
        if yield_file not in csv_files or loc_file not in csv_files:
            continue
            
        with z.open(yield_file) as zf:
            df_yield = pd.read_csv(io.BytesIO(zf.read()))
        
        with z.open(loc_file) as zf:
            df_loc = pd.read_csv(io.BytesIO(zf.read()))
        
        # Merge loc to yield
        df = pd.merge(df_yield, df_loc[['adm_id', 'latitude', 'longitude']], on='adm_id', how='left')
        
        # Helper function to process daily files
        def process_daily(file_name, agg_dict):
            if file_name not in csv_files: return None
            with z.open(file_name) as zf:
                df_daily = pd.read_csv(io.BytesIO(zf.read()))
            # Convert YYYYMMDD to Year
            df_daily['harvest_year'] = (df_daily['date'] // 10000).astype(int)
            # Group by adm_id and harvest_year
            df_agg = df_daily.groupby(['adm_id', 'harvest_year']).agg(agg_dict).reset_index()
            # Flatten columns
            df_agg.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_agg.columns.values]
            return df_agg

        # Meteo: sum prec, mean for temperatures and others
        meteo_agg = {
            'prec': ['sum'],
            'tmin': ['mean'], 'tmax': ['mean'], 'tavg': ['mean'],
            'rad': ['mean'], 'et0': ['mean'], 'vpd': ['mean'], 'cwb': ['mean']
        }
        df_meteo = process_daily(f'cybench-data/maize/{country}/meteo_maize_{country}.csv', meteo_agg)
        if df_meteo is not None:
            df = pd.merge(df, df_meteo, on=['adm_id', 'harvest_year'], how='left')
            
        # NDVI: mean, max
        df_ndvi = process_daily(f'cybench-data/maize/{country}/ndvi_maize_{country}.csv', {'ndvi': ['mean', 'max']})
        if df_ndvi is not None:
            df = pd.merge(df, df_ndvi, on=['adm_id', 'harvest_year'], how='left')
            
        # FPAR: mean, max
        df_fpar = process_daily(f'cybench-data/maize/{country}/fpar_maize_{country}.csv', {'fpar': ['mean', 'max']})
        if df_fpar is not None:
            df = pd.merge(df, df_fpar, on=['adm_id', 'harvest_year'], how='left')

        # Merge soil if exists
        soil_file = f'cybench-data/maize/{country}/soil_maize_{country}.csv'
        if soil_file in csv_files:
            with z.open(soil_file) as zf:
                df_soil = pd.read_csv(io.BytesIO(zf.read()))
            if 'adm_id' in df_soil.columns:
                df = pd.merge(df, df_soil, on='adm_id', how='left')
                
        all_data.append(df)
        
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("cybench_full.csv", index=False)
    print("\n--- TASK 1 DONE ---")
    print(f"Total Rows: {len(final_df)}")
    print(f"Unique Regions: {final_df['adm_id'].nunique()}")
    print(f"Predictor Columns: {len(final_df.columns)}")

if __name__ == '__main__':
    main()
