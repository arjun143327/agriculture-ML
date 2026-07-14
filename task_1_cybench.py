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
        
        # Merge meteo if exists
        if meteo_file in csv_files:
            with z.open(meteo_file) as zf:
                df_meteo = pd.read_csv(io.BytesIO(zf.read()))
            # Meteo might have multiple months, so let's just aggregate or take a specific month
            # For simplicity, if it has 'month', pivot it, or if it's already wide, just merge
            # Let's check columns first
            if 'harvest_year' in df_meteo.columns and 'adm_id' in df_meteo.columns:
                # If there's a 'month' column, we need to pivot
                if 'month' in df_meteo.columns:
                    df_meteo = df_meteo.pivot_table(index=['adm_id', 'harvest_year'], 
                                                    columns='month', 
                                                    aggfunc='mean').reset_index()
                    df_meteo.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_meteo.columns]
                
                df = pd.merge(df, df_meteo, on=['adm_id', 'harvest_year'], how='left')
                
        # Merge soil if exists
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
