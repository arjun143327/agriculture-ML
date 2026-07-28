import requests
import zipfile
import io
import pandas as pd
import numpy as np

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        r = requests.head(url, allow_redirects=True)
        self.length = int(r.headers.get('Content-Length', 0))
        self.pos = 0
        self.session = requests.Session()
    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.length + offset
        return self.pos
    def tell(self): return self.pos
    def read(self, size=-1):
        if size == -1: size = self.length - self.pos
        if size == 0: return b''
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

    target_crop = "maize"
    target_country = "ZM"

    print(f"Processing {target_country} for {target_crop}...")

    # Load Yield
    yield_file = f"cybench-data/{target_crop}/{target_country}/yield_{target_crop}_{target_country}.csv"
    with z.open(yield_file) as zf:
        df_y = pd.read_csv(io.BytesIO(zf.read()))
    
    # Select only needed columns from yield file (drop crop_name, use as label only)
    df_y = df_y[['adm_id', 'harvest_year', 'yield', 'harvest_area', 'production', 'country_code']]
    
    # Strictly enforce identical zero-yield filtering
    df_y = df_y[df_y['yield'] > 0]

    # Load Location
    loc_file = f"cybench-data/{target_crop}/{target_country}/location_{target_crop}_{target_country}.csv"
    with z.open(loc_file) as zf:
        df_loc = pd.read_csv(io.BytesIO(zf.read()))
    
    # Load Soil (Static)
    soil_file = f"cybench-data/{target_crop}/{target_country}/soil_{target_crop}_{target_country}.csv"
    with z.open(soil_file) as zf:
        df_soil = pd.read_csv(io.BytesIO(zf.read()))
    
    # Load Time-series Data
    meteo_file = f"cybench-data/{target_crop}/{target_country}/meteo_{target_crop}_{target_country}.csv"
    with z.open(meteo_file) as zf:
        df_meteo = pd.read_csv(io.BytesIO(zf.read()))
    
    ndvi_file = f"cybench-data/{target_crop}/{target_country}/ndvi_{target_crop}_{target_country}.csv"
    with z.open(ndvi_file) as zf:
        df_ndvi = pd.read_csv(io.BytesIO(zf.read()))
        
    fpar_file = f"cybench-data/{target_crop}/{target_country}/fpar_{target_crop}_{target_country}.csv"
    with z.open(fpar_file) as zf:
        df_fpar = pd.read_csv(io.BytesIO(zf.read()))

    # Convert date to harvest_year for aggregation
    for df in [df_meteo, df_ndvi, df_fpar]:
        df['harvest_year'] = df['date'] // 10000

    # Aggregate to yearly level identical to Task 1
    agg_meteo = df_meteo.groupby(['adm_id', 'harvest_year']).agg(
        prec_sum=('prec', 'sum'),
        tmin_mean=('tmin', 'mean'),
        tmax_mean=('tmax', 'mean'),
        tavg_mean=('tavg', 'mean'),
        rad_mean=('rad', 'mean'),
        et0_mean=('et0', 'mean'),
        vpd_mean=('vpd', 'mean'),
        cwb_mean=('cwb', 'mean')
    ).reset_index()

    agg_ndvi = df_ndvi.groupby(['adm_id', 'harvest_year']).agg(
        ndvi_mean=('ndvi', 'mean'),
        ndvi_max=('ndvi', 'max')
    ).reset_index()

    agg_fpar = df_fpar.groupby(['adm_id', 'harvest_year']).agg(
        fpar_mean=('fpar', 'mean'),
        fpar_max=('fpar', 'max')
    ).reset_index()

    # Merge exactly as before — only pull lat/lon from location file
    df_merged = df_y.merge(df_loc[['adm_id', 'latitude', 'longitude']], on='adm_id', how='left')
    df_merged = df_merged.merge(df_soil[['adm_id', 'awc', 'bulk_density', 'drainage_class']], on='adm_id', how='left')
    df_merged = df_merged.merge(agg_meteo, on=['adm_id', 'harvest_year'], how='inner')
    df_merged = df_merged.merge(agg_ndvi, on=['adm_id', 'harvest_year'], how='inner')
    df_merged = df_merged.merge(agg_fpar, on=['adm_id', 'harvest_year'], how='inner')

    # Drop NaNs
    df_merged = df_merged.dropna()
    print(f"Total Rows for {target_country}: {len(df_merged)}")

    df_merged.to_csv("cybench_zambia_maize.csv", index=False)
    print("Saved to cybench_zambia_maize.csv")

if __name__ == '__main__':
    main()
