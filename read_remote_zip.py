import requests
import zipfile
import io
import pandas as pd

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

    def seekable(self):
        return True

    def readable(self):
        return True

url = "https://zenodo.org/api/records/17279151/files/cybench-data.zip/content"
print(f"Opening remote zip file at {url}")
try:
    f = HttpFile(url)
    z = zipfile.ZipFile(f)
    print("Zip file opened successfully. Listing contents:")
    csv_files = []
    for info in z.infolist():
        if info.filename.endswith('.csv'):
            csv_files.append(info.filename)
            
    print(f"Found {len(csv_files)} CSV files.")
    
    if csv_files:
        # Print a few to see what they are named
        print("Sample files:", csv_files[:10])
        yield_files = [f for f in csv_files if 'yield' in f.lower()]
        target = yield_files[0] if yield_files else csv_files[0]
        print(f"Extracting {target}...")
        with z.open(target) as zf:
            df = pd.read_csv(zf)
        
        print(df.info())
        print(df.head(10))
        
        # Save a sample
        sample = df.head(1000)
        sample.to_csv("cybench_sample.csv", index=False)
        print("Saved cybench_sample.csv")
    else:
        print("No CSV files found in zip.")
except Exception as e:
    print("Error:", e)
