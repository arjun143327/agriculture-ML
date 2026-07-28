import requests
import json
url = 'https://datasets-server.huggingface.co/info?dataset=CropNet/CropNet'
r = requests.get(url)
info = r.json()
config = info.get('dataset_info', {}).get('default', {})
features = config.get('features', {})
print('CropNet Features:')
for k, v in features.items():
    print(f"- {k}: {v}")
