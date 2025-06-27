import os
import requests

def download_if_not_exists(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading model to {dest}...")
        r = requests.get(url)
        with open(dest, "wb") as f:
            f.write(r.content)