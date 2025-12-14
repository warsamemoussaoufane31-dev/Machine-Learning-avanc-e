import os
import tarfile
import urllib.request

HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
HOUSING_PATH = "data/housing"


def fetch_housing_data(url: str = HOUSING_URL, path: str = HOUSING_PATH) -> None:
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "housing.tgz")
    csv_path = os.path.join(path, "housing.csv")
    if not os.path.exists(csv_path):
        print("Downloading housing dataset...")
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path) as housing_tgz:
            housing_tgz.extractall(path=path)
        print("Download complete.")
    else:
        print("Dataset already present, skipping download.")
