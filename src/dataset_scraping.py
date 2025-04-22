import os
import requests
import zipfile
from io import BytesIO


def scrap_dataset():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    english_datasets = [
        "Krapivin2009",
        "Nguyen2007",
        "PubMed",
        "Schutz2008",
        "SemEval2010",
        "SemEval2017",
        "citeulike180",
        "fao30",
        "fao780",
        "kdd",
        "theses100",
        "wiki20",
        "www"
    ]
    base_url = "https://raw.githubusercontent.com/INESCTEC/KeywordExtractor-Datasets/refs/heads/master/datasets/"

    for dataset in english_datasets:
        zip_url = f"{base_url}{dataset}.zip"
        print(f"Downloading {dataset}...")
        response = requests.get(zip_url)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                extract_path = os.path.join(data_dir, dataset)
                os.makedirs(extract_path, exist_ok=True)
                zf.extractall(extract_path)
            print(f"Extracted {dataset} to {extract_path}")
        else:
            print(f"Failed to download {dataset}: {response.status_code}")


if __name__ == '__main__':
    scrap_dataset()
