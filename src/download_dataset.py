import os
import sys
import zipfile
import requests
import torch
from tqdm import tqdm

DATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "rizkyyk/dataset-food-classification"
)

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIRECTORY, "..", "data", "raw")
ZIP_NAME = "dataset-food-classification.zip"
ZIP_PATH = os.path.join(OUTPUT_DIR, ZIP_NAME)

def download_dataset(url, output_path):
    print("📥 Downloading dataset using PyTorch...")
    torch.hub.download_url_to_file(
        url=url,
        dst=output_path,
        progress=True
    )
    print(f"✅ Download complete: {output_path}")

def unzip_dataset(zip_path, extract_to):
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.infolist()

        for file in tqdm(file_list, desc="Extracting"):
            original_path = file.filename

            if original_path.endswith(os.sep):
                continue

            path_components = original_path.split(os.sep)
            new_relative_path = os.path.join(*path_components[1:])

            target_path = os.path.join(extract_to, new_relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with zip_ref.open(file) as source, open(target_path, "wb") as target:
                target.write(source.read())
                
    print("✅ Extraction complete.")

def directory_size_mb(path):
    total_size = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if fp.endswith(".zip"):
                continue
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(ZIP_PATH):
        print("📥 Starting dataset download...")
        try:
            download_dataset(DATA_URL, ZIP_PATH)
            print(f"✅ Download complete: {ZIP_PATH}")
        except Exception as e:
            print("❌ Download failed.")
            raise e
    else:
        print(f"ℹ️ Dataset zip already exists: {ZIP_PATH}")
    
    print("📦 Checking extracted dataset size...")
    size_mb = directory_size_mb(OUTPUT_DIR)

    if size_mb < 10:
        unzip_dataset(ZIP_PATH, OUTPUT_DIR)
    else:
        print(f"ℹ️ data/raw already contains {size_mb:.2f} MB. Skipping extraction.")

    print("🎉 Dataset ready in data/raw/")

if __name__ == "__main__":
    main()