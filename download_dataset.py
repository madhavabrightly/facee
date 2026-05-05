"""Helper script to download the PlantVillage dataset using kagglehub.

This script wraps the example the user provided and unpacks the archive into
our local ``Dataset/`` directory so that the training notebook can use it.

Usage::

    pip install kagglehub
    python download_dataset.py

Note: ``kagglehub`` must be configured with your Kaggle API credentials (either
via environment variables or its usual config file). See
https://pypi.org/project/kagglehub/ for details.
"""

import os
import zipfile
import kagglehub


def main():
    # download dataset archive from Kaggle
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("emmarex/plantdisease")
    print("Path to downloaded file:", path)

    # create Dataset directory if it doesn't exist
    target_dir = os.path.join(os.getcwd(), "Dataset")
    os.makedirs(target_dir, exist_ok=True)

    # unzip if necessary
    if path.lower().endswith(".zip"):
        print(f"Extracting {path} to {target_dir}...")
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(target_dir)
        print("Extraction complete.")
    else:
        print("Downloaded file is not a ZIP archive; please unpack it manually.")

    print("Dataset ready in", target_dir)


if __name__ == "__main__":
    main()
