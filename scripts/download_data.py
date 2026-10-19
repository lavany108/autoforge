"""
AutoForge Seed Data Downloader
Downloads Titanic dataset and fetches California Housing dataset via scikit-learn.
"""

import os
import urllib.request
import pandas as pd
from core.logging_config import logger


def download_titanic(output_path: str = "data/titanic.csv") -> str:
    """Download Titanic dataset if not already present."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        logger.info(f"Titanic dataset already exists at {output_path}")
        return output_path

    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    logger.info(f"Downloading Titanic dataset from {url}...")
    urllib.request.urlretrieve(url, output_path)
    logger.info(f"Saved Titanic dataset to {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


def download_california_housing(output_path: str = "data/california_housing.csv") -> str:
    """Fetch California Housing dataset from scikit-learn and save as CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        logger.info(f"California Housing dataset already exists at {output_path}")
        return output_path

    from sklearn.datasets import fetch_california_housing

    logger.info("Fetching California Housing dataset via sklearn...")
    housing = fetch_california_housing(as_frame=True)
    df: pd.DataFrame = housing.frame
    df.to_csv(output_path, index=False)
    logger.info(f"Saved California Housing dataset to {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    return output_path


def main():
    download_titanic()
    download_california_housing()
    print("Seed datasets ready in ./data/")


if __name__ == "__main__":
    main()
