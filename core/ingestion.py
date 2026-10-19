"""
AutoForge Data Ingestion & Inspection Module
Handles loading, basic cleaning, validation, and schema inspection of datasets.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
from core.logging_config import logger


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from CSV or parquet file into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"Loading dataset from: {file_path} (format: {ext})")

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(file_path)
    else:
        # Fallback to read_csv for other delimited text files
        df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Dataset at {file_path} is empty.")

    logger.info(f"Loaded dataset with shape {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects DataFrame and returns summary statistics, missing values,
    and column type categorizations.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    missing_counts = df.isnull().sum().to_dict()
    missing_pct = {col: round(count / len(df) * 100, 2) for col, count in missing_counts.items()}

    summary = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "datetime_features": datetime_cols,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "head_sample": df.head(5).to_dict(orient="records"),
    }
    return summary
