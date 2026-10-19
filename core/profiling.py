"""
AutoForge Dataset Profiling Engine
Generates detailed exploratory data profiling reports using ydata-profiling
and saves HTML reports as artifacts.
"""

import os
import time
from typing import Dict, Any, Optional
import pandas as pd
from ydata_profiling import ProfileReport
from core.logging_config import logger


def generate_profiling_report(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str = "artifacts/reports",
    minimal: bool = False,
) -> Dict[str, Any]:
    """
    Generates a comprehensive ydata-profiling HTML report and saves it to output_dir.

    Args:
        df: Pandas DataFrame to profile
        dataset_name: Identifier name for the dataset (used in file naming and report title)
        output_dir: Directory where the HTML report will be saved
        minimal: Whether to use minimal calculation mode (faster for large datasets)

    Returns:
        Dict with report file path, generation duration, and dataset summary metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in dataset_name)
    output_path = os.path.join(output_dir, f"{safe_name}_profile.html")

    start_time = time.time()
    logger.info(f"Generating ydata-profiling report for '{dataset_name}' (rows: {len(df)}, cols: {len(df.columns)})...")

    # Build profile report
    profile = ProfileReport(
        df,
        title=f"AutoForge Profile: {dataset_name}",
        dataset={"description": f"AutoForge Automated Profiling for {dataset_name}"},
        explorative=True,
        minimal=minimal,
        progress_bar=False,
    )

    # Save to HTML file
    profile.to_file(output_path)
    duration = time.time() - start_time
    file_size_kb = round(os.path.getsize(output_path) / 1024, 2)

    logger.info(f"Profiling report generated successfully at: {output_path} ({file_size_kb} KB, took {duration:.2f}s)")

    # Extract summary metrics from description if available
    summary_metrics = {
        "dataset_name": dataset_name,
        "report_path": output_path,
        "file_size_kb": file_size_kb,
        "generation_duration_sec": round(duration, 2),
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_cells_pct": round(float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return summary_metrics
