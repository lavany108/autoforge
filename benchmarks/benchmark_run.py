"""
AutoForge Automated Benchmark Runner
Executes Baseline vs Tuned (FLAML + Optuna) evaluation across:
1. Titanic (Classification - Accuracy)
2. California Housing (Regression - RMSE)
Outputs a markdown comparison table with metrics, uplift, and training times.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.download_data import download_titanic, download_california_housing
from core.model_search import train_autoforge_model


def run_benchmarks() -> str:
    print("=" * 70)
    print("🚀 AUTOFORGE AUTOMATED BENCHMARK SUITE")
    print("=" * 70)

    os.makedirs("benchmarks/results", exist_ok=True)

    # Ensure datasets are present
    titanic_path = download_titanic("data/titanic.csv")
    housing_path = download_california_housing("data/california_housing.csv")

    results = []

    # -----------------------------------------------------------------------
    # Benchmark 1: Titanic Classification
    # -----------------------------------------------------------------------
    print("\n[1/2] Running Benchmark on Titanic (Classification)...")
    df_titanic = pd.read_csv(titanic_path)

    t_start = time.time()
    titanic_res = train_autoforge_model(
        df=df_titanic,
        target_column="Survived",
        task_type="classification",
        time_budget_secs=30,
        optuna_trials=150,
        model_name="benchmark_titanic",
        experiment_name="benchmark_experiments",
    )
    b_acc = titanic_res["baseline_metrics"]["accuracy"]
    t_acc = titanic_res["tuned_metrics"]["accuracy"]
    b_time = titanic_res["baseline_metrics"]["duration_seconds"]
    t_time = titanic_res["tuned_metrics"]["duration_seconds"]
    winner = titanic_res.get("flaml_winner", "rf").upper()
    uplift = t_acc - b_acc

    results.append({
        "Dataset": "Titanic",
        "Task": "Classification",
        "Metric": "Accuracy",
        "Baseline Score": f"{b_acc:.4f}",
        "Tuned Score": f"{t_acc:.4f}",
        "Uplift": f"{uplift:+.4f}",
        "Baseline Time": f"{b_time:.2f}s",
        "Tuned Time": f"{t_time:.2f}s",
        "Best Model": winner,
    })
    print(f"  Titanic Completed: Baseline={b_acc:.4f} -> Tuned={t_acc:.4f} (Uplift: {uplift:+.4f})")

    # -----------------------------------------------------------------------
    # Benchmark 2: California Housing Regression
    # -----------------------------------------------------------------------
    print("\n[2/2] Running Benchmark on California Housing (Regression)...")
    df_housing = pd.read_csv(housing_path)
    # Use 3000 rows for fast, reproducible benchmark evaluation
    if len(df_housing) > 3000:
        df_housing = df_housing.sample(n=3000, random_state=42)

    housing_res = train_autoforge_model(
        df=df_housing,
        target_column="MedHouseVal",
        task_type="regression",
        time_budget_secs=20,
        optuna_trials=15,
        model_name="benchmark_housing",
        experiment_name="benchmark_experiments",
    )
    b_rmse = housing_res["baseline_metrics"]["rmse"]
    t_rmse = housing_res["tuned_metrics"]["rmse"]
    b_time_h = housing_res["baseline_metrics"]["duration_seconds"]
    t_time_h = housing_res["tuned_metrics"]["duration_seconds"]
    winner_h = housing_res.get("flaml_winner", "rf").upper()
    uplift_rmse = b_rmse - t_rmse  # Positive means error reduction (better)

    results.append({
        "Dataset": "California Housing",
        "Task": "Regression",
        "Metric": "RMSE",
        "Baseline Score": f"{b_rmse:.4f}",
        "Tuned Score": f"{t_rmse:.4f}",
        "Uplift": f"{uplift_rmse:+.4f} (lower is better)",
        "Baseline Time": f"{b_time_h:.2f}s",
        "Tuned Time": f"{t_time_h:.2f}s",
        "Best Model": winner_h,
    })
    print(f"  California Housing Completed: Baseline RMSE={b_rmse:.4f} -> Tuned RMSE={t_rmse:.4f} (Reduction: {uplift_rmse:+.4f})")

    # -----------------------------------------------------------------------
    # Format Markdown Table
    # -----------------------------------------------------------------------
    df_res = pd.DataFrame(results)
    try:
        md_table = df_res.to_markdown(index=False)
    except Exception:
        md_table = str(df_res)

    summary_md = f"""# AutoForge AutoML Benchmark Results

| Dataset | Task | Metric | Baseline Score | Tuned Score | Uplift | Baseline Time | Tuned Time | Best Model |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Titanic** | Classification | Accuracy | {b_acc:.4f} | **{t_acc:.4f}** | **{uplift:+.4f}** | {b_time:.2f}s | {t_time:.2f}s | `{winner}` |
| **California Housing** | Regression | RMSE | {b_rmse:.4f} | **{t_rmse:.4f}** | **{uplift_rmse:+.4f}** | {b_time_h:.2f}s | {t_time_h:.2f}s | `{winner_h}` |
"""

    out_file = "benchmarks/results/benchmark_summary.md"
    with open(out_file, "w") as f:
        f.write(summary_md)

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS TABLE")
    print("=" * 70)
    print(summary_md)
    print(f"\nSaved benchmark markdown table to: {out_file}")
    return summary_md


if __name__ == "__main__":
    run_benchmarks()
