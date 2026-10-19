"""
AutoForge Drift Simulation Script
Simulates realistic production data drift by shifting feature distributions
(e.g., Age distribution shifted to elderly population, Fare shifted to extreme luxury values)
and sending inference payloads to the live prediction service.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import requests

from core.logging_config import logger

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")


def generate_drifted_data(n_samples: int = 35) -> list:
    """
    Creates synthetic drift by systematically shifting feature distributions
    away from the Titanic training baseline:
    - Age: shifted from mean ~29 to elderly population (68 - 85 years)
    - Fare: shifted from mean ~32 to extreme luxury rates (450 - 950)
    - Pclass: shifted entirely to 1st class
    - SibSp: shifted to high family counts (4 - 8)
    """
    df_base = pd.read_csv("data/titanic.csv")
    sample_rows = df_base.drop(columns=["Survived"], errors="ignore").head(n_samples).copy()

    # Apply severe synthetic distribution shifts
    np.random.seed(42)
    sample_rows["Age"] = np.random.uniform(68.0, 85.0, size=len(sample_rows))
    sample_rows["Fare"] = np.random.uniform(450.0, 950.0, size=len(sample_rows))
    sample_rows["Pclass"] = 1
    sample_rows["SibSp"] = np.random.randint(4, 9, size=len(sample_rows))

    # Convert NaNs to None for valid JSON serialization
    clean_df = sample_rows.where(pd.notnull(sample_rows), None)
    return clean_df.to_dict(orient="records")


def simulate_production_drift(
    n_samples: int = 35,
    inference_url: str = INFERENCE_URL,
    batch_size: int = 10,
) -> int:
    """Sends drifted payloads to the live inference endpoint."""
    drifted_records = generate_drifted_data(n_samples)
    logger.info(f"Generated {len(drifted_records)} drifted records. Sending to {inference_url}/predict...")

    total_sent = 0
    for i in range(0, len(drifted_records), batch_size):
        batch = drifted_records[i : i + batch_size]
        try:
            resp = requests.post(f"{inference_url}/predict", json={"rows": batch}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                total_sent += len(batch)
                logger.info(f"Batch {i//batch_size + 1}: sent {len(batch)} rows (latency: {data.get('latency_ms')}ms)")
            else:
                logger.warning(f"Inference request returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to send prediction request to {inference_url}: {e}")

    logger.info(f"Successfully simulated and logged {total_sent} drifted requests into SQLite!")
    return total_sent


def main():
    parser = argparse.ArgumentParser(description="Simulate production data drift for AutoForge")
    parser.add_argument("--samples", type=int, default=35, help="Number of drifted requests to send")
    parser.add_argument("--url", type=str, default=INFERENCE_URL, help="Inference service URL")
    args = parser.parse_args()

    print("=================================================================")
    print("AutoForge Production Drift Simulator")
    print(f"Target: {args.url}/predict | Samples: {args.samples}")
    print("Features shifted: Age (68-85), Fare (450-950), Pclass (1), SibSp (4-8)")
    print("=================================================================")

    sent = simulate_production_drift(n_samples=args.samples, inference_url=args.url)
    print(f"\n[OK] Sent {sent} drifted records to inference service.")
    print("Run `python -m jobs.drift_worker --once` to evaluate drift and auto-trigger retraining.")


if __name__ == "__main__":
    main()
