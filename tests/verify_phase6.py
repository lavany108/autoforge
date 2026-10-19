"""
Phase 6 Verification Script
Tests end-to-end drift detection and automated retraining lifecycle:
1. Check current registered model version in MLflow
2. Run simulate_drift.py to inject feature distribution shifts into SQLite
3. Execute drift worker cycle:
   - Compares recent requests vs reference data using Evidently & PSI
   - Checks trigger rules (drift share > 30% OR PSI > 0.20)
   - Triggers automated AutoML retraining run via API
   - Registers new model version in MLflow registry
   - Hot-swaps inference service
4. Verifies new model version is registered and active
"""

import os
import sys
import json
import time
import requests
import mlflow

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.simulate_drift import simulate_production_drift
from jobs.drift_worker import run_drift_cycle

API_URL = "http://localhost:8000"
INFERENCE_URL = "http://localhost:8001"
MODEL_NAME = "autoforge_titanic"


def get_latest_registry_version(model_name: str) -> int:
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('artifacts/mlflow.db')}")
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max([int(v.version) for v in versions])
    except Exception:
        pass
    return 1


def verify_phase6():
    print("==================================================================")
    print("STARTING PHASE 6 VERIFICATION: DRIFT DETECTION & AUTO-RETRAINING")
    print("==================================================================")

    # Step 1: Check initial registered version
    initial_version = get_latest_registry_version(MODEL_NAME)
    print(f"Step 1: Current registered version of '{MODEL_NAME}' in MLflow: v{initial_version}")

    # Step 2: Inject synthetic drift into live inference service
    print("\nStep 2: Simulating production drift via simulate_production_drift()...")
    sent = simulate_production_drift(n_samples=40, inference_url=INFERENCE_URL, batch_size=10)
    assert sent >= 30, f"Expected at least 30 drifted requests, sent {sent}"
    time.sleep(1)

    # Step 3: Run drift worker evaluation cycle
    print("\nStep 3: Executing drift worker monitoring cycle...")
    drift_res = run_drift_cycle()

    print("\nDrift Worker Output:")
    print("  Status:", drift_res.get("status"))
    print("  Drift Detected:", drift_res.get("drift_detected"))
    print("  Drift Share:", f"{drift_res.get('drift_share', 0):.1%}")
    print("  Trigger Reason:", drift_res.get("trigger_reason"))
    print("  High PSI Features:", drift_res.get("high_psi_features"))

    # Assertions on drift detection
    assert drift_res.get("status") == "success", f"Drift cycle failed: {drift_res}"
    assert drift_res.get("drift_detected") is True, "Expected drift_detected to be True!"

    # Verify Evidently HTML report artifact
    report_path = drift_res.get("report_path")
    assert report_path and os.path.exists(report_path), f"Drift report missing at {report_path}"
    assert os.path.getsize(report_path) > 10000, f"Drift report unexpectedly small: {os.path.getsize(report_path)} bytes"
    print(f"Evidently Drift HTML Report: {report_path} ({os.path.getsize(report_path)/1024:.1f} KB)")

    # Step 4: Verify auto-retraining triggered and new version registered
    time.sleep(2)
    new_version = get_latest_registry_version(MODEL_NAME)
    print(f"\nStep 4: Checking MLflow Model Registry...")
    print(f"  Initial Version: v{initial_version}")
    print(f"  New Version:     v{new_version}")
    assert new_version > initial_version, f"Expected new_version ({new_version}) > initial_version ({initial_version})"

    # Step 5: Verify inference service hot-swapped to new version
    inf_health = requests.get(f"{INFERENCE_URL}/health", timeout=5).json()
    print(f"Inference Service Active Model Version: v{inf_health.get('model_version')}")
    assert str(inf_health.get("model_version")) == str(new_version), (
        f"Inference service model_version ({inf_health.get('model_version')}) does not match new_version ({new_version})"
    )

    print("\n>>> PHASE 6 VERIFICATION PASSED: Drift detected, retrain auto-triggered, new model version v"
          f"{new_version} registered and deployed! <<<")


if __name__ == "__main__":
    verify_phase6()
