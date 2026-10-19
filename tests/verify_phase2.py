"""
Phase 2 Verification Script
Tests end-to-end training engine:
1. ColumnTransformer preprocessing + model pipeline
2. FLAML search + Optuna refinement
3. Baseline and tuned runs logged in MLflow
4. Tuned >= baseline accuracy verification
5. Full pipeline pickling and model registration in MLflow
"""

import os
import sys
import pickle
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


def verify_phase2():
    client = TestClient(app)

    train_payload = {
        "file_path": "data/titanic.csv",
        "target_column": "Survived",
        "task_type": "classification",
        "time_budget_secs": 15,
        "optuna_trials": 10,
        "model_name": "autoforge_titanic",
        "experiment_name": "autoforge_experiments",
    }

    print("Launching Phase 2 training via POST /training/train...")
    resp = client.post("/training/train", json=train_payload)
    assert resp.status_code == 200, f"Training API failed ({resp.status_code}): {resp.text}"

    data = resp.json()
    print("Training API response received:")
    print("  Status:", data.get("status"))
    print("  Baseline Run ID:", data.get("baseline_run_id"))
    print("  Tuned Run ID:", data.get("tuned_run_id"))
    print("  Registered Model Name:", data.get("registered_model_name"))
    print("  Registered Version:", data.get("registered_version"))
    print("  FLAML Winner:", data.get("flaml_winner"))
    print("  Baseline Metrics:", data.get("baseline_metrics"))
    print("  Tuned Metrics:", data.get("tuned_metrics"))
    print("  Uplift:", data.get("uplift"))

    # Assertions
    assert data.get("status") == "success"
    baseline_id = data.get("baseline_run_id")
    tuned_id = data.get("tuned_run_id")
    assert baseline_id and tuned_id, "Both baseline and tuned run IDs must be present"
    assert baseline_id != tuned_id, "Baseline and tuned run IDs must be distinct"

    base_acc = data["baseline_metrics"]["accuracy"]
    tuned_acc = data["tuned_metrics"]["accuracy"]
    print(f"\nAccuracy Comparison: Tuned ({tuned_acc}) vs Baseline ({base_acc})")
    assert tuned_acc >= base_acc, f"Tuned accuracy ({tuned_acc}) must be >= Baseline ({base_acc})"

    # Verify pipeline artifact exists and can be loaded
    pipeline_path = data.get("pipeline_artifact_path")
    assert os.path.exists(pipeline_path), f"Pipeline artifact missing at {pipeline_path}"
    with open(pipeline_path, "rb") as f:
        loaded_pipeline = pickle.load(f)
    print("Loaded pickled pipeline successfully:", type(loaded_pipeline))

    # Test prediction on a sample
    df_sample = pd.read_csv("data/titanic.csv").head(3)
    preds = loaded_pipeline.predict(df_sample)
    print(f"Sample predictions from loaded pipeline: {preds}")
    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"

    # Verify MLflow runs query endpoint
    runs_resp = client.get("/runs?experiment_name=autoforge_experiments")
    assert runs_resp.status_code == 200, f"Failed to list runs: {runs_resp.text}"
    runs_list = runs_resp.json().get("runs", [])
    run_ids = [r["run_id"] for r in runs_list]
    print(f"MLflow Runs in experiment ({len(run_ids)}): {run_ids}")
    assert baseline_id in run_ids, f"Baseline run {baseline_id} missing from MLflow runs"
    assert tuned_id in run_ids, f"Tuned run {tuned_id} missing from MLflow runs"

    # Verify MLflow registered models endpoint
    models_resp = client.get("/models")
    assert models_resp.status_code == 200, f"Failed to list models: {models_resp.text}"
    models_list = models_resp.json().get("registered_models", [])
    model_names = [m["name"] for m in models_list]
    print(f"Registered Models in MLflow: {model_names}")
    assert "autoforge_titanic" in model_names, "autoforge_titanic missing from registered models"

    print("\n>>> PHASE 2 VERIFICATION PASSED: Baseline + Tuned runs logged, tuned >= baseline accuracy, model registered <<<")


if __name__ == "__main__":
    verify_phase2()
