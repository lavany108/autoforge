"""
Phase 4 Verification Script
Tests FastAPI Inference Microservice:
1. GET /health
2. POST /model/load (dynamic loading of registered model / artifact pipeline)
3. POST /predict (rows -> predictions + probabilities + latency)
4. Request logging to SQLite store for drift tracking
5. Curl / HTTP client verification with realistic rows
"""

import os
import sys
import json
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.main import app, request_logger, model_holder


def verify_phase4():
    client = TestClient(app)

    # 1. Health check before load
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    print("Initial Health Check:", health_resp.json())

    # 2. Deploy / Load model
    latest_pipeline = "artifacts/models/latest_pipeline.pkl"
    assert os.path.exists(latest_pipeline), f"Pipeline file missing at {latest_pipeline}. Run Phase 2/3 first."

    print(f"\nStep 2: Deploying model via POST /model/load from {latest_pipeline}...")
    load_resp = client.post(
        "/model/load",
        json={"pipeline_path": latest_pipeline, "model_name": "autoforge_titanic", "version": "1"},
    )
    assert load_resp.status_code == 200, f"Model load failed ({load_resp.status_code}): {load_resp.text}"
    print("Model Load Response:", load_resp.json())

    # Health check after load
    health_after = client.get("/health").json()
    assert health_after["model_loaded"] is True
    print("Health Check after load:", health_after)

    # 3. Predict with real Titanic rows
    df_titanic = pd.read_csv("data/titanic.csv")
    df_no_target = df_titanic.drop(columns=["Survived"]).head(4)
    df_clean = df_no_target.where(pd.notnull(df_no_target), None)
    test_samples = df_clean.to_dict(orient="records")

    payload = {"rows": test_samples}
    print(f"\nStep 3: Sending POST /predict with {len(test_samples)} rows...")
    pred_resp = client.post("/predict", json=payload)
    assert pred_resp.status_code == 200, f"Predict failed ({pred_resp.status_code}): {pred_resp.text}"

    pred_data = pred_resp.json()
    print("Prediction Response:")
    print("  Status:", pred_data.get("status"))
    print("  Model:", pred_data.get("model_name"), "v" + str(pred_data.get("model_version")))
    print("  Count:", pred_data.get("count"))
    print("  Predictions:", pred_data.get("predictions"))
    print("  Probabilities:", pred_data.get("probabilities"))
    print(f"  Latency: {pred_data.get('latency_ms')} ms")

    assert pred_data.get("status") == "success"
    assert pred_data.get("count") == 4
    preds = pred_data.get("predictions")
    assert len(preds) == 4
    for p in preds:
        assert p in [0, 1], f"Unexpected prediction value: {p}"

    probs = pred_data.get("probabilities")
    assert probs is not None
    assert len(probs) == 4
    for row_p in probs:
        assert len(row_p) == 2
        assert abs(sum(row_p) - 1.0) < 1e-4

    assert pred_data.get("latency_ms") > 0

    # 4. Check SQLite Request Logging
    recent_df = request_logger.get_recent_requests(10)
    print(f"\nStep 4: Verified {len(recent_df)} recent requests logged in SQLite.")
    assert len(recent_df) >= 4, f"Expected at least 4 requests logged, got {len(recent_df)}"
    assert "_prediction" in recent_df.columns
    assert "_timestamp" in recent_df.columns

    print("\n>>> PHASE 4 VERIFICATION PASSED: Model loaded, /predict responded with valid predictions + probabilities, latency logged, SQLite populated <<<")


if __name__ == "__main__":
    verify_phase4()
