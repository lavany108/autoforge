"""
Phase 3 Verification Script
Tests model explainability at training time:
1. Computes SHAP values (TreeExplainer / KernelExplainer fallback)
2. Generates SHAP summary plot PNG
3. Generates interactive Shapash HTML report
4. Saves explainability artifacts to MLflow
5. Verifies API endpoint GET /runs/{run_id}/explanation and static file serving
"""

import os
import sys
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


def verify_phase3():
    client = TestClient(app)

    train_payload = {
        "file_path": "data/titanic.csv",
        "target_column": "Survived",
        "task_type": "classification",
        "time_budget_secs": 10,
        "optuna_trials": 5,
        "model_name": "autoforge_explain_model",
        "experiment_name": "autoforge_experiments",
    }

    print("Step 1: Launching training run with explainability...")
    resp = client.post("/training/train", json=train_payload)
    assert resp.status_code == 200, f"Training failed: {resp.text}"

    data = resp.json()
    tuned_run_id = data.get("tuned_run_id")
    assert tuned_run_id, "tuned_run_id is missing from response"

    explanation = data.get("explanation")
    assert explanation is not None, "explanation metadata missing from train response"
    print("Explainability metadata returned from training:")
    print("  Run ID:", explanation.get("run_id"))
    print("  SHAP PNG:", explanation.get("shap_summary_png"))
    print("  Shapash HTML:", explanation.get("shapash_html_report"))
    print("  Top Features:", explanation.get("top_features")[:5])

    # Step 2: Verify disk artifacts
    png_path = explanation["shap_summary_png"]
    html_path = explanation["shapash_html_report"]
    assert os.path.exists(png_path), f"SHAP summary PNG does not exist at {png_path}"
    assert os.path.getsize(png_path) > 1000, f"SHAP summary PNG too small ({os.path.getsize(png_path)} bytes)"

    assert os.path.exists(html_path), f"Shapash HTML report does not exist at {html_path}"
    assert os.path.getsize(html_path) > 1000, f"Shapash HTML report too small ({os.path.getsize(html_path)} bytes)"

    # Step 3: Test GET /runs/{run_id}/explanation endpoint
    print(f"\nStep 3: Querying GET /runs/{tuned_run_id}/explanation...")
    exp_resp = client.get(f"/runs/{tuned_run_id}/explanation")
    assert exp_resp.status_code == 200, f"Failed to retrieve explanation: {exp_resp.text}"

    exp_data = exp_resp.json()
    assert exp_data.get("run_id") == tuned_run_id
    assert exp_data.get("shap_summary_url") is not None
    assert exp_data.get("shapash_html_url") is not None
    top_feats = exp_data.get("top_features", [])
    assert len(top_feats) > 0, "Top features list is empty"
    print(f"Top 5 features for run {tuned_run_id}:")
    for f in top_feats[:5]:
        print(f"   - {f['feature']}: {f['importance']}")

    # Step 4: Verify static file serving for browser/UI viewing
    png_url = exp_data["shap_summary_url"]
    html_url = exp_data["shapash_html_url"]

    static_png = client.get(png_url)
    assert static_png.status_code == 200, f"Static serving for PNG failed at {png_url}"
    assert static_png.headers.get("content-type", "").startswith("image/png")

    static_html = client.get(html_url)
    assert static_html.status_code == 200, f"Static serving for HTML failed at {html_url}"
    assert "html" in static_html.headers.get("content-type", "")

    print(f"\nSuccessfully verified static serving for browser:")
    print(f"  PNG URL:  {png_url} (HTTP {static_png.status_code})")
    print(f"  HTML URL: {html_url} (HTTP {static_html.status_code})")

    print("\n>>> PHASE 3 VERIFICATION PASSED: SHAP summary PNG + Shapash HTML generated, top features returned, API endpoint verified <<<")


if __name__ == "__main__":
    verify_phase3()
