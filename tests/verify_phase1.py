"""
Phase 1 Verification Script
Tests dataset loading via API, ydata-profiling HTML generation,
and task detection heuristic (classification on titanic.csv with target=Survived).
"""

import os
import sys
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


def verify_phase1():
    client = TestClient(app)

    # 1. Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    print("Health check OK:", health_resp.json())

    # 2. List datasets
    list_resp = client.get("/datasets")
    assert list_resp.status_code == 200, f"List datasets failed: {list_resp.text}"
    datasets = list_resp.json().get("datasets", [])
    print(f"Discovered {len(datasets)} datasets:", [d["name"] for d in datasets])

    # 3. Profile Titanic dataset with target='Survived'
    titanic_path = "data/titanic.csv"
    assert os.path.exists(titanic_path), f"Titanic dataset not found at {titanic_path}"

    payload = {
        "file_path": titanic_path,
        "dataset_name": "titanic",
        "target_column": "Survived",
        "minimal": True,  # Minimal mode for fast verification
    }
    print("Calling POST /datasets/profile with Titanic dataset...")
    resp = client.post("/datasets/profile", json=payload)
    assert resp.status_code == 200, f"Profiling failed ({resp.status_code}): {resp.text}"

    data = resp.json()
    print("Profiling Response Status:", data.get("status"))

    task_detection = data.get("task_detection")
    assert task_detection is not None, "task_detection missing from response"
    task_type = task_detection.get("task_type")
    confidence = task_detection.get("confidence")
    reasoning = task_detection.get("reasoning")

    print(f"Detected Task: {task_type} (confidence={confidence})")
    print(f"Reasoning: {reasoning}")
    assert task_type == "classification", f"Expected task=classification, got {task_type}"
    assert confidence >= 0.85, f"Expected confidence >= 0.85, got {confidence}"

    # 4. Check HTML profiling report exists and has non-trivial size
    profiling_meta = data.get("profiling")
    assert profiling_meta is not None, "profiling metadata missing from response"
    report_path = profiling_meta.get("report_path")
    assert os.path.exists(report_path), f"Report HTML file does not exist at {report_path}"
    file_size_kb = os.path.getsize(report_path) / 1024
    print(f"Report HTML generated successfully: {report_path} ({file_size_kb:.2f} KB)")
    assert file_size_kb > 50, f"Report file unexpectedly small: {file_size_kb} KB"

    # 5. Verify regression detection heuristic on continuous target 'Fare'
    fare_detect = client.post("/task/detect", json={"file_path": titanic_path, "target_column": "Fare"}).json()
    print(f"Continuous Target 'Fare' Detected Task: {fare_detect.get('task_type')} (confidence={fare_detect.get('confidence')})")
    assert fare_detect.get("task_type") == "regression", f"Expected regression for Fare, got {fare_detect.get('task_type')}"

    print("\n>>> PHASE 1 VERIFICATION PASSED: titanic.csv profiled with task=classification <<<")


if __name__ == "__main__":
    verify_phase1()
