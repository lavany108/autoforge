"""
Unit Tests for FastAPI Prediction Microservice Endpoint
"""

import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from inference.main import app, model_holder


from sklearn.linear_model import LogisticRegression
from core.pipeline_builder import build_preprocessor, build_full_pipeline


@pytest.fixture(scope="module")
def client():
    # Save any currently loaded pipeline
    saved_pipeline = model_holder.pipeline
    saved_name = model_holder.model_name
    saved_version = model_holder.model_version
    saved_source = model_holder.source

    # Train a minimal, deterministic pipeline for unit testing
    df_train = pd.DataFrame({
        "age": [20.0, 30.0, 40.0, 50.0],
        "fare": [10.0, 20.0, 30.0, 40.0],
        "sex": ["male", "female", "female", "male"]
    })
    y_train = [0, 1, 1, 0]

    prep = build_preprocessor(
        num_cols=["age", "fare"],
        cat_cols=["sex"],
    )
    test_pipe = build_full_pipeline(preprocessor=prep, estimator=LogisticRegression())
    test_pipe.fit(df_train, y_train)

    model_holder.pipeline = test_pipe
    model_holder.model_name = "test_classifier"
    model_holder.model_version = "1"
    model_holder.source = "fixture:memory"

    yield TestClient(app)

    # Restore prior pipeline if any
    model_holder.pipeline = saved_pipeline
    model_holder.model_name = saved_name
    model_holder.model_version = saved_version
    model_holder.source = saved_source


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "autoforge-inference"


def test_predict_endpoint_valid_payload(client):
    rows = [
        {"age": 25.0, "fare": 15.0, "sex": "male"},
        {"age": 35.0, "fare": 25.0, "sex": "female"},
    ]
    resp = client.post("/predict", json={"rows": rows})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["latency_ms"] > 0
    assert data["probabilities"] is not None
    assert len(data["probabilities"]) == 2


def test_predict_endpoint_empty_payload(client):
    resp = client.post("/predict", json={"rows": []})
    assert resp.status_code == 400
