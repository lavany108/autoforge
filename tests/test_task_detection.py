"""
Unit Tests for Task Detection Heuristic
"""

import pytest
import pandas as pd
import numpy as np
from core.task_detection import detect_task_type


def test_classification_object_target():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "target": ["cat", "dog", "cat", "dog"],
    })
    res = detect_task_type(df, "target")
    assert res["task_type"] == "classification"
    assert res["confidence"] >= 0.95
    assert res["is_binary"] is True


def test_classification_discrete_integer_target():
    # <= 20 unique values and not float
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [0, 1, 0, 1, 0],
    })
    res = detect_task_type(df, "target")
    assert res["task_type"] == "classification"
    assert res["confidence"] >= 0.95
    assert res["n_unique"] == 2


def test_regression_float_target():
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": range(100),
        "target": np.random.uniform(10.0, 100.0, 100),
    })
    res = detect_task_type(df, "target")
    assert res["task_type"] == "regression"
    assert res["confidence"] >= 0.90
    assert res["n_unique"] > 20


def test_regression_high_cardinality_integer():
    df = pd.DataFrame({
        "feature1": range(150),
        "target": list(range(150)),
    })
    res = detect_task_type(df, "target")
    assert res["task_type"] == "regression"
    assert res["n_unique"] == 150


def test_invalid_target_column():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        detect_task_type(df, "nonexistent")
