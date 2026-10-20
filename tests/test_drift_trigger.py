"""
Unit Tests for Drift Calculation & Trigger Rules
"""

import pytest
import numpy as np
from core.drift import calculate_psi, check_and_evaluate_drift


def test_psi_identical_distributions():
    np.random.seed(42)
    data = np.random.normal(loc=50, scale=10, size=500)
    psi = calculate_psi(data, data)
    assert psi == 0.0 or psi < 0.02


def test_psi_shifted_distributions():
    np.random.seed(42)
    expected = np.random.normal(loc=30, scale=5, size=500)
    # Significant shift in mean
    actual = np.random.normal(loc=70, scale=5, size=500)
    psi = calculate_psi(expected, actual)
    assert psi > 0.20, f"Expected PSI > 0.20 for large distribution shift, got {psi}"


def test_drift_trigger_rules_evaluation():
    # If requests are insufficient, should skip safely
    res = check_and_evaluate_drift(min_requests_required=99999)
    assert res["status"] == "skipped"
    assert res["drift_detected"] is False
