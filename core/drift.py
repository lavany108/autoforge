"""
AutoForge Data Drift Detection Engine
Compares production inference requests against the training reference dataset using
Evidently and Population Stability Index (PSI).
Trigger rules:
- Drift share > 30% of features OR
- Any feature PSI > 0.20
"""

import os
import json
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from core.logging_config import logger
from inference.request_logger import RequestLogger


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current feature distributions.
    PSI < 0.1: No shift
    0.1 <= PSI <= 0.2: Moderate shift
    PSI > 0.2: Significant distribution shift (triggers retraining)
    """
    exp_clean = expected[~np.isnan(expected)] if expected.dtype.kind in "fc" else expected
    act_clean = actual[~np.isnan(actual)] if actual.dtype.kind in "fc" else actual

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    try:
        # Determine quantile bins from expected distribution
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(exp_clean, quantiles)
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        exp_counts, _ = np.histogram(exp_clean, bins=bins)
        act_counts, _ = np.histogram(act_clean, bins=bins)

        # Convert to percentages
        exp_pct = exp_counts / max(1, len(exp_clean))
        act_pct = act_counts / max(1, len(act_clean))

        # Add epsilon to prevent log(0) or div by 0
        exp_pct = np.clip(exp_pct, epsilon, 1.0)
        act_pct = np.clip(act_pct, epsilon, 1.0)

        psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(round(psi_val, 4))
    except Exception:
        return 0.0


def check_and_evaluate_drift(
    reference_path: Optional[str] = None,
    artifacts_dir: str = "artifacts",
    min_requests_required: int = 10,
    drift_share_threshold: float = 0.30,
    psi_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Compares recent inference logs from SQLite with reference dataset snapshot.
    Generates Evidently HTML report and evaluates trigger rules.
    """
    ref_file = reference_path or os.path.join(artifacts_dir, "reference_data.csv")
    if not os.path.exists(ref_file):
        # Fallback to seed dataset if reference snapshot not saved yet
        ref_file = "data/titanic.csv"

    if not os.path.exists(ref_file):
        return {"status": "error", "message": f"Reference dataset not found at {ref_file}"}

    ref_df = pd.read_csv(ref_file)
    logger.info(f"Loaded reference dataset from {ref_file} ({len(ref_df)} rows)")

    # Load recent requests from SQLite
    req_logger = RequestLogger(os.path.join(artifacts_dir, "autoforge.db"))
    curr_df = req_logger.get_recent_requests(limit=500)

    if len(curr_df) < min_requests_required:
        msg = f"Insufficient inference requests for drift check: {len(curr_df)} logged, minimum {min_requests_required} required."
        logger.info(msg)
        return {
            "status": "skipped",
            "message": msg,
            "requests_logged": len(curr_df),
            "min_required": min_requests_required,
            "drift_detected": False,
        }

    # Find common feature columns (exclude target and metadata)
    exclude_cols = {"_prediction", "_probability", "_timestamp", "Survived", "PassengerId", "target"}
    feature_cols = [c for c in ref_df.columns if c in curr_df.columns and c not in exclude_cols]

    if not feature_cols:
        return {"status": "error", "message": "No common feature columns between reference and inference data."}

    logger.info(f"Evaluating drift across {len(feature_cols)} features: {feature_cols}")

    ref_subset = ref_df[feature_cols].copy()
    curr_subset = curr_df[feature_cols].copy()

    # Align data types
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(ref_subset[col]):
            curr_subset[col] = pd.to_numeric(curr_subset[col], errors="coerce")

    # 1. Compute Feature PSIs
    feature_psis = {}
    high_psi_features = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(ref_subset[col]):
            psi_score = calculate_psi(ref_subset[col].dropna().values, curr_subset[col].dropna().values)
            feature_psis[col] = psi_score
            if psi_score > psi_threshold:
                high_psi_features.append({"feature": col, "psi": psi_score})

    # 2. Run Evidently Data Drift Preset Report
    reports_dir = os.path.join(artifacts_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_html_path = os.path.join(reports_dir, "drift_report.html")

    drift_share = 0.0
    num_drifted = 0

    try:
        logger.info("Executing Evidently DataDriftPreset report...")
        drift_report = Report(metrics=[DataDriftPreset()])
        snapshot = drift_report.run(reference_data=ref_subset, current_data=curr_subset)
        snapshot.save_html(report_html_path)
        logger.info(f"Saved Evidently HTML drift report to {report_html_path}")

        # Extract drift metrics from snapshot
        for k, v in snapshot.metric_results.items():
            if "Count of Drifted Columns" in getattr(v, "display_name", ""):
                metric_share = getattr(v, "share", None)
                if metric_share is not None and hasattr(metric_share, "value"):
                    drift_share = float(metric_share.value)
                elif hasattr(v, "value") and v.value is not None:
                    drift_share = float(v.value)
            elif "Drifted" in getattr(v, "display_name", "") and getattr(v, "value", None) is not None:
                drift_share = max(drift_share, float(v.value))
    except Exception as e:
        logger.error(f"Evidently report execution error: {e}", exc_info=True)
        # Fallback drift share calculation from PSI
        num_drifted = len(high_psi_features)
        drift_share = round(num_drifted / max(1, len(feature_cols)), 4)

    # If Evidently didn't parse drift_share directly, compute from PSI
    if drift_share == 0.0 and high_psi_features:
        drift_share = round(len(high_psi_features) / len(feature_cols), 4)

    # 3. Evaluate Trigger Rules
    # Rule 1: drift share > 30% of columns
    # Rule 2: any feature PSI > 0.20
    is_drift_share_triggered = drift_share > drift_share_threshold
    is_psi_triggered = len(high_psi_features) > 0
    drift_detected = is_drift_share_triggered or is_psi_triggered

    trigger_reasons = []
    if is_drift_share_triggered:
        trigger_reasons.append(f"Drift share {drift_share:.1%} exceeded threshold {drift_share_threshold:.1%}")
    if is_psi_triggered:
        feats_str = ", ".join([f"{f['feature']} (PSI={f['psi']:.3f})" for f in high_psi_features])
        trigger_reasons.append(f"Features exceeded PSI threshold {psi_threshold}: {feats_str}")

    reason_str = " | ".join(trigger_reasons) if drift_detected else "No significant drift detected."

    if drift_detected:
        logger.warning(f"🚨 DRIFT ALERT TRIGGERED: {reason_str}")
    else:
        logger.info(f"✅ Drift check healthy (drift share: {drift_share:.1%})")

    result = {
        "status": "success",
        "timestamp": time.time(),
        "drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "drift_share_threshold": drift_share_threshold,
        "psi_threshold": psi_threshold,
        "high_psi_features": high_psi_features,
        "feature_psis": feature_psis,
        "trigger_reason": reason_str,
        "report_path": report_html_path,
        "report_url": "/artifacts/reports/drift_report.html",
        "requests_analyzed": len(curr_df),
        "reference_rows": len(ref_df),
    }

    # Save metrics JSON record
    metrics_path = os.path.join(reports_dir, "drift_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
