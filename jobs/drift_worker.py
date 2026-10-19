"""
AutoForge Scheduled Drift Worker
Monitors production inference traffic from SQLite against reference data.
Trigger rules:
- Drift share > 30% of features OR
- Any feature PSI > 0.20
Actions on Trigger:
1. Generates Evidently HTML report + JSON alert
2. Auto-triggers new training run via the API orchestrator
3. Registers the new model version in MLflow
4. Hot-swaps the deployed model on the Inference service
"""

import os
import sys
import time
import json
import argparse
import requests

from core.logging_config import logger
from core.drift import check_and_evaluate_drift

API_URL = os.getenv("API_URL", "http://localhost:8000")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
INTERVAL_SECS = int(os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", "300"))
RETRAIN_EVENTS_FILE = os.path.join(ARTIFACTS_DIR, "retrain_events.json")


def record_retrain_event(event: dict):
    events = []
    if os.path.exists(RETRAIN_EVENTS_FILE):
        try:
            with open(RETRAIN_EVENTS_FILE, "r") as f:
                events = json.load(f)
        except Exception:
            events = []
    events.append(event)
    with open(RETRAIN_EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)


def trigger_retraining_via_api(drift_result: dict) -> dict:
    """Invokes API endpoint to retrain model and hot-swap inference service."""
    logger.info("Auto-retraining triggered! Contacting API orchestrator...")
    train_payload = {
        "file_path": "data/titanic.csv",
        "target_column": "Survived",
        "time_budget_secs": 20,
        "optuna_trials": 10,
        "model_name": "autoforge_titanic",
        "experiment_name": "autoforge_experiments",
    }

    try:
        r = requests.post(f"{API_URL}/training/train", json=train_payload, timeout=600)
        if r.status_code == 200:
            retrain_res = r.json()
            new_version = retrain_res.get("registered_version")
            model_name = retrain_res.get("registered_model_name")
            logger.info(f"Retraining complete! Registered model '{model_name}' version {new_version}")

            # Hot-swap deployed model in Inference Service
            try:
                dep_res = requests.post(
                    f"{INFERENCE_URL}/model/load",
                    json={"model_name": model_name, "version": str(new_version)},
                    timeout=30,
                )
                if dep_res.status_code == 200:
                    logger.info(f"Successfully hot-swapped Inference service to version {new_version}")
                else:
                    logger.warning(f"Inference service update returned: {dep_res.text}")
            except Exception as inf_err:
                logger.warning(f"Could not contact inference service to hot-swap: {inf_err}")

            event = {
                "timestamp": time.time(),
                "trigger_reason": drift_result.get("trigger_reason"),
                "drift_share": drift_result.get("drift_share"),
                "new_version": new_version,
                "model_name": model_name,
                "tuned_metrics": retrain_res.get("tuned_metrics"),
            }
            record_retrain_event(event)
            return retrain_res
        else:
            logger.error(f"API retraining request failed ({r.status_code}): {r.text}")
            return {"status": "error", "detail": r.text}
    except Exception as e:
        logger.error(f"Exception during auto-retraining trigger: {e}", exc_info=True)
        # Fallback local training if API unavailable
        from core.model_search import train_autoforge_model
        import pandas as pd
        df = pd.read_csv("data/titanic.csv")
        retrain_res = train_autoforge_model(
            df=df,
            target_column="Survived",
            time_budget_secs=20,
            optuna_trials=10,
            model_name="autoforge_titanic",
        )
        return retrain_res


def run_drift_cycle() -> dict:
    """Executes a single drift evaluation cycle and triggers retraining if needed."""
    logger.info("Running drift monitoring evaluation cycle...")
    result = check_and_evaluate_drift(
        reference_path=os.path.join(ARTIFACTS_DIR, "reference_data.csv"),
        artifacts_dir=ARTIFACTS_DIR,
        min_requests_required=5,
        drift_share_threshold=0.30,
        psi_threshold=0.20,
    )

    if result.get("drift_detected"):
        logger.warning(f"🚨 DRIFT ALERT: {result.get('trigger_reason')}")
        retrain_result = trigger_retraining_via_api(result)
        result["retrain_result"] = retrain_result
    else:
        logger.info(f"Drift status: {result.get('message', 'Healthy')}")

    return result


def main():
    parser = argparse.ArgumentParser(description="AutoForge Drift Monitoring Worker")
    parser.add_argument("--once", action="store_true", help="Run once and exit (for verification/testing)")
    parser.add_argument("--interval", type=int, default=INTERVAL_SECS, help="Check interval in seconds")
    args = parser.parse_args()

    if args.once:
        logger.info("Executing single drift evaluation pass (--once)...")
        res = run_drift_cycle()
        print(json.dumps(res, indent=2, default=str))
        sys.exit(0)

    logger.info(f"AutoForge Drift Worker started (interval: {args.interval}s, API: {API_URL})")
    while True:
        try:
            run_drift_cycle()
        except Exception as e:
            logger.error(f"Drift worker error: {e}", exc_info=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
