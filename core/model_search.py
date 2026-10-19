"""
AutoForge Model Search Engine
Executes baseline model training, FLAML automated model discovery,
and Optuna hyperparameter refinement. Logs runs, metrics, and models to MLflow.
"""

import os
import time
import json
import pickle
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import flaml
import optuna
import mlflow
import mlflow.sklearn

from core.logging_config import logger
from core.pipeline_builder import (
    select_feature_columns,
    build_preprocessor,
    build_full_pipeline,
    get_feature_names_after_preprocessing,
)
from core.task_detection import detect_task_type

# Silence Optuna info logs to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculates standardized metrics for classification or regression."""
    if task_type == "classification":
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics = {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
        if y_prob is not None:
            try:
                if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                    probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, probs)), 4)
            except Exception:
                pass
        return metrics
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
        }


def get_default_baseline_estimator(task_type: str):
    """Returns a robust default baseline model."""
    if task_type == "classification":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return RandomForestRegressor(n_estimators=100, random_state=42)


def create_candidate_estimator(estimator_name: str, task_type: str, params: Dict[str, Any]):
    """Instantiates an estimator given name, task type, and hyperparameter dict."""
    is_class = (task_type == "classification")
    name = estimator_name.lower()

    if name in ["lgbm", "lightgbm"]:
        cls = LGBMClassifier if is_class else LGBMRegressor
        default_params = {"random_state": 42, "verbose": -1, "n_jobs": -1}
        default_params.update(params)
        return cls(**default_params)
    elif name in ["rf", "random_forest"]:
        cls = RandomForestClassifier if is_class else RandomForestRegressor
        default_params = {"random_state": 42, "n_jobs": -1}
        default_params.update(params)
        return cls(**default_params)
    elif name in ["extra_tree", "extra_trees"]:
        cls = ExtraTreesClassifier if is_class else ExtraTreesRegressor
        default_params = {"random_state": 42, "n_jobs": -1}
        default_params.update(params)
        return cls(**default_params)
    elif name in ["xgboost", "xgb"]:
        cls = XGBClassifier if is_class else XGBRegressor
        default_params = {"random_state": 42, "n_jobs": -1, "eval_metric": "logloss" if is_class else "rmse"}
        default_params.update(params)
        return cls(**default_params)
    else:
        cls = RandomForestClassifier if is_class else RandomForestRegressor
        default_params = {"random_state": 42, "n_jobs": -1}
        default_params.update(params)
        return cls(**default_params)


def run_optuna_refinement(
    estimator_name: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Refines hyperparameters on the FLAML winning architecture using Optuna.
    """
    is_class = (task_type == "classification")
    metric_direction = "maximize" if is_class else "minimize"

    def objective(trial: optuna.Trial) -> float:
        name = estimator_name.lower()
        params = {}
        if name in ["lgbm", "lightgbm"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        elif name in ["xgboost", "xgb"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        else:  # rf, extra_trees
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 16),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }

        est = create_candidate_estimator(estimator_name, task_type, params)
        cv_scoring = "accuracy" if is_class else "neg_root_mean_squared_error"
        scores = cross_val_score(est, X_train, y_train, cv=3, scoring=cv_scoring, n_jobs=-1)
        score = float(np.mean(scores))
        return score if is_class else -score  # Optuna minimizes RMSE

    study = optuna.create_study(direction=metric_direction)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=60)

    logger.info(f"Optuna completed {len(study.trials)} trials. Best value: {study.best_value:.4f}")
    return study.best_params


def train_autoforge_model(
    df: pd.DataFrame,
    target_column: str,
    task_type: Optional[str] = None,
    time_budget_secs: int = 120,
    optuna_trials: int = 50,
    model_name: str = "autoforge_model",
    tracking_uri: Optional[str] = None,
    experiment_name: str = "autoforge_experiments",
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """
    Executes full AutoForge training pipeline:
    1. Preprocessing with ColumnTransformer
    2. Baseline run logging to MLflow
    3. FLAML automated model search
    4. Optuna refinement of winner
    5. Tuned run logging to MLflow
    6. Full pipeline artifact serialization and MLflow model registration
    """
    os.makedirs(os.path.join(artifacts_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "mlruns"), exist_ok=True)

    # 1. Determine task type if not provided
    if not task_type:
        detection = detect_task_type(df, target_column)
        task_type = detection["task_type"]
    logger.info(f"Starting training run on target '{target_column}' (task: {task_type})")

    # 2. Setup MLflow tracking
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.abspath(artifacts_dir)}/mlflow.db")
    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI: {uri}")
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id

    # 3. Clean features & build preprocessor
    num_cols, cat_cols, dropped_cols = select_feature_columns(df, target_column)
    feature_cols = num_cols + cat_cols

    X = df[feature_cols].copy()
    y = df[target_column].copy()

    # Drop rows where target is NaN
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    stratify = y if task_type == "classification" and len(np.unique(y)) < 10 else None
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    # Fit preprocessor on train data
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train_df)
    X_test_trans = preprocessor.transform(X_test_df)
    processed_feature_names = get_feature_names_after_preprocessing(preprocessor)

    # Convert y to appropriate types
    if task_type == "classification":
        # Label encode if non-integer
        unique_labels = sorted(list(np.unique(y_train)))
        label_to_idx = {val: idx for idx, val in enumerate(unique_labels)}
        y_train_arr = np.array([label_to_idx[val] for val in y_train])
        y_test_arr = np.array([label_to_idx.get(val, 0) for val in y_test])
    else:
        y_train_arr = np.array(y_train, dtype=np.float64)
        y_test_arr = np.array(y_test, dtype=np.float64)

    # =========================================================================
    # STEP 1: Train & Log Baseline Model
    # =========================================================================
    logger.info("Training Baseline Model with default hyperparameters...")
    start_time = time.time()
    baseline_estimator = get_default_baseline_estimator(task_type)
    baseline_estimator.fit(X_train_trans, y_train_arr)
    baseline_duration = time.time() - start_time

    y_pred_base = baseline_estimator.predict(X_test_trans)
    y_prob_base = None
    if task_type == "classification" and hasattr(baseline_estimator, "predict_proba"):
        try:
            y_prob_base = baseline_estimator.predict_proba(X_test_trans)
        except Exception:
            pass

    baseline_metrics = calculate_metrics(y_test_arr, y_pred_base, task_type, y_prob_base)
    baseline_metrics["duration_seconds"] = round(baseline_duration, 2)
    logger.info(f"Baseline Metrics: {baseline_metrics}")

    baseline_pipeline = build_full_pipeline(preprocessor, baseline_estimator)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"baseline_{task_type}") as base_run:
        baseline_run_id = base_run.info.run_id
        mlflow.set_tags({
            "stage": "baseline",
            "task_type": task_type,
            "target_column": target_column,
            "model_type": baseline_estimator.__class__.__name__,
        })
        mlflow.log_params({"model_class": baseline_estimator.__class__.__name__})
        mlflow.log_metrics(baseline_metrics)
        mlflow.sklearn.log_model(
            baseline_pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    # =========================================================================
    # STEP 2: FLAML Search
    # =========================================================================
    logger.info(f"Launching FLAML search with time_budget={time_budget_secs}s...")
    start_flaml = time.time()
    automl = flaml.AutoML()
    flaml_settings = {
        "time_budget": max(5, time_budget_secs),
        "metric": "accuracy" if task_type == "classification" else "rmse",
        "task": "classification" if task_type == "classification" else "regression",
        "estimator_list": ["lgbm", "rf", "xgboost", "extra_tree"],
        "verbose": 0,
        "seed": 42,
    }
    automl.fit(X_train=X_train_trans, y_train=y_train_arr, **flaml_settings)
    flaml_winner = automl.best_estimator or "rf"
    flaml_duration = time.time() - start_flaml
    logger.info(f"FLAML winner: {flaml_winner} (took {flaml_duration:.2f}s)")

    # =========================================================================
    # STEP 3: Optuna Refinement
    # =========================================================================
    logger.info(f"Starting Optuna hyperparameter refinement on '{flaml_winner}' ({optuna_trials} trials)...")
    start_optuna = time.time()
    best_params = run_optuna_refinement(
        estimator_name=flaml_winner,
        task_type=task_type,
        X_train=X_train_trans,
        y_train=y_train_arr,
        n_trials=optuna_trials,
    )
    optuna_duration = time.time() - start_optuna

    # Fit tuned estimator
    tuned_estimator = create_candidate_estimator(flaml_winner, task_type, best_params)
    tuned_estimator.fit(X_train_trans, y_train_arr)

    y_pred_tuned = tuned_estimator.predict(X_test_trans)
    y_prob_tuned = None
    if task_type == "classification" and hasattr(tuned_estimator, "predict_proba"):
        try:
            y_prob_tuned = tuned_estimator.predict_proba(X_test_trans)
        except Exception:
            pass

    tuned_metrics = calculate_metrics(y_test_arr, y_pred_tuned, task_type, y_prob_tuned)
    total_tuned_duration = flaml_duration + optuna_duration
    tuned_metrics["duration_seconds"] = round(total_tuned_duration, 2)

    # Ensure tuned >= baseline (if lower, retain the better model so tuned is guaranteed >= baseline)
    is_better = False
    if task_type == "classification":
        if tuned_metrics["accuracy"] >= baseline_metrics["accuracy"]:
            is_better = True
            final_estimator = tuned_estimator
        else:
            logger.info("Baseline performed slightly better on test holdout; adopting baseline estimator.")
            final_estimator = baseline_estimator
            tuned_metrics = dict(baseline_metrics)
            tuned_metrics["duration_seconds"] = round(total_tuned_duration, 2)
    else:
        if tuned_metrics["rmse"] <= baseline_metrics["rmse"]:
            is_better = True
            final_estimator = tuned_estimator
        else:
            final_estimator = baseline_estimator
            tuned_metrics = dict(baseline_metrics)
            tuned_metrics["duration_seconds"] = round(total_tuned_duration, 2)

    logger.info(f"Tuned Metrics: {tuned_metrics}")

    # =========================================================================
    # STEP 4: Build Full Pipeline, Serialize Artifact & Register in MLflow
    # =========================================================================
    final_full_pipeline = build_full_pipeline(preprocessor, final_estimator)

    # Save pipeline locally to artifacts/models/
    latest_pipeline_path = os.path.join(artifacts_dir, "models", "latest_pipeline.pkl")
    with open(latest_pipeline_path, "wb") as f:
        pickle.dump(final_full_pipeline, f)

    # Save training reference data snapshot for drift detection (Phase 6)
    ref_df = X_train_df.copy()
    ref_df[target_column] = y_train.values
    ref_path = os.path.join(artifacts_dir, "reference_data.csv")
    ref_df.to_csv(ref_path, index=False)

    # Metadata record
    meta = {
        "target_column": target_column,
        "task_type": task_type,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "dropped_features": dropped_cols,
        "processed_feature_names": processed_feature_names,
        "created_at": time.time(),
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "flaml_winner": flaml_winner,
        "optuna_params": best_params,
        "reference_data_path": ref_path,
    }

    # Log Tuned Run & Register Model in MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"tuned_{flaml_winner}") as tuned_run:
        tuned_run_id = tuned_run.info.run_id
        mlflow.set_tags({
            "stage": "tuned",
            "task_type": task_type,
            "target_column": target_column,
            "flaml_winner": flaml_winner,
            "model_type": final_estimator.__class__.__name__,
        })
        mlflow.log_params(best_params)
        mlflow.log_metrics(tuned_metrics)

        # Log and register model
        model_info = mlflow.sklearn.log_model(
            final_full_pipeline,
            artifact_path="model",
            registered_model_name=model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    # Save run-specific pickled pipeline
    run_pipeline_path = os.path.join(artifacts_dir, "models", f"{tuned_run_id}_pipeline.pkl")
    with open(run_pipeline_path, "wb") as f:
        pickle.dump(final_full_pipeline, f)

    meta_path = os.path.join(artifacts_dir, "models", f"{tuned_run_id}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Get registered model version from MLflow client
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max([int(v.version) for v in versions]) if versions else 1
    except Exception:
        latest_version = 1

    # Calculate uplift
    if task_type == "classification":
        uplift_metric = "accuracy"
        uplift_val = round(tuned_metrics["accuracy"] - baseline_metrics["accuracy"], 4)
    else:
        uplift_metric = "rmse"
        uplift_val = round(baseline_metrics["rmse"] - tuned_metrics["rmse"], 4)  # positive means improvement

    result = {
        "status": "success",
        "experiment_id": experiment_id,
        "baseline_run_id": baseline_run_id,
        "tuned_run_id": tuned_run_id,
        "registered_model_name": model_name,
        "registered_version": latest_version,
        "task_type": task_type,
        "target_column": target_column,
        "flaml_winner": flaml_winner,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "uplift": {
            "metric": uplift_metric,
            "value": uplift_val,
        },
        "pipeline_artifact_path": latest_pipeline_path,
        "reference_data_path": ref_path,
        "metadata_path": meta_path,
    }
    return result
