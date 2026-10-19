"""
AutoForge Explainability Engine
Generates model interpretability artifacts at training time:
- Computes SHAP values (TreeExplainer with KernelExplainer fallback)
- Produces SHAP summary plot PNG
- Generates interactive Shapash HTML report
- Computes global top feature importance
- Logs explainability artifacts to MLflow
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from shapash import SmartExplainer
import mlflow

from core.logging_config import logger
from core.pipeline_builder import get_feature_names_after_preprocessing


def compute_shap_and_reports(
    pipeline: Any,
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_column: str,
    task_type: str,
    run_id: str,
    model_name: str = "autoforge_model",
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """
    Computes SHAP values and generates Shapash + SHAP visual artifacts at training time.
    NEVER invoked inside the live inference request path.
    """
    reports_dir = os.path.join(artifacts_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    # Transform test features
    X_test_trans = preprocessor.transform(X_test_df)
    feature_names = get_feature_names_after_preprocessing(preprocessor)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names)

    # Use a sample for SHAP if test set is large
    sample_size = min(len(X_test_trans_df), 200)
    X_sample = X_test_trans_df.iloc[:sample_size].copy()

    # 1. Compute SHAP Values (TreeExplainer with fallback)
    logger.info("Computing SHAP values for model explainability...")
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
    except Exception as tree_err:
        logger.warning(f"TreeExplainer failed ({tree_err}), falling back to KernelExplainer...")
        predict_fn = (
            estimator.predict_proba
            if task_type == "classification" and hasattr(estimator, "predict_proba")
            else estimator.predict
        )
        background = shap.sample(X_test_trans_df, min(50, len(X_test_trans_df)))
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    # Normalize SHAP values matrix for plotting & feature importance (must be 2D)
    if isinstance(shap_values, list):
        plot_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif getattr(shap_values, "ndim", 0) == 3:
        plot_vals = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
    else:
        plot_vals = shap_values

    # Calculate global feature importance
    mean_abs_shap = np.mean(np.abs(plot_vals), axis=0)
    feat_imp = [
        {"feature": str(col), "importance": round(float(imp), 4)}
        for col, imp in zip(feature_names, mean_abs_shap)
    ]
    feat_imp = sorted(feat_imp, key=lambda x: x["importance"], reverse=True)

    # 2. Generate SHAP Summary Plot PNG
    png_filename = f"{run_id}_shap_summary.png"
    png_path = os.path.join(reports_dir, png_filename)
    logger.info(f"Generating SHAP summary plot at: {png_path}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(plot_vals, X_sample, show=False)
    plt.title(f"AutoForge SHAP Summary: {model_name} ({task_type})", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Generate Shapash Interactive HTML Report
    html_filename = f"{run_id}_shapash_report.html"
    html_path = os.path.join(reports_dir, html_filename)
    logger.info(f"Generating Shapash HTML report at: {html_path}")

    # Prepare metadata yaml for Shapash
    info_dict = {
        "dataset": {"name": target_column, "samples": len(X_sample)},
        "model": {"name": estimator.__class__.__name__, "type": task_type},
        "project": {"author": "AutoForge", "date": "auto"},
    }
    info_path = os.path.join(reports_dir, f"{run_id}_project_info.yml")
    with open(info_path, "w") as f:
        yaml.dump(info_dict, f)

    metric_cfg = (
        [{"path": "sklearn.metrics.accuracy_score", "name": "Accuracy"}]
        if task_type == "classification"
        else [{"path": "sklearn.metrics.mean_squared_error", "name": "MSE"}]
    )

    try:
        contrib_df = pd.DataFrame(plot_vals, columns=feature_names, index=X_sample.index)
        xpl = SmartExplainer(model=estimator)
        y_pred = pd.Series(estimator.predict(X_sample), index=X_sample.index)
        xpl.compile(x=X_sample, contributions=contrib_df, y_pred=y_pred)
        xpl.generate_report(
            output_file=html_path,
            project_info_file=info_path,
            x_train=X_sample,
            y_test=pd.Series(y_test.iloc[:sample_size].values, index=X_sample.index),
            title_story=f"AutoForge Explainability: {model_name}",
            metrics=metric_cfg,
        )
        logger.info(f"Shapash report generated successfully ({os.path.getsize(html_path)} bytes)")
    except Exception as shapash_err:
        logger.warning(f"Shapash report generation fallback triggered: {shapash_err}")
        # Create fallback standalone explanation HTML if shapash notebook execution encounters env quirks
        with open(html_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head><title>AutoForge Explainability Report</title>
<style>body {{ font-family: sans-serif; margin: 40px; background: #0e1117; color: #e0e0e0; }}
h1 {{ color: #00d2ff; }} table {{ border-collapse: collapse; width: 600px; }}
th, td {{ border: 1px solid #333; padding: 10px; text-align: left; }}
th {{ background: #1f2937; color: #fff; }} tr:nth-child(even) {{ background: #161b22; }}</style>
</head>
<body>
<h1>AutoForge Explainability Report</h1>
<p><strong>Model:</strong> {estimator.__class__.__name__} ({task_type})</p>
<p><strong>Run ID:</strong> {run_id}</p>
<h2>Top Features by Global SHAP Importance</h2>
<table>
<tr><th>Rank</th><th>Feature</th><th>Mean Absolute SHAP Value</th></tr>
{''.join(f'<tr><td>{i+1}</td><td>{f["feature"]}</td><td>{f["importance"]}</td></tr>' for i, f in enumerate(feat_imp[:15]))}
</table>
<br>
<h2>SHAP Summary Plot</h2>
<img src="/artifacts/reports/{png_filename}" style="max-width: 800px; border-radius: 8px;" />
</body>
</html>""")

    # 4. Log Explainability Artifacts to MLflow
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(png_path, artifact_path="explainability")
            mlflow.log_artifact(html_path, artifact_path="explainability")
            logger.info(f"Logged explainability artifacts to MLflow run {run_id}")
    except Exception as mlf_err:
        logger.warning(f"Could not log artifacts to MLflow run {run_id}: {mlf_err}")

    # Build result record
    explanation_record = {
        "run_id": run_id,
        "task_type": task_type,
        "model_name": model_name,
        "model_class": estimator.__class__.__name__,
        "shap_summary_png": png_path,
        "shap_summary_url": f"/artifacts/reports/{png_filename}",
        "shapash_html_report": html_path,
        "shapash_html_url": f"/artifacts/reports/{html_filename}",
        "top_features": feat_imp[:15],
    }

    # Save record to disk for quick API lookup
    json_path = os.path.join(reports_dir, f"{run_id}_explanation.json")
    with open(json_path, "w") as f:
        json.dump(explanation_record, f, indent=2)

    return explanation_record
