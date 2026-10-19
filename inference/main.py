"""
AutoForge Inference Service
FastAPI prediction microservice running on port 8001.
Loads trained scikit-learn Pipelines from MLflow or local artifact store,
serves low-latency predictions with probabilities, logs per-request latencies,
and archives incoming payloads to SQLite for drift monitoring.
"""

import os
import time
import pickle
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.logging_config import logger
from inference.request_logger import RequestLogger

app = FastAPI(
    title="AutoForge Inference Service",
    description="Low-latency model serving API with request logging for drift detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
request_logger = RequestLogger()


class ModelHolder:
    def __init__(self):
        self.pipeline: Any = None
        self.model_name: str = "unloaded"
        self.model_version: str = "0"
        self.source: str = "none"

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load_from_file(self, path: str, name: str = "local_model", version: str = "1"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        logger.info(f"Loading model pipeline from disk: {path}")
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.model_name = name
        self.model_version = str(version)
        self.source = f"file:{path}"
        logger.info(f"Successfully loaded model '{name}' (version {version}) from {path}")

    def load_from_mlflow(self, model_name: str, version: Optional[str] = None):
        import mlflow.sklearn
        uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.abspath(ARTIFACTS_DIR)}/mlflow.db")
        mlflow.set_tracking_uri(uri)

        model_uri = f"models:/{model_name}/{version}" if version else f"models:/{model_name}/latest"
        logger.info(f"Loading model from MLflow registry: {model_uri}")
        self.pipeline = mlflow.sklearn.load_model(model_uri)
        self.model_name = model_name
        self.model_version = str(version or "latest")
        self.source = model_uri
        logger.info(f"Successfully loaded model from MLflow: {model_uri}")


model_holder = ModelHolder()


@app.on_event("startup")
def startup_load_model():
    """Attempts to auto-load latest model if present."""
    latest_path = os.path.join(MODELS_DIR, "latest_pipeline.pkl")
    if os.path.exists(latest_path):
        try:
            model_holder.load_from_file(latest_path, name="autoforge_model", version="latest")
        except Exception as e:
            logger.warning(f"Could not auto-load latest pipeline from disk: {e}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="List of feature dict rows to predict")


class LoadModelRequest(BaseModel):
    model_name: Optional[str] = Field("autoforge_model", description="Model name in MLflow registry")
    version: Optional[str] = Field(None, description="Model version in MLflow registry")
    pipeline_path: Optional[str] = Field(None, description="Local path to .pkl pipeline file")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint showing current model and logging status."""
    return {
        "status": "healthy",
        "service": "autoforge-inference",
        "model_loaded": model_holder.is_loaded(),
        "model_name": model_holder.model_name,
        "model_version": model_holder.model_version,
        "model_source": model_holder.source,
        "total_requests_logged": request_logger.count_requests(),
    }


@app.post("/model/load")
def load_model(req: LoadModelRequest):
    """Dynamically loads or hot-swaps the active model from MLflow or disk."""
    try:
        if req.pipeline_path and os.path.exists(req.pipeline_path):
            model_holder.load_from_file(req.pipeline_path, req.model_name or "local_model", req.version or "1")
        elif req.model_name:
            # Try loading from MLflow
            try:
                model_holder.load_from_mlflow(req.model_name, req.version)
            except Exception as mlf_err:
                logger.warning(f"MLflow load failed ({mlf_err}), checking local disk models...")
                # Fallback to local pickled pipeline
                latest_path = os.path.join(MODELS_DIR, "latest_pipeline.pkl")
                if os.path.exists(latest_path):
                    model_holder.load_from_file(latest_path, req.model_name, req.version or "1")
                else:
                    raise mlf_err
        else:
            raise HTTPException(status_code=400, detail="Must specify model_name or pipeline_path.")

        return {
            "status": "success",
            "message": f"Loaded model '{model_holder.model_name}' version '{model_holder.model_version}'",
            "source": model_holder.source,
        }
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Main prediction endpoint. Accepts feature rows, executes pipeline,
    computes latency, logs request to SQLite, and returns predictions.
    """
    if not model_holder.is_loaded():
        # Attempt just-in-time auto-load from latest pipeline
        latest_path = os.path.join(MODELS_DIR, "latest_pipeline.pkl")
        if os.path.exists(latest_path):
            model_holder.load_from_file(latest_path, name="autoforge_model", version="latest")
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model is currently loaded in the inference service. Please train or deploy a model first.",
            )

    if not req.rows:
        raise HTTPException(status_code=400, detail="Payload 'rows' list is empty.")

    t_start = time.perf_counter()

    try:
        df_input = pd.DataFrame(req.rows)
        pipeline = model_holder.pipeline

        # Predict
        raw_preds = pipeline.predict(df_input)
        predictions = raw_preds.tolist() if hasattr(raw_preds, "tolist") else list(raw_preds)

        # Probabilities if available
        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            try:
                raw_probs = pipeline.predict_proba(df_input)
                probabilities = raw_probs.tolist() if hasattr(raw_probs, "tolist") else [list(p) for p in raw_probs]
            except Exception:
                probabilities = None

        latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

        # Log request to SQLite for drift monitoring
        try:
            request_logger.log_requests(
                rows=req.rows,
                predictions=predictions,
                probabilities=probabilities,
                model_name=model_holder.model_name,
                model_version=model_holder.model_version,
                latency_ms=latency_ms,
            )
        except Exception as log_err:
            logger.warning(f"Could not persist inference log to SQLite: {log_err}")

        logger.info(f"POST /predict: {len(req.rows)} rows -> latency={latency_ms}ms")

        return {
            "status": "success",
            "model_name": model_holder.model_name,
            "model_version": model_holder.model_version,
            "count": len(predictions),
            "predictions": predictions,
            "probabilities": probabilities,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
