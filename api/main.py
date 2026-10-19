"""
AutoForge API Service
Main FastAPI application orchestrating dataset operations, profiling,
training runs, MLflow integration, and model management.
"""

import os
import shutil
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.logging_config import logger
from core.ingestion import load_dataset, inspect_dataset
from core.task_detection import detect_task_type
from core.profiling import generate_profiling_report

app = FastAPI(
    title="AutoForge API",
    description="Automated ML & MLOps Platform API",
    version="1.0.0",
)

# Enable CORS for Streamlit and external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required artifact directories exist
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
DATA_DIR = os.getenv("DATA_DIR", "data")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
UPLOADS_DIR = os.path.join(ARTIFACTS_DIR, "uploads")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")

for d in [ARTIFACTS_DIR, DATA_DIR, REPORTS_DIR, UPLOADS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Mount static files to serve generated HTML reports and plots directly
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class ProfileRequest(BaseModel):
    file_path: str = Field(..., description="Path to the dataset file (CSV/Parquet)")
    dataset_name: Optional[str] = Field(None, description="Optional custom name for the dataset")
    target_column: Optional[str] = Field(None, description="Optional target column for task detection")
    minimal: Optional[bool] = Field(False, description="Run minimal profiling for large datasets")


class TaskDetectionRequest(BaseModel):
    file_path: str = Field(..., description="Path to the dataset file")
    target_column: str = Field(..., description="Target column to analyze")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint for container and service monitoring."""
    return {
        "status": "healthy",
        "service": "autoforge-api",
        "artifacts_dir": ARTIFACTS_DIR,
    }


@app.get("/datasets")
def list_datasets():
    """Lists available CSV datasets in data/ and uploaded directories."""
    datasets = []

    # Check data/
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith((".csv", ".parquet")):
                path = os.path.join(DATA_DIR, f)
                datasets.append({
                    "name": f,
                    "path": path,
                    "size_kb": round(os.path.getsize(path) / 1024, 2),
                    "source": "seed_data",
                })

    # Check uploads/
    if os.path.exists(UPLOADS_DIR):
        for f in os.listdir(UPLOADS_DIR):
            if f.endswith((".csv", ".parquet")):
                path = os.path.join(UPLOADS_DIR, f)
                datasets.append({
                    "name": f,
                    "path": path,
                    "size_kb": round(os.path.getsize(path) / 1024, 2),
                    "source": "uploaded",
                })

    return {"datasets": datasets}


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a CSV dataset and returns its metadata and column schema."""
    if not file.filename.endswith((".csv", ".parquet")):
        raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported.")

    file_path = os.path.join(UPLOADS_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = load_dataset(file_path)
        summary = inspect_dataset(df)
        summary["file_path"] = file_path
        summary["filename"] = file.filename
        logger.info(f"Dataset '{file.filename}' uploaded successfully to {file_path}")
        return summary
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")


@app.post("/task/detect")
def detect_task(req: TaskDetectionRequest):
    """Detects whether a column represents classification or regression."""
    try:
        df = load_dataset(req.file_path)
        decision = detect_task_type(df, req.target_column)
        return decision
    except Exception as e:
        logger.error(f"Task detection failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/datasets/profile")
def profile_dataset(req: ProfileRequest):
    """
    Generates a ydata-profiling HTML report and performs task detection
    if target_column is specified.
    """
    try:
        df = load_dataset(req.file_path)
        dataset_name = req.dataset_name or os.path.splitext(os.path.basename(req.file_path))[0]

        # Generate ydata-profiling report
        report_meta = generate_profiling_report(
            df=df,
            dataset_name=dataset_name,
            output_dir=REPORTS_DIR,
            minimal=req.minimal or False,
        )

        # Build relative URL for static serving
        report_filename = os.path.basename(report_meta["report_path"])
        report_meta["report_url"] = f"/artifacts/reports/{report_filename}"

        # If target column is specified, perform task detection
        task_info = None
        if req.target_column:
            task_info = detect_task_type(df, req.target_column)

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "profiling": report_meta,
            "task_detection": task_info,
            "columns": list(df.columns),
            "num_rows": len(df),
            "num_columns": len(df.columns),
        }
    except Exception as e:
        logger.error(f"Dataset profiling failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")
