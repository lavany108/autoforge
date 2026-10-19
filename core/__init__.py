"""
AutoForge Core Machine Learning & MLOps Engine
"""

from core.logging_config import setup_logging, logger
from core.ingestion import load_dataset, inspect_dataset
from core.task_detection import detect_task_type
from core.profiling import generate_profiling_report
from core.pipeline_builder import (
    select_feature_columns,
    build_preprocessor,
    build_full_pipeline,
    get_feature_names_after_preprocessing,
)
from core.model_search import train_autoforge_model

__all__ = [
    "setup_logging",
    "logger",
    "load_dataset",
    "inspect_dataset",
    "detect_task_type",
    "generate_profiling_report",
    "select_feature_columns",
    "build_preprocessor",
    "build_full_pipeline",
    "get_feature_names_after_preprocessing",
    "train_autoforge_model",
]
