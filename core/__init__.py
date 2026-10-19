"""
AutoForge Core Machine Learning & MLOps Engine
"""

from core.logging_config import setup_logging, logger
from core.ingestion import load_dataset, inspect_dataset
from core.task_detection import detect_task_type
from core.profiling import generate_profiling_report

__all__ = [
    "setup_logging",
    "logger",
    "load_dataset",
    "inspect_dataset",
    "detect_task_type",
    "generate_profiling_report",
]
