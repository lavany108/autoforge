"""
AutoForge Task Detection Engine
Determines whether a machine learning task is classification or regression
using statistical heuristics, calculates confidence, and logs decisions.
"""

from typing import Dict, Any, List
import pandas as pd
from core.logging_config import logger


def detect_task_type(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Task detection heuristic:
    - Target is object/bool OR (<= 20 unique values AND not float) -> classification
    - Else -> regression

    Returns:
        Dict containing task_type ('classification' or 'regression'),
        confidence score (0.0 to 1.0), reasoning, and target metadata.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in dataset.")

    series = df[target_column].dropna()
    if len(series) == 0:
        raise ValueError(f"Target column '{target_column}' contains only missing values.")

    dtype = series.dtype
    unique_vals = series.unique()
    n_unique = len(unique_vals)

    is_object = pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
    is_bool = pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype)
    is_float = pd.api.types.is_float_dtype(dtype)
    is_numeric = pd.api.types.is_numeric_dtype(dtype)

    is_object_or_bool = is_object or is_bool

    # Heuristic evaluation
    if is_object_or_bool or (n_unique <= 20 and not is_float):
        task_type = "classification"
        # Determine confidence
        if is_object_or_bool:
            confidence = 0.99
            reasoning = (
                f"Target '{target_column}' is non-numeric/boolean (dtype: {dtype}) "
                f"with {n_unique} distinct classes."
            )
        elif n_unique <= 2:
            confidence = 0.99
            reasoning = (
                f"Target '{target_column}' has only {n_unique} unique numeric values "
                f"(binary classification)."
            )
        elif n_unique <= 10:
            confidence = 0.95
            reasoning = (
                f"Target '{target_column}' has {n_unique} discrete integer values "
                f"(<= 20 unique values and not float)."
            )
        else:
            confidence = 0.88
            reasoning = (
                f"Target '{target_column}' has {n_unique} distinct discrete values "
                f"(<= 20 unique values and not float)."
            )

        classes = [str(x) for x in sorted(unique_vals.tolist()[:20])]
        is_binary = (n_unique == 2)
    else:
        task_type = "regression"
        classes = []
        is_binary = False
        if is_float and n_unique > 50:
            confidence = 0.98
            reasoning = (
                f"Target '{target_column}' is continuous floating-point (dtype: {dtype}) "
                f"with {n_unique} unique values."
            )
        elif is_float:
            confidence = 0.92
            reasoning = (
                f"Target '{target_column}' is floating-point (dtype: {dtype}) "
                f"with {n_unique} unique values (treated as continuous)."
            )
        else:
            confidence = 0.90
            reasoning = (
                f"Target '{target_column}' is integer with {n_unique} distinct values "
                f"(> 20 unique values, high cardinality continuous target)."
            )

    decision = {
        "target_column": target_column,
        "task_type": task_type,
        "confidence": round(confidence, 4),
        "reasoning": reasoning,
        "n_unique": int(n_unique),
        "target_dtype": str(dtype),
        "is_binary": is_binary,
        "classes": classes,
    }

    logger.info(
        f"[Task Detection] Target='{target_column}' -> Task='{task_type}' "
        f"(confidence={decision['confidence']:.2f}, n_unique={n_unique}, dtype={dtype}). "
        f"Reason: {reasoning}"
    )

    return decision
