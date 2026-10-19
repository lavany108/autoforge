"""
AutoForge Pipeline Builder
Constructs unified scikit-learn Pipelines combining ColumnTransformer
preprocessing (numeric imputation + scaling, categorical imputation + one-hot encoding)
with downstream ML estimators.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from core.logging_config import logger


def select_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    ignore_columns: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorizes columns into numeric, categorical, and dropped/ignored features.
    Automatically identifies high-cardinality ID columns or extreme missingness.
    """
    ignore_set = set(ignore_columns or [])
    ignore_set.add(target_column)

    num_cols = []
    cat_cols = []
    dropped_cols = []

    n_rows = len(df)

    for col in df.columns:
        if col in ignore_set:
            continue

        series = df[col]
        missing_rate = series.isnull().mean()
        n_unique = series.nunique(dropna=True)

        # Drop columns with > 90% missing values
        if missing_rate > 0.90:
            logger.info(f"Dropping column '{col}': missing rate {missing_rate:.1%} exceeds 90%")
            dropped_cols.append(col)
            continue

        # Drop unique text identifiers (e.g., Name, Ticket, PassengerId)
        is_obj = pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)
        if is_obj and n_unique > 0.8 * n_rows and n_unique > 50:
            logger.info(f"Dropping unique identifier column '{col}' ({n_unique} distinct text values)")
            dropped_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series.dtype):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    logger.info(
        f"Features selected: {len(num_cols)} numeric, {len(cat_cols)} categorical, "
        f"{len(dropped_cols)} dropped"
    )
    return num_cols, cat_cols, dropped_cols


def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
) -> ColumnTransformer:
    """
    Builds a ColumnTransformer for numeric and categorical feature preprocessing.
    """
    transformers = []

    if num_cols:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_full_pipeline(
    preprocessor: ColumnTransformer,
    estimator: Any,
) -> Pipeline:
    """
    Combines ColumnTransformer preprocessor and an estimator into a single sklearn Pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    return pipeline


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts the names of features after ColumnTransformer fit.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out is not available
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" or trans == "drop":
                continue
            if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                onehot = trans.named_steps["onehot"]
                try:
                    cat_names = onehot.get_feature_names_out(cols)
                    names.extend(cat_names)
                except Exception:
                    names.extend(cols)
            else:
                names.extend(cols)
        return names
