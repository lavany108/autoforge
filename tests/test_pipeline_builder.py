"""
Unit Tests for Pipeline Builder & ColumnTransformer
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from core.pipeline_builder import (
    select_feature_columns,
    build_preprocessor,
    build_full_pipeline,
    get_feature_names_after_preprocessing,
)


def test_select_feature_columns():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0],
        "cat1": ["a", "b", "a", "b"],
        "id_col": ["ID_1", "ID_2", "ID_3", "ID_4"],  # High cardinality text identifier
        "mostly_nan": [1.0, np.nan, np.nan, np.nan],  # > 75% nan
        "target": [0, 1, 0, 1],
    })
    num_cols, cat_cols, dropped = select_feature_columns(df, target_column="target")
    assert "num1" in num_cols
    assert "cat1" in cat_cols
    assert "target" not in num_cols
    assert "target" not in cat_cols


def test_build_preprocessor_and_pipeline():
    df = pd.DataFrame({
        "age": [25.0, np.nan, 40.0, 35.0, 50.0],
        "embarked": ["S", "C", np.nan, "S", "Q"],
        "target": [1, 0, 1, 0, 1],
    })
    num_cols = ["age"]
    cat_cols = ["embarked"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    pipeline = build_full_pipeline(preprocessor, clf)

    X = df[["age", "embarked"]]
    y = df["target"]

    # Test fitting full pipeline
    pipeline.fit(X, y)

    # Test prediction
    preds = pipeline.predict(X)
    assert len(preds) == len(df)
    assert all(p in [0, 1] for p in preds)

    # Test feature names out
    names = get_feature_names_after_preprocessing(preprocessor)
    assert len(names) >= 2
    assert any("age" in n for n in names)
