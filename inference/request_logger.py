"""
AutoForge Inference Request Logger
Stores incoming prediction requests and model outputs into SQLite for
latency tracking and downstream Evidently drift analysis.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from core.logging_config import logger


class RequestLogger:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "artifacts/autoforge.db")
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self):
        """Initializes the inference_logs table schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    model_name TEXT,
                    model_version TEXT,
                    features_json TEXT NOT NULL,
                    prediction TEXT,
                    probability REAL,
                    latency_ms REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_inference_logs_ts 
                ON inference_logs(timestamp)
            """)
            conn.commit()
        logger.info(f"Initialized inference logs SQLite store at: {self.db_path}")

    def log_requests(
        self,
        rows: List[Dict[str, Any]],
        predictions: List[Any],
        probabilities: Optional[List[Any]],
        model_name: str,
        model_version: str,
        latency_ms: float,
    ) -> str:
        """Logs a batch of inference requests into the SQLite store."""
        req_id = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat()

        records = []
        for i, row in enumerate(rows):
            pred_str = str(predictions[i]) if i < len(predictions) else None
            prob_val = None
            if probabilities is not None and i < len(probabilities):
                p = probabilities[i]
                if isinstance(p, (list, tuple)):
                    prob_val = float(max(p))
                elif isinstance(p, (float, int)):
                    prob_val = float(p)

            records.append((
                now_iso,
                req_id,
                model_name,
                str(model_version),
                json.dumps(row, default=str),
                pred_str,
                prob_val,
                latency_ms,
            ))

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO inference_logs 
                (timestamp, request_id, model_name, model_version, features_json, prediction, probability, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()

        return req_id

    def get_recent_requests(self, limit: int = 1000) -> pd.DataFrame:
        """Retrieves the most recent inference requests unpacked into a DataFrame."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT features_json, prediction, probability, timestamp FROM inference_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            data = cursor.fetchall()

        if not data:
            return pd.DataFrame()

        rows = []
        for feat_str, pred, prob, ts in data:
            try:
                row_dict = json.loads(feat_str)
                row_dict["_prediction"] = pred
                row_dict["_probability"] = prob
                row_dict["_timestamp"] = ts
                rows.append(row_dict)
            except Exception:
                continue

        return pd.DataFrame(rows)

    def count_requests(self) -> int:
        """Returns total number of inference requests logged."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM inference_logs")
            return cursor.fetchone()[0]
