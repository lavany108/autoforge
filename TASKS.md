# AutoForge Implementation Progress & Task Tracker

## Architecture Overview
- **UI (Streamlit, port 8501)**: Upload → Profile → Train → Leaderboard → Explain → Deploy → Monitor
- **API (FastAPI, port 8000)**: Orchestrates training runs, artifact management, MLflow integration
- **Inference (FastAPI, port 8001)**: Loads registered sklearn Pipeline, exposes `/predict`, `/health`, logs inference requests
- **MLflow (port 5000)**: Local tracking server + model registry, SQLite backend, disk volume artifacts
- **Drift Worker**: Background job comparing recent requests vs training reference with Evidently; triggers auto-retraining

---

## Phases Status

- [x] **Phase 0: Scaffold & Dependencies**
  - [x] Environment verified (Python 3.11, Docker Desktop running on Apple Silicon ARM64)
  - [x] Pinned `requirements.txt` with mutually compatible versions
  - [x] `.env.example`, `.gitignore`, `TASKS.md`
  - [x] Structured logging config (`core/logging_config.py`)
  - [x] `docker-compose.yml` skeleton
  - [x] Verify core dependencies import cleanly
  - [x] Git commit & push

- [ ] **Phase 1: Data & Profiling**
  - [ ] Dataset ingestion & validation API (`/datasets/upload`, `/datasets/profile`)
  - [ ] `ydata-profiling` HTML report generation and artifact saving
  - [ ] Task detection heuristic (classification vs regression with confidence score)
  - [ ] Titanic seed dataset committed (`data/titanic.csv`), `scripts/download_data.py`
  - [ ] Verify profiling on Titanic produces HTML report with task=classification
  - [ ] Git commit & push

- [ ] **Phase 2: Training Engine**
  - [ ] Scikit-learn Pipeline builder with `ColumnTransformer` (numeric impute+scale, categorical impute+one-hot)
  - [ ] Baseline model training with default hyperparameters & MLflow logging
  - [ ] Automated model search via FLAML with configurable time budget
  - [ ] Optuna hyperparameter refinement on the FLAML winner
  - [ ] Metrics logging (accuracy, F1, precision, recall for classification; RMSE, MAE, R2 for regression) + timings
  - [ ] Single full Pipeline serialization (preprocessing + estimator) and MLflow model registration
  - [ ] Verify baseline + tuned runs logged in MLflow with tuned >= baseline accuracy
  - [ ] Git commit & push

- [ ] **Phase 3: Explainability**
  - [ ] Model explainability at training time (SHAP TreeExplainer / KernelExplainer fallback)
  - [ ] Shapash HTML interactive report + SHAP summary plot PNG
  - [ ] Save reports as MLflow artifacts
  - [ ] API endpoint `GET /runs/{run_id}/explanation` returning artifact paths/links
  - [ ] Verify report generation and visual features
  - [ ] Git commit & push

- [ ] **Phase 4: Inference Service**
  - [ ] FastAPI prediction service (`POST /predict`, `GET /health`)
  - [ ] Dynamic loading of registered models from MLflow
  - [ ] Pydantic input/output schemas with probability support
  - [ ] Per-request latency logging and request storage for drift monitoring
  - [ ] Verify deployment and curl `/predict` with test payloads
  - [ ] Git commit & push

- [ ] **Phase 5: Streamlit UI**
  - [ ] Multi-page / tabbed modern dashboard:
    1. Upload & Profiling (file uploader, task heuristic, embedded profile report)
    2. Training (dataset selector, budget slider, live training progress)
    3. Leaderboard (baseline vs tuned comparison table from MLflow)
    4. Explainability (embedded Shapash HTML + SHAP summary plot)
    5. Deployment (model registry version selector, one-click deploy to inference service)
    6. Monitoring (Evidently drift metrics & retrain history)
  - [ ] UI visual testing & screenshots saved to `docs/screenshots/`
  - [ ] Git commit & push

- [ ] **Phase 6: Drift & Auto-Retraining**
  - [ ] SQLite inference request logging (`inference/request_logger.py`)
  - [ ] Drift worker job (`jobs/drift_worker.py`) comparing recent requests against reference snapshot
  - [ ] Trigger rules: drift share > 30% of features OR any feature PSI > 0.2
  - [ ] Automatic API trigger to retrain and register new model version
  - [ ] `scripts/simulate_drift.py` for synthetic feature distribution shifts
  - [ ] Verify simulated drift triggers auto-retraining and model registration
  - [ ] Git commit & push

- [ ] **Phase 7: Packaging, Tests & Benchmarks**
  - [ ] Finalize `docker-compose.yml` for zero-friction `docker compose up -d`
  - [ ] Unit & integration test suite (`tests/` with pytest)
  - [ ] Benchmark runner (`benchmarks/benchmark_run.py`) on Titanic + California Housing
  - [ ] Locust load test script (`benchmarks/locustfile.py`) measuring p95 latency
  - [ ] Documentation: `README.md` and `WALKTHROUGH.md`
  - [ ] Final end-to-end verification and presentation
  - [ ] Git commit & push
