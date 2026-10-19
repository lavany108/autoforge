"""
AutoForge Web Dashboard
Streamlit-based platform interface for AutoML & MLOps lifecycle:
1. Upload & Profiling
2. Automated Training (FLAML + Optuna)
3. Experiment Leaderboard (MLflow)
4. Explainability (SHAP + Shapash)
5. Model Deployment & Live Testing
6. Drift Monitoring & Auto-Retrain Alerts
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Service URLs
API_URL = os.getenv("API_URL", "http://localhost:8000")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
DATA_DIR = os.getenv("DATA_DIR", "data")

st.set_page_config(
    page_title="AutoForge | AutoML & MLOps Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Dark glassmorphic theme styling */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #111827 0%, #030712 100%);
        color: #f3f4f6;
    }
    .main-header {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #7928ca 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    .metric-card {
        background: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .delta-pos { color: #10b981; }
    .delta-neg { color: #ef4444; }
    .badge-class {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-reg {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid #3b82f6;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-drift {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(31, 41, 55, 0.4);
        border-radius: 8px 8px 0px 0px;
        padding: 8px 18px;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #60a5fa !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar & Navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("<h2 style='color:#00d2ff; margin-bottom:0;'>⚡ AutoForge</h2>", unsafe_allow_html=True)
st.sidebar.caption("Automated ML & Local MLOps Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. 📂 Upload & Profiling",
        "2. 🚀 Automated Training",
        "3. 🏆 Experiment Leaderboard",
        "4. 🔍 Model Explainability",
        "5. 🚢 Model Deployment",
        "6. 📊 Monitoring & Drift",
    ],
    index=0,
)

# Service status indicators
st.sidebar.markdown("---")
st.sidebar.markdown("**System Services:**")
try:
    r_api = requests.get(f"{API_URL}/health", timeout=1)
    api_ok = (r_api.status_code == 200)
except Exception:
    api_ok = False

try:
    r_inf = requests.get(f"{INFERENCE_URL}/health", timeout=1)
    inf_ok = (r_inf.status_code == 200)
    inf_data = r_inf.json() if inf_ok else {}
except Exception:
    inf_ok = False
    inf_data = {}

st.sidebar.markdown(f"API Orchestrator: `{'🟢 Online' if api_ok else '🔴 Offline'}`")
st.sidebar.markdown(f"Inference Service: `{'🟢 Online' if inf_ok else '🔴 Offline'}`")
if inf_ok and inf_data.get("model_loaded"):
    st.sidebar.caption(f"Active Model: {inf_data.get('model_name')} (v{inf_data.get('model_version')})")


# ===========================================================================
# PAGE 1: Upload & Profiling
# ===========================================================================
if page.startswith("1."):
    st.markdown("<div class='main-header'>Dataset Ingestion & Automated Profiling</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload datasets, inspect data types, generate ydata-profiling reports, and detect task type.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📥 Select or Upload Dataset")
        upload_choice = st.radio("Dataset Source:", ["Seed Dataset", "Upload Custom CSV"], horizontal=True)

        selected_file_path = None
        if upload_choice == "Seed Dataset":
            seed_files = []
            if os.path.exists("data"):
                seed_files = [f"data/{f}" for f in os.listdir("data") if f.endswith(".csv")]
            if not seed_files:
                seed_files = ["data/titanic.csv"]
            selected_file_path = st.selectbox("Choose Seed Dataset:", seed_files, index=0)
        else:
            uploaded = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded is not None:
                save_dir = os.path.join(ARTIFACTS_DIR, "uploads")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, uploaded.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.success(f"Uploaded: {uploaded.name}")
                selected_file_path = save_path

    with col2:
        if selected_file_path and os.path.exists(selected_file_path):
            df_preview = pd.read_csv(selected_file_path)
            st.markdown("### 🎯 Task Detection & Target")
            target_col = st.selectbox(
                "Select Target Column:",
                options=list(df_preview.columns),
                index=list(df_preview.columns).index("Survived") if "Survived" in df_preview.columns else len(df_preview.columns) - 1,
            )

            # Query API task detection
            task_resp = None
            if api_ok:
                try:
                    t_resp = requests.post(f"{API_URL}/task/detect", json={"file_path": selected_file_path, "target_column": target_col})
                    if t_resp.status_code == 200:
                        task_resp = t_resp.json()
                except Exception:
                    pass

            # Fallback local heuristic if API offline
            if not task_resp:
                from core.task_detection import detect_task_type
                task_resp = detect_task_type(df_preview, target_col)

            t_type = task_resp.get("task_type", "unknown")
            badge_html = f"<span class='badge-class'>Classification</span>" if t_type == "classification" else f"<span class='badge-reg'>Regression</span>"

            st.markdown(f"**Detected Task Type:** {badge_html} &nbsp; (Confidence: **{task_resp.get('confidence', 1.0):.1%}**)", unsafe_allow_html=True)
            st.info(f"💡 **Reasoning:** {task_resp.get('reasoning')}")

    if selected_file_path and os.path.exists(selected_file_path):
        df_preview = pd.read_csv(selected_file_path)
        st.markdown("---")
        # Summary KPI metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{len(df_preview):,}")
        m2.metric("Total Columns", f"{len(df_preview.columns):,}")
        m3.metric("Missing Values", f"{df_preview.isnull().sum().sum():,} ({df_preview.isnull().sum().sum() / (len(df_preview)*len(df_preview.columns)):.1%})")
        m4.metric("Duplicate Rows", f"{df_preview.duplicated().sum():,}")

        st.markdown("#### Preview Data Table")
        st.dataframe(df_preview.head(5), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Dataset Profiling Report")
        p_col1, p_col2 = st.columns([1, 3])
        with p_col1:
            fast_mode = st.checkbox("Fast Minimal Profiling", value=True)
            gen_btn = st.button("⚡ Generate Profile Report", type="primary")

        dataset_base = os.path.splitext(os.path.basename(selected_file_path))[0]
        report_file = os.path.join(ARTIFACTS_DIR, "reports", f"{dataset_base}_profile.html")

        if gen_btn:
            with st.spinner("Analyzing dataset with ydata-profiling..."):
                if api_ok:
                    res = requests.post(
                        f"{API_URL}/datasets/profile",
                        json={
                            "file_path": selected_file_path,
                            "dataset_name": dataset_base,
                            "target_column": target_col,
                            "minimal": fast_mode,
                        },
                    )
                    if res.status_code == 200:
                        st.success(f"Report generated successfully! (Took {res.json()['profiling']['generation_duration_sec']}s)")
                    else:
                        st.error(f"Profiling error: {res.text}")
                else:
                    from core.profiling import generate_profiling_report
                    generate_profiling_report(df_preview, dataset_base, minimal=fast_mode)
                    st.success("Report generated locally!")

        if os.path.exists(report_file):
            st.markdown(f"**Viewing Saved Report:** `{report_file}` ({os.path.getsize(report_file)/1024:.1f} KB)")
            with open(report_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=750, scrolling=True)


# ===========================================================================
# PAGE 2: Automated Training
# ===========================================================================
elif page.startswith("2."):
    st.markdown("<div class='main-header'>AutoML Training Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>FLAML automated model search combined with Optuna hyperparameter refinement.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### ⚙️ Training Configuration")
        dataset_path = st.selectbox(
            "Select Training Dataset:",
            ["data/titanic.csv"] + ([f"data/{f}" for f in os.listdir("data") if f != "titanic.csv and f.endswith('.csv')"] if os.path.exists("data") else []),
        )

        df_train_view = pd.read_csv(dataset_path)
        target_name = st.selectbox(
            "Target Column:",
            options=list(df_train_view.columns),
            index=list(df_train_view.columns).index("Survived") if "Survived" in df_train_view.columns else len(df_train_view.columns) - 1,
        )

        time_budget = st.slider("FLAML Search Time Budget (seconds):", min_value=10, max_value=300, value=20, step=5)
        optuna_trials = st.slider("Optuna Refinement Trials:", min_value=5, max_value=100, value=15, step=5)
        model_reg_name = st.text_input("MLflow Registered Model Name:", value="autoforge_titanic")

        launch_btn = st.button("🚀 Launch AutoForge Training Run", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📋 Engine Pipeline Details")
        st.markdown("""
        **Execution Stages:**
        1. **Feature Ingestion & ColumnTransformer:** Numeric median impute + standard scaling, categorical most-frequent impute + one-hot encoding.
        2. **Baseline Model:** Default Random Forest/LGBM benchmark with metrics logged to MLflow.
        3. **FLAML Discovery:** Evaluates LightGBM, Random Forest, XGBoost, and ExtraTrees within the time budget.
        4. **Optuna Tuning:** Hyperparameter refinement on the winning architecture.
        5. **Pipeline Assembly & Registry:** Single pipeline pickled as artifact and registered in MLflow model registry.
        """)

    if launch_btn:
        progress_bar = st.progress(0, text="Initializing AutoML environment...")
        status_box = st.empty()

        with st.spinner("Training baseline model & searching candidates via FLAML + Optuna..."):
            progress_bar.progress(25, text="Training baseline benchmark...")
            train_payload = {
                "file_path": dataset_path,
                "target_column": target_name,
                "time_budget_secs": time_budget,
                "optuna_trials": optuna_trials,
                "model_name": model_reg_name,
            }

            train_res = None
            if api_ok:
                try:
                    r = requests.post(f"{API_URL}/training/train", json=train_payload, timeout=600)
                    if r.status_code == 200:
                        train_res = r.json()
                    else:
                        st.error(f"Training failed: {r.text}")
                except Exception as e:
                    st.error(f"API Error: {e}")
            else:
                from core.model_search import train_autoforge_model
                df_load = pd.read_csv(dataset_path)
                train_res = train_autoforge_model(
                    df=df_load,
                    target_column=target_name,
                    time_budget_secs=time_budget,
                    optuna_trials=optuna_trials,
                    model_name=model_reg_name,
                )

        if train_res and train_res.get("status") == "success":
            progress_bar.progress(100, text="Training & Explainability Complete!")
            st.success("✅ Training completed, artifacts persisted, and model registered in MLflow!")

            st.markdown("---")
            st.markdown("### 🏅 Training Run Results")
            k1, k2, k3, k4 = st.columns(4)

            b_acc = train_res["baseline_metrics"].get("accuracy", train_res["baseline_metrics"].get("rmse", 0))
            t_acc = train_res["tuned_metrics"].get("accuracy", train_res["tuned_metrics"].get("rmse", 0))
            metric_label = "Accuracy" if train_res.get("task_type") == "classification" else "RMSE"

            k1.metric(f"Baseline {metric_label}", f"{b_acc:.4f}", f"{train_res['baseline_metrics'].get('duration_seconds')}s")
            k2.metric(f"Tuned {metric_label}", f"{t_acc:.4f}", f"{train_res['tuned_metrics'].get('duration_seconds')}s")
            k3.metric("Winner Model", f"{train_res.get('flaml_winner').upper()}", f"v{train_res.get('registered_version')}")
            k4.metric("Uplift", f"{train_res['uplift']['value']:+.4f} {metric_label}")

            st.json({
                "baseline_run_id": train_res.get("baseline_run_id"),
                "tuned_run_id": train_res.get("tuned_run_id"),
                "registered_model": train_res.get("registered_model_name"),
                "registered_version": train_res.get("registered_version"),
                "pipeline_path": train_res.get("pipeline_artifact_path"),
            })


# ===========================================================================
# PAGE 3: Leaderboard
# ===========================================================================
elif page.startswith("3."):
    st.markdown("<div class='main-header'>Experiment Leaderboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Comprehensive comparison table of baseline benchmarks and tuned model runs from MLflow.</div>", unsafe_allow_html=True)

    runs = []
    if api_ok:
        try:
            r = requests.get(f"{API_URL}/runs", timeout=5)
            if r.status_code == 200:
                runs = r.json().get("runs", [])
        except Exception:
            pass

    if not runs:
        # Check SQLite MLflow directly
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(ARTIFACTS_DIR)}/mlflow.db")
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("autoforge_experiments")
            if exp:
                mlflow_runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
                for mr in mlflow_runs:
                    runs.append({
                        "run_id": mr.info.run_id,
                        "run_name": mr.data.tags.get("mlflow.runName", "unnamed"),
                        "stage": mr.data.tags.get("stage", "unknown"),
                        "task_type": mr.data.tags.get("task_type", ""),
                        "model_type": mr.data.tags.get("model_type", ""),
                        "metrics": mr.data.metrics,
                        "start_time": mr.info.start_time,
                    })
        except Exception:
            pass

    if runs:
        rows = []
        for r in runs:
            m = r.get("metrics", {})
            rows.append({
                "Run ID": r.get("run_id")[:10] + "...",
                "Stage": r.get("stage", "").upper(),
                "Model Type": r.get("model_type", "sklearn"),
                "Task": r.get("task_type", ""),
                "Accuracy": f"{m.get('accuracy', 0):.4f}" if "accuracy" in m else "-",
                "F1 Score": f"{m.get('f1', 0):.4f}" if "f1" in m else "-",
                "RMSE": f"{m.get('rmse', 0):.4f}" if "rmse" in m else "-",
                "R2": f"{m.get('r2', 0):.4f}" if "r2" in m else "-",
                "Duration (s)": f"{m.get('duration_seconds', 0):.2f}",
                "Full Run ID": r.get("run_id"),
            })
        df_lb = pd.DataFrame(rows)
        st.dataframe(df_lb.drop(columns=["Full Run ID"]), use_container_width=True)

        # Baseline vs Tuned comparison
        baseline_runs = [r for r in runs if r.get("stage") == "baseline"]
        tuned_runs = [r for r in runs if r.get("stage") == "tuned"]

        if baseline_runs and tuned_runs:
            st.markdown("### 📊 Baseline vs Tuned Performance Delta")
            b_latest = baseline_runs[0]["metrics"]
            t_latest = tuned_runs[0]["metrics"]

            c1, c2, c3 = st.columns(3)
            if "accuracy" in b_latest and "accuracy" in t_latest:
                delta = t_latest["accuracy"] - b_latest["accuracy"]
                c1.metric("Accuracy Uplift", f"{t_latest['accuracy']:.4f}", f"{delta:+.4f} vs Baseline")
            if "f1" in b_latest and "f1" in t_latest:
                delta_f1 = t_latest["f1"] - b_latest["f1"]
                c2.metric("F1 Score Uplift", f"{t_latest['f1']:.4f}", f"{delta_f1:+.4f} vs Baseline")
            if "duration_seconds" in t_latest:
                c3.metric("AutoML Search & Tune Duration", f"{t_latest['duration_seconds']:.1f}s")
    else:
        st.info("No MLflow experiment runs found yet. Launch an AutoML training run in the Training tab first!")


# ===========================================================================
# PAGE 4: Model Explainability
# ===========================================================================
elif page.startswith("4."):
    st.markdown("<div class='main-header'>Model Explainability & Feature Attribution</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Computed at training time: Global SHAP value distributions and interactive Shapash reports.</div>", unsafe_allow_html=True)

    # Discover available explanation files in artifacts/reports
    reports_dir = os.path.join(ARTIFACTS_DIR, "reports")
    explanation_files = [f for f in os.listdir(reports_dir) if f.endswith("_explanation.json")] if os.path.exists(reports_dir) else []

    if explanation_files:
        chosen_file = st.selectbox("Select Model Explanation Run:", explanation_files)
        with open(os.path.join(reports_dir, chosen_file), "r") as f:
            exp_data = json.load(f)

        run_id = exp_data.get("run_id")
        st.markdown(f"**Run ID:** `{run_id}` &nbsp; | &nbsp; **Model:** `{exp_data.get('model_class')}` &nbsp; | &nbsp; **Task:** `{exp_data.get('task_type')}`")

        tab_shap, tab_shapash, tab_features = st.tabs(["📊 SHAP Summary Plot", "🌟 Shapash Interactive Report", "📋 Top Features Table"])

        with tab_shap:
            png_path = exp_data.get("shap_summary_png")
            if png_path and os.path.exists(png_path):
                st.image(png_path, caption=f"SHAP Summary Plot for {exp_data.get('model_name')}", use_column_width=True)
            else:
                st.warning("SHAP summary plot image not found on disk.")

        with tab_shapash:
            html_path = exp_data.get("shapash_html_report")
            if html_path and os.path.exists(html_path):
                st.caption(f"Interactive Shapash HTML Report ({os.path.getsize(html_path)/1024:.1f} KB)")
                with open(html_path, "r", encoding="utf-8") as f:
                    shapash_html = f.read()
                components.html(shapash_html, height=750, scrolling=True)
            else:
                st.warning("Shapash HTML report not found on disk.")

        with tab_features:
            top_f = exp_data.get("top_features", [])
            if top_f:
                df_top = pd.DataFrame(top_f)
                st.dataframe(df_top, use_container_width=True)
                # Plot horizontal bar chart
                st.bar_chart(df_top.set_index("feature")["importance"])
            else:
                st.info("No feature importance array found.")
    else:
        st.info("No model explainability artifacts found yet. Train a model in the Training tab to generate SHAP reports!")


# ===========================================================================
# PAGE 5: Model Deployment
# ===========================================================================
elif page.startswith("5."):
    st.markdown("<div class='main-header'>Model Registry & Live Deployment</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Manage model registry versions, hot-swap the active inference service, and test live predictions.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📦 Model Registry Versions")
        models_data = []
        if api_ok:
            try:
                m_resp = requests.get(f"{API_URL}/models", timeout=3)
                if m_resp.status_code == 200:
                    models_data = m_resp.json().get("registered_models", [])
            except Exception:
                pass

        if models_data:
            model_options = [m["name"] for m in models_data]
            selected_model = st.selectbox("Select Registered Model:", model_options)
            chosen_obj = next((m for m in models_data if m["name"] == selected_model), None)
            versions = [v["version"] for v in chosen_obj.get("latest_versions", [])] if chosen_obj else ["1"]
            selected_ver = st.selectbox("Select Version:", versions)

            deploy_btn = st.button("🚢 Deploy to Inference Service", type="primary")
            if deploy_btn:
                if inf_ok:
                    dep_res = requests.post(
                        f"{INFERENCE_URL}/model/load",
                        json={"model_name": selected_model, "version": selected_ver},
                    )
                    if dep_res.status_code == 200:
                        st.success(f"Successfully deployed '{selected_model}' version {selected_ver} to Inference Service!")
                    else:
                        st.error(f"Deployment failed: {dep_res.text}")
                else:
                    st.error("Inference service is currently offline.")
        else:
            st.info("No models registered in MLflow registry yet. Train a model to register it!")
            # Fallback to local artifacts/models/latest_pipeline.pkl
            local_pkl = os.path.join(ARTIFACTS_DIR, "models", "latest_pipeline.pkl")
            if os.path.exists(local_pkl):
                st.markdown(f"**Found local model pipeline:** `{local_pkl}`")
                if st.button("Deploy Local Pipeline to Inference Service"):
                    if inf_ok:
                        dep_res = requests.post(
                            f"{INFERENCE_URL}/model/load",
                            json={"pipeline_path": local_pkl, "model_name": "local_pipeline", "version": "1"},
                        )
                        st.success("Deployed local pipeline successfully!")

    with col2:
        st.markdown("### 📡 Active Inference Service Status")
        if inf_ok:
            h = inf_data
            st.markdown(f"**Status:** `🟢 Running (Port 8001)`")
            st.markdown(f"**Model Loaded:** `{h.get('model_loaded')}`")
            st.markdown(f"**Active Model:** `{h.get('model_name')}` (version `{h.get('model_version')}`)")
            st.markdown(f"**Total Requests Handled:** `{h.get('total_requests_logged')}`")
        else:
            st.markdown("**Status:** `🔴 Inference Service Offline`")

    st.markdown("---")
    st.markdown("### 🧪 Live Inference Playground")
    st.markdown("Send real feature payloads to the `/predict` endpoint and measure live p95 response latencies.")

    # Load seed dataset for easy testing
    if os.path.exists("data/titanic.csv"):
        df_sample = pd.read_csv("data/titanic.csv").drop(columns=["Survived"])
        selected_idx = st.slider("Select Sample Row from Titanic:", min_value=0, max_value=len(df_sample)-1, value=0)
        sample_dict = df_sample.iloc[selected_idx].where(pd.notnull(df_sample.iloc[selected_idx]), None).to_dict()

        col_input, col_pred = st.columns([1, 1])
        with col_input:
            st.markdown("**Payload Features (Editable JSON):**")
            input_json_str = st.text_area("Input JSON:", value=json.dumps([sample_dict], indent=2), height=220)

        with col_pred:
            st.markdown("**Prediction Output:**")
            if st.button("⚡ Send POST /predict Request", type="primary"):
                try:
                    payload_rows = json.loads(input_json_str)
                    t_start = time.perf_counter()
                    p_res = requests.post(f"{INFERENCE_URL}/predict", json={"rows": payload_rows})
                    t_ms = (time.perf_counter() - t_start) * 1000

                    if p_res.status_code == 200:
                        pred_json = p_res.json()
                        st.success(f"Prediction: **{pred_json['predictions']}**")
                        if pred_json.get("probabilities"):
                            st.markdown(f"**Class Probabilities:** `{pred_json['probabilities']}`")
                        st.metric("Service Latency", f"{pred_json.get('latency_ms', t_ms):.2f} ms")
                    else:
                        st.error(f"Inference error ({p_res.status_code}): {p_res.text}")
                except Exception as e:
                    st.error(f"Invalid JSON or request failed: {e}")


# ===========================================================================
# PAGE 6: Monitoring & Drift
# ===========================================================================
elif page.startswith("6."):
    st.markdown("<div class='main-header'>Data Drift Monitoring & Auto-Retraining</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Continuous Evidently comparison of production inference traffic vs training reference data.</div>", unsafe_allow_html=True)

    # 1. Trigger rules documentation
    st.markdown("""
    <div class='metric-card'>
        <span class='metric-title'>Auto-Retraining Trigger Rules</span>
        <p style='margin: 8px 0 0 0; color: #d1d5db; font-size: 0.95rem;'>
            Drift is automatically evaluated by the <strong>drift-worker</strong> every 5 minutes comparing recent SQLite inference requests against the training reference snapshot.<br>
            🚨 <strong>Trigger Condition:</strong> <code>Drifted Feature Share &gt; 30%</code> OR <code>Any Feature PSI &gt; 0.20</code>.<br>
            When triggered, an automated retraining run is triggered via the API and a new version is registered in MLflow.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Check recent inference logs in SQLite
    from inference.request_logger import RequestLogger
    r_logger = RequestLogger()
    recent_logs = r_logger.get_recent_requests(limit=50)

    c1, c2, c3 = st.columns(3)
    c1.metric("Production Requests Logged", f"{r_logger.count_requests():,}")
    c2.metric("Reference Dataset Rows", "712 (Titanic)")
    c3.metric("Drift Trigger Threshold", "> 30% features / PSI > 0.2")

    st.markdown("### 🚦 Recent Production Request Logs (SQLite Store)")
    if not recent_logs.empty:
        st.dataframe(recent_logs.head(10), use_container_width=True)
    else:
        st.info("No inference requests logged yet. Send predictions in the Deployment tab or run `simulate_drift.py`!")

    # 3. Check for Evidently Drift Report HTML
    drift_report_path = os.path.join(ARTIFACTS_DIR, "reports", "drift_report.html")
    drift_meta_path = os.path.join(ARTIFACTS_DIR, "reports", "drift_metrics.json")

    st.markdown("---")
    st.markdown("### 🔬 Evidently Drift Analysis Report")
    
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        run_drift_now = st.button("🔍 Run Drift Check Now", type="primary")

    if run_drift_now:
        with st.spinner("Comparing recent requests against reference snapshot with Evidently..."):
            from core.drift import check_and_evaluate_drift
            drift_res = check_and_evaluate_drift()
            if drift_res.get("status") == "success":
                st.success(f"Drift evaluation completed! (Drift detected: {drift_res.get('drift_detected')})")
            else:
                st.warning(f"Drift check note: {drift_res.get('message')}")

    if os.path.exists(drift_meta_path):
        with open(drift_meta_path, "r") as f:
            d_meta = json.load(f)
        d_share = d_meta.get("drift_share", 0)
        d_detected = d_meta.get("drift_detected", False)
        badge = "<span class='badge-drift'>DRIFT DETECTED</span>" if d_detected else "<span class='badge-class'>NO DRIFT</span>"
        st.markdown(f"**Latest Drift Status:** {badge} &nbsp; (Drift Share: **{d_share:.1%}**)", unsafe_allow_html=True)

    if os.path.exists(drift_report_path):
        st.markdown(f"**Embedded Evidently Report:** `{drift_report_path}`")
        with open(drift_report_path, "r", encoding="utf-8") as f:
            drift_html = f.read()
        components.html(drift_html, height=750, scrolling=True)
    else:
        st.info("Evidently HTML drift report will be generated after running the drift check or running `scripts/simulate_drift.py`.")
