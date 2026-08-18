"""
ML Assignment 2 - Streamlit App
Interactive dashboard to demonstrate 6 classification models
(Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, SVM)
trained on the Breast Cancer Wisconsin (Diagnostic) dataset.

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FEATURE_INFO_PATH = BASE_DIR / "feature_info.json"
DEFAULT_TEST_DATA_PATH = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
    "SVM": "svm.joblib",
}

ACCENT = "#5B6CF9"
ACCENT_DARK = "#3E4CD6"
BG = "#0F1116"
CARD = "#171A23"
CARD_BORDER = "#262B3A"
TEXT_MUTED = "#9AA3B5"

st.set_page_config(
    page_title="Classification Model Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme / CSS - clean dashboard look
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        .stApp {{
            background: radial-gradient(circle at top left, #161925 0%, {BG} 45%);
        }}

        [data-testid="stSidebar"] {{
            background: #10121A;
            border-right: 1px solid {CARD_BORDER};
        }}

        section.main > div {{
            padding-top: 1.2rem;
        }}

        h1, h2, h3, h4 {{
            letter-spacing: -0.01em;
        }}

        /* Hero header */
        .dash-hero {{
            background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px rgba(91, 108, 249, 0.25);
        }}
        .dash-hero h1 {{
            color: white;
            font-size: 1.9rem;
            margin: 0 0 6px 0;
        }}
        .dash-hero p {{
            color: rgba(255,255,255,0.85);
            margin: 0;
            font-size: 0.95rem;
        }}
        .dash-hero .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.18);
            color: white;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            margin-top: 12px;
            margin-right: 8px;
        }}

        /* Section cards */
        .dash-card {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 22px 24px;
            margin-bottom: 20px;
        }}
        .dash-card h3 {{
            margin-top: 0;
            font-size: 1.15rem;
            color: #F2F3F7;
        }}
        .dash-card .subtitle {{
            color: {TEXT_MUTED};
            font-size: 0.85rem;
            margin-top: -8px;
            margin-bottom: 14px;
        }}

        /* Metric tiles */
        [data-testid="stMetric"] {{
            background: #12141C;
            border: 1px solid {CARD_BORDER};
            border-radius: 12px;
            padding: 14px 10px 10px 14px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED};
        }}
        [data-testid="stMetricValue"] {{
            color: {ACCENT};
        }}

        /* Dataframes */
        [data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {CARD_BORDER};
        }}

        /* Buttons */
        .stDownloadButton button, .stButton button {{
            background: {ACCENT};
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 500;
        }}
        .stDownloadButton button:hover, .stButton button:hover {{
            background: {ACCENT_DARK};
            color: white;
        }}

        /* Winner banner */
        .winner-banner {{
            background: linear-gradient(135deg, #1D6E4B 0%, #12452F 100%);
            border: 1px solid #2E8B5E;
            border-radius: 14px;
            padding: 16px 20px;
            color: #E7FCEF;
            font-size: 1rem;
            margin-top: 6px;
        }}

        hr {{
            border-color: {CARD_BORDER};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    for display_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[display_name] = joblib.load(path)
    return models


@st.cache_data
def load_feature_info():
    with open(FEATURE_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_default_test_data():
    return pd.read_csv(DEFAULT_TEST_DATA_PATH)


def get_scores(model, X):
    """Return probability/score for the positive class, used for AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_metrics(model, X, y_true):
    y_pred = model.predict(X)
    y_score = get_scores(model, X)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }, y_pred


def card_open(title, subtitle=None):
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="dash-card"><h3>{title}</h3>{subtitle_html}',
        unsafe_allow_html=True,
    )


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar - controls
# --------------------------------------------------------------------------
feature_info = load_feature_info()
feature_names = feature_info["feature_names"]
target_column = feature_info["target_column"]

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.caption(f"{feature_info['dataset_name']}")
    st.markdown(
        f"**Rows:** {feature_info['rows']}   |   **Features:** {feature_info['features']}"
    )
    st.markdown("---")

    st.markdown("**1. Test data**")
    uploaded_file = st.file_uploader(
        "Upload CSV (feature columns + optional target)",
        type=["csv"],
        label_visibility="collapsed",
    )
    use_default = st.checkbox(
        "Use bundled sample test_data.csv", value=uploaded_file is None
    )

    st.markdown("**2. Model**")
    models = load_models()
    model_names = list(models.keys())
    selected_model_name = st.selectbox(
        "Choose a classification model", model_names, label_visibility="collapsed"
    )
    selected_model = models[selected_model_name]

if uploaded_file is not None and not use_default:
    data_df = pd.read_csv(uploaded_file)
    data_source_label = f"Uploaded file: {uploaded_file.name}"
else:
    data_df = load_default_test_data()
    data_source_label = "Bundled sample: test_data.csv"

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="dash-hero">
        <h1>🧪 Classification Model Dashboard</h1>
        <p>Compare 6 classification models trained on {feature_info['dataset_name']}.</p>
        <span class="badge">Dataset: {feature_info['rows']} rows · {feature_info['features']} features</span>
        <span class="badge">Data source: {data_source_label}</span>
        <span class="badge">Active model: {selected_model_name}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Validate uploaded data has required feature columns
missing_cols = [c for c in feature_names if c not in data_df.columns]
if missing_cols:
    st.error(
        "The uploaded CSV is missing required feature columns:\n\n"
        f"{missing_cols}\n\n"
        "Please upload a CSV containing all the model's feature columns."
    )
    st.stop()

has_target = target_column in data_df.columns
X = data_df[feature_names]

# --------------------------------------------------------------------------
# Data preview
# --------------------------------------------------------------------------
card_open("📄 Data Preview", f"{data_df.shape[0]} rows × {data_df.shape[1]} columns")
st.dataframe(data_df.head(10), use_container_width=True)
card_close()

if not has_target:
    st.warning(
        f"No '{target_column}' column found in the uploaded data, so evaluation "
        "metrics, the confusion matrix, and model comparison cannot be computed. "
        "Predictions will still be generated below."
    )
    y_pred = selected_model.predict(X)
    result_df = data_df.copy()
    result_df["predicted"] = y_pred

    card_open(f"🔍 Predictions — {selected_model_name}")
    st.dataframe(result_df, use_container_width=True)
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download predictions as CSV",
        data=csv_bytes,
        file_name=f"predictions_{selected_model_name.replace(' ', '_')}.csv",
        mime="text/csv",
    )
    card_close()
    st.stop()

y_true = data_df[target_column]

# --------------------------------------------------------------------------
# Selected model - metrics
# --------------------------------------------------------------------------
metrics, y_pred = compute_metrics(selected_model, X, y_true)

card_open(f"📌 Evaluation Metrics — {selected_model_name}")
metric_cols = st.columns(6)
metric_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
for col, label in zip(metric_cols, metric_labels):
    value = metrics[label]
    col.metric(label, f"{value:.4f}" if not np.isnan(value) else "N/A")
card_close()

# --------------------------------------------------------------------------
# Confusion matrix + classification report
# --------------------------------------------------------------------------
card_open("🧩 Confusion Matrix & Classification Report")
col_cm, col_report = st.columns([1, 1.3])

with col_cm:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{selected_model_name}", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for text in ax.texts:
        text.set_color("black")
    st.pyplot(fig, use_container_width=True)

with col_report:
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

result_df = data_df.copy()
result_df["predicted"] = y_pred
csv_bytes = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download predictions as CSV",
    data=csv_bytes,
    file_name=f"predictions_{selected_model_name.replace(' ', '_')}.csv",
    mime="text/csv",
)
card_close()

# --------------------------------------------------------------------------
# Compare all models
# --------------------------------------------------------------------------
card_open("📊 Compare All Models", "Same test data, evaluated across all 6 models")

rows = []
for name, model in models.items():
    m, _ = compute_metrics(model, X, y_true)
    rows.append({"ML Model Name": name, **m})

comparison_df = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(
    comparison_df.style.highlight_max(axis=0, color="#1D6E4B"),
    use_container_width=True,
)

st.markdown("##### Metric comparison charts")
all_metric_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
chart_cols = st.columns(3)
for i, metric_name in enumerate(all_metric_labels):
    with chart_cols[i % 3]:
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.2))
        fig2.patch.set_facecolor(CARD)
        ax2.set_facecolor(CARD)
        ax2.bar(comparison_df.index, comparison_df[metric_name], color=ACCENT)
        ax2.set_title(metric_name, color="white", fontsize=11)
        ax2.tick_params(colors="white", labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color(CARD_BORDER)
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

best_model = comparison_df["F1"].idxmax()
st.markdown(
    f'<div class="winner-banner">🏆 Best model on this data (by F1 score): '
    f"<b>{best_model}</b></div>",
    unsafe_allow_html=True,
)

csv_bytes2 = comparison_df.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download comparison table as CSV",
    data=csv_bytes2,
    file_name="model_comparison_results.csv",
    mime="text/csv",
)
card_close()
