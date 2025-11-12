import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from metrics_utils import (
    compute_metrics,
    metrics_to_dataframe,
    confusion_matrix_to_dataframe,
)
from inference import predict_with_weights


# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(
    page_title="EE6405 Model Evaluator",
    page_icon="📊",
    layout="wide",
)


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(file)


def draw_confusion_matrix_heatmap(conf_mat: np.ndarray, labels: List[Any]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig, clear_figure=True)


def download_bytes_from_df(df: pd.DataFrame, filename: str, label: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📚 Domain & Data")
domain = st.sidebar.selectbox("Select domain", ["Books", "Movies", "Games"], index=0)
st.sidebar.caption("Upload predictions to compute metrics. Use interactive prediction below with trained weights.")


# ----------------------------
# Main Content
# ----------------------------
st.title("EE6405 Model Evaluator")
st.subheader(f"Domain: {domain}")

with st.expander("What this app does", expanded=False):
    st.markdown(
        "- Computes: accuracy, precision (weighted), recall (weighted), micro F1, macro F1, weighted F1\n"
        "- Displays labeled confusion matrix\n"
        "- Supports CSV upload with columns `y_true`, `y_pred` or column mapping\n"
        "- Interactive prediction powered by your trained weights"
    )

left_col, right_col = st.columns([1, 2], gap="large")

metrics_df: Optional[pd.DataFrame] = None
conf_mat_df: Optional[pd.DataFrame] = None

with left_col:
    st.markdown("### Upload predictions")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = read_csv_from_upload(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None and not df.empty:
            st.dataframe(df.head(), use_container_width=True)
            # Column mapping
            st.markdown("#### Map columns")
            cols = list(df.columns)
            col_true = st.selectbox("Ground-truth column", options=cols, index=min(0, len(cols) - 1))
            col_pred = st.selectbox(
                "Predictions column", options=cols, index=min(1, len(cols) - 1) if len(cols) > 1 else 0
            )
            label_hint = st.text_input(
                "Optional: comma-separated class labels (order used in confusion matrix)",
                value="",
                placeholder="e.g., Positive,Negative or 0,1,2",
            )

            if st.button("Evaluate", type="primary"):
                y_true = df[col_true]
                y_pred = df[col_pred]
                labels: Optional[List[Any]] = None
                if label_hint.strip():
                    labels = [x.strip() for x in label_hint.split(",") if x.strip() != ""]
                try:
                    metrics, conf_mat, lbls = compute_metrics(y_true, y_pred, labels=labels)
                    metrics_df = metrics_to_dataframe(metrics)
                    conf_mat_df = confusion_matrix_to_dataframe(conf_mat, lbls)
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
with right_col:
    if metrics_df is not None:
        st.markdown("### Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        st.markdown("### Confusion Matrix")
        draw_confusion_matrix_heatmap(conf_mat_df.values, list(conf_mat_df.columns))

        # Downloads
        st.markdown("#### Downloads")
        download_bytes_from_df(metrics_df, f"{domain.lower()}_metrics.csv", "Download metrics CSV")
        download_bytes_from_df(conf_mat_df.reset_index(), f"{domain.lower()}_confusion_matrix.csv", "Download confusion matrix CSV")
    else:
        st.info("Upload a CSV and click Evaluate to see results.")


st.caption(
    "Accuracy, precision (weighted), recall (weighted), micro/macro/weighted F1 reported. "
    "Confusion matrix uses the displayed label order."
)

# ----------------------------
# Interactive Prediction (Trained Weights)
# ----------------------------
st.markdown("---")
st.header("Interactive Genre Prediction")
st.write("Enter a description to get a prediction using your trained weights (TorchScript bundle).")

pred_col_left, pred_col_right = st.columns([2, 1])

with pred_col_left:
    user_text = st.text_area(
        "Description",
        height=160,
        placeholder="Paste a synopsis, blurb, or description here...",
    )

with pred_col_right:
    st.caption("Requires `artifacts/<domain>/model.ts`, `labels.txt`, and `preprocessor.json`.")

go = st.button("Predict Genre", type="primary")

if go:
    if not user_text or not user_text.strip():
        st.warning("Please enter a description.")
    else:
        # Trained weights path: requires standardized bundle artifacts/{domain}/
        try:
            pred_label, model_labels, probabilities = predict_with_weights(domain, user_text)
            st.success(f"Predicted genre: {pred_label}")
            prob_df = pd.DataFrame(
                {"label": model_labels, "probability": probabilities.astype(float)}
            ).sort_values("probability", ascending=False, ignore_index=True)
            st.dataframe(prob_df, use_container_width=True, height=300)
            st.bar_chart(prob_df.set_index("label"))
            csv_bytes = prob_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download probabilities CSV",
                data=csv_bytes,
                file_name=f"{domain.lower()}_prediction_probs.csv",
                mime="text/csv",
            )
        except FileNotFoundError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Prediction with trained weights failed: {e}")