from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def _coerce_to_array(series_like: Iterable[Any]) -> np.ndarray:
    """
    Coerce an iterable of labels into a 1D numpy array without modifying values.
    Strings and integers are preserved to maintain class names.
    """
    if isinstance(series_like, (pd.Series, pd.Index)):
        return series_like.to_numpy()
    if isinstance(series_like, pd.DataFrame):
        if series_like.shape[1] != 1:
            raise ValueError("Expected a single column when passing a DataFrame.")
        return series_like.iloc[:, 0].to_numpy()
    return np.asarray(list(series_like))


def compute_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    labels: Optional[Sequence[Any]] = None,
) -> Tuple[Dict[str, float], np.ndarray, List[Any]]:
    """
    Compute common classification metrics and a confusion matrix.

    Returns:
        - metrics: Mapping containing:
            - accuracy
            - precision_weighted
            - recall_weighted
            - f1_micro
            - f1_macro
            - f1_weighted
        - conf_mat: Confusion matrix (shape: [n_classes, n_classes])
        - ordered_labels: Class labels in the order used by the confusion matrix
    """
    y_true_arr = _coerce_to_array(y_true)
    y_pred_arr = _coerce_to_array(y_pred)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")

    if labels is None:
        # Stable ordering: sort unique labels as strings to avoid type comparison issues
        unique_true = pd.unique(y_true_arr)
        unique_pred = pd.unique(y_pred_arr)
        combined = pd.unique(pd.Series(list(unique_true) + list(unique_pred)))
        # Convert all to string for sorting, but keep original mapping for display
        ordered_labels = list(combined)
        try:
            # Prefer natural sort if possible
            ordered_labels = sorted(ordered_labels)
        except Exception:
            # Fallback to string sort if types are mixed
            ordered_labels = sorted(ordered_labels, key=lambda x: str(x))
    else:
        ordered_labels = list(labels)

    # Accuracy
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))

    # Precision/Recall/F1 aggregates
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=ordered_labels, average="weighted", zero_division=0
    )
    _, _, f1_micro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=ordered_labels, average="micro", zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=ordered_labels, average="macro", zero_division=0
    )

    # Confusion matrix with fixed label order
    conf_mat = confusion_matrix(y_true_arr, y_pred_arr, labels=ordered_labels)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    return metrics, conf_mat, ordered_labels


def metrics_to_dataframe(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Convert metrics mapping into a tidy two-column DataFrame for display.
    """
    return pd.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": [round(float(v), 6) for v in metrics.values()],
        }
    )


def confusion_matrix_to_dataframe(conf_mat: np.ndarray, labels: Sequence[Any]) -> pd.DataFrame:
    """
    Represent a confusion matrix as a labeled DataFrame with rows=true, cols=pred.
    """
    return pd.DataFrame(conf_mat, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="pred"))


