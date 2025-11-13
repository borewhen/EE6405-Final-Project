import io
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from metrics_utils import (
    metrics_to_dataframe,
    confusion_matrix_to_dataframe,
)
from inference import predict_with_weights

# Optional explainability deps (loaded lazily where used)
try:
    import lime  # type: ignore
    from lime.lime_text import LimeTextExplainer  # type: ignore
except Exception:
    LimeTextExplainer = None  # type: ignore
try:
    import shap  # type: ignore
except Exception:
    shap = None  # type: ignore
import re
import logging


# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(
    page_title="EE6405 Model Evaluator",
    page_icon="📊",
    layout="wide",
)


# ----------------------------
# Logging
# ----------------------------
if "_log_configured" not in st.session_state:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    st.session_state["_log_configured"] = True
logger = logging.getLogger("ee6405_app")


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(file)


def draw_confusion_matrix_heatmap(conf_mat: np.ndarray, labels: List[Any], width: int = 500, dpi: int = 200, title: str = "Confusion Matrix") -> None:
    # Render to image and control container size via st.image width
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 8},
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    # Control the absolute container footprint via width
    st.image(buf, width=width)
    plt.close(fig)


def download_bytes_from_df(df: pd.DataFrame, filename: str, label: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

# ----------------------------
# Main Content
# ----------------------------
st.title("EE6405 Model Evaluator")
st.text("AY2025/26 Semester 1, Table A8")
st.text("Dave Marteen Gunawan, Jin Zixuan, John Ang Yi Heng, Shen Bowen, Wu Huaye")
domain = st.selectbox("Select domain", ["Books", "Movies", "Games"], index=0)

debug_logging = st.checkbox("Enable debug logging", value=False, help="Writes detailed logs to your console")
if debug_logging:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
logger.info("Domain selected: %s", domain)

metrics_df: Optional[pd.DataFrame] = None
conf_mat_df: Optional[pd.DataFrame] = None

# ----------------------------
# Artifact-backed Results Loader
# ----------------------------
def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner=False)
def load_artifact_results(domain: str) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray], Optional[List[Any]]]:
    """
    Load previously exported experiment results for a domain from artifacts/<domain>/.

    Supported files:
      - metrics.json (preferred) or results.json with {"metrics": {...}, "confusion_matrix": [...], "labels": [...]}
      - confusion_matrix.csv (DataFrame: rows=true, cols=pred)
      - confusion_matrix.npy (numpy array)
      - labels.txt (fallback labels if not embedded in results.json/CSV header)
    """
    root = _project_root()
    domain_key = domain.strip().lower()
    bundle_dir = os.path.join(root, "artifacts", domain_key)

    metrics: Optional[Dict[str, float]] = None
    conf_mat: Optional[np.ndarray] = None
    labels: Optional[List[Any]] = None

    metrics_path = os.path.join(bundle_dir, "metrics.json")
    results_path = os.path.join(bundle_dir, "results.json")
    conf_csv_path = os.path.join(bundle_dir, "confusion_matrix.csv")
    conf_npy_path = os.path.join(bundle_dir, "confusion_matrix.npy")
    labels_path = os.path.join(bundle_dir, "labels.txt")

    # metrics.json
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    # results.json (optional combined file)
    if metrics is None and os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            metrics = obj.get("metrics") or obj.get("test_metrics")
            cm = obj.get("confusion_matrix")
            if cm is not None:
                conf_mat = np.asarray(cm, dtype=float)
            labels = obj.get("labels")
        except Exception:
            pass

    # confusion_matrix.csv
    if conf_mat is None and os.path.exists(conf_csv_path):
        try:
            df = pd.read_csv(conf_csv_path, index_col=0)
            labels = list(df.columns)
            conf_mat = df.values
        except Exception:
            conf_mat = None

    # confusion_matrix.npy
    if conf_mat is None and os.path.exists(conf_npy_path):
        try:
            conf_mat = np.load(conf_npy_path)
        except Exception:
            conf_mat = None

    # labels.txt (fallback labels)
    if labels is None and os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
        except Exception:
            labels = None

    return metrics, conf_mat, labels


@st.cache_data(show_spinner=False)
def load_samples_table(domain: str, model_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load a small sample table of predictions vs actual from artifacts/<domain>/(samples.csv|samples_{model}.csv), if present.
    Expected columns (flexible): text, true, pred, correct, prob_pred, true_labels, pred_labels
    """
    root = _project_root()
    domain_key = domain.strip().lower()
    bundle_dir = os.path.join(root, "artifacts", domain_key)
    samples_csv = (
        os.path.join(bundle_dir, f"samples_{model_key}.csv")
        if model_key
        else os.path.join(bundle_dir, "samples.csv")
    )
    if model_key and not os.path.exists(samples_csv):
        samples_csv = os.path.join(bundle_dir, "samples.csv")
    if not os.path.exists(samples_csv):
        return None
    try:
        df = pd.read_csv(samples_csv)
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_per_model_metrics(domain: str) -> Dict[str, Dict[str, float]]:
    """
    Load metrics for each model type if exported separately, e.g. metrics_rnn.json, metrics_lstm.json, metrics_gru.json.
    Returns mapping { model_key -> metrics_dict } for keys in {"rnn","lstm","gru"} if found.
    """
    root = _project_root()
    domain_key = domain.strip().lower()
    bundle_dir = os.path.join(root, "artifacts", domain_key)

    results: Dict[str, Dict[str, float]] = {}
    for mk in ("rnn", "lstm", "gru"):
        mk_path = os.path.join(bundle_dir, f"metrics_{mk}.json")
        alt_path = os.path.join(bundle_dir, f"results_{mk}.json")
        metrics_obj: Optional[Dict[str, float]] = None
        if os.path.exists(mk_path):
            try:
                with open(mk_path, "r", encoding="utf-8") as f:
                    metrics_obj = json.load(f)
            except Exception:
                metrics_obj = None
        elif os.path.exists(alt_path):
            try:
                with open(alt_path, "r", encoding="utf-8") as f:
                    robj = json.load(f)
                metrics_obj = robj.get("metrics") or robj.get("test_metrics")
            except Exception:
                metrics_obj = None
        if isinstance(metrics_obj, dict) and metrics_obj:
            results[mk] = metrics_obj
    return results


# ----------------------------
# Explainability helpers (LIME/SHAP)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_per_model_confusions(domain: str) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Load per-model confusion matrices if present: confusion_matrix_{rnn,lstm,gru}.csv
    Returns mapping { model_key -> (matrix, labels) }.
    """
    root = _project_root()
    domain_key = domain.strip().lower()
    bundle_dir = os.path.join(root, "artifacts", domain_key)
    out: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for mk in ("rnn", "lstm", "gru"):
        path = os.path.join(bundle_dir, f"confusion_matrix_{mk}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0)
                labels = list(df.columns)
                out[mk] = (df.values, labels)
            except Exception:
                continue
    return out

@st.cache_data(show_spinner=False)
def compute_genre_counts(domain: str) -> Optional[Counter]:
    """
    Build genre frequency counts primarily from artifacts samples, with fallbacks:
      1) samples_{model}.csv (true or true_labels)
      2) samples.csv
      3) confusion_matrix.csv (row sums interpreted as true label counts)
    Returns a Counter or None if nothing available.
    """
    root = _project_root()
    domain_key = domain.strip().lower()
    bundle_dir = os.path.join(root, "artifacts", domain_key)

    def _split_multi(s: str) -> List[str]:
        parts = re.split(r"[|,/;]+", str(s))
        return [p.strip() for p in parts if p and p.strip()]

    # 1) Per-model samples
    for mk in ("rnn", "lstm", "gru"):
        path = os.path.join(bundle_dir, f"samples_{mk}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                counts: Counter = Counter()
                if "true" in df.columns:
                    for v in df["true"].dropna().astype(str):
                        counts[v.strip()] += 1
                elif "true_labels" in df.columns:
                    for v in df["true_labels"].dropna().astype(str):
                        for lbl in _split_multi(v):
                            counts[lbl] += 1
                if counts:
                    return counts
            except Exception:
                pass

    # 2) Combined samples
    path = os.path.join(bundle_dir, "samples.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            counts = Counter()
            if "true" in df.columns:
                for v in df["true"].dropna().astype(str):
                    counts[v.strip()] += 1
            elif "true_labels" in df.columns:
                for v in df["true_labels"].dropna().astype(str):
                    for lbl in _split_multi(v):
                        counts[lbl] += 1
            if counts:
                return counts
        except Exception:
            pass

    # 3) Confusion matrix row sums
    cm_path = os.path.join(bundle_dir, "confusion_matrix.csv")
    if os.path.exists(cm_path):
        try:
            df = pd.read_csv(cm_path, index_col=0)
            labels = list(df.columns)
            row_sums = df.sum(axis=1).tolist()
            counts = Counter({str(lbl): int(val) for lbl, val in zip(labels, row_sums)})
            if counts:
                return counts
        except Exception:
            pass

    return None

@st.cache_data(show_spinner=False)
def compute_dataset_genre_counts(domain: str) -> Optional[Counter]:
    """
    Compute genre/category frequency directly from the raw datasets in the repo.
      - Books: Books/BooksDatasetClean.csv, column 'Category'
      - Movies: Movies/wiki_movie_plots_deduped.csv, columns like 'Genre'/'Genres'
      - Games: Games/steam.csv or Games/steam_description_data.csv, columns like 'genre'/'genres'
    """
    root = _project_root()
    dkey = domain.strip().lower()

    def _split_multi(val: str) -> List[str]:
        parts = re.split(r"[|,/;]+", str(val))
        return [p.strip() for p in parts if p and p.strip()]

    try:
        # Helper: normalize token into one of the known labels (when available)
        def _norm_token_to_label(tok: str, dkey: str, labels: Optional[List[str]]) -> Optional[str]:
            t = str(tok).strip()
            tl = t.lower()
            # Common trims
            tl = tl.replace("&", "and").replace("_", " ").replace("-", " ").strip()
            # Domain-specific synonyms
            if dkey == "games":
                if tl == "massively multiplayer":
                    t = "MMO"
                elif tl in ("role playing", "roleplaying", "rpg"):
                    t = "RPG"
                else:
                    t = t
            elif dkey == "movies":
                if tl in ("science fiction", "sci fi", "sci–fi", "scifi"):
                    t = "Sci-Fi"
                else:
                    t = t.title()
            elif dkey == "books":
                t = t.lower()
            # If we have a labels list, match against it (case-insensitive)
            if labels:
                for L in labels:
                    if str(L).strip().lower() == str(t).strip().lower():
                        return L
                # For Books, map unknowns to 'other' if present
                if dkey == "books" and any(str(L).strip().lower() == "other" for L in labels):
                    return next(L for L in labels if str(L).strip().lower() == "other")
                return None
            return t

        if dkey == "books":
            csv_path = os.path.join(root, "Books", "BooksDatasetClean.csv")
            if not os.path.exists(csv_path):
                return None
            df = pd.read_csv(csv_path)
            if "Category" not in df.columns:
                return None
            # Try to align with model labels if present
            labels_txt = os.path.join(root, "artifacts", dkey, "labels.txt")
            labels: Optional[List[str]] = None
            if os.path.exists(labels_txt):
                with open(labels_txt, "r", encoding="utf-8") as f:
                    labels = [line.strip() for line in f if line.strip()]
            cnt: Counter = Counter()
            for v in df["Category"].dropna().astype(str):
                for g in _split_multi(v):
                    mapped = _norm_token_to_label(g, dkey, labels)
                    if mapped and str(mapped).strip().lower() != "unknown":
                        cnt[mapped] += 1
            return cnt if cnt else None

        if dkey == "movies":
            csv_path = os.path.join(root, "Movies", "wiki_movie_plots_deduped.csv")
            if not os.path.exists(csv_path):
                return None
            df = pd.read_csv(csv_path)
            labels_txt = os.path.join(root, "artifacts", dkey, "labels.txt")
            labels: Optional[List[str]] = None
            if os.path.exists(labels_txt):
                with open(labels_txt, "r", encoding="utf-8") as f:
                    labels = [line.strip() for line in f if line.strip()]
            for col in ["Genre", "Genres", "genre", "genres"]:
                if col in df.columns:
                    cnt = Counter()
                    for v in df[col].dropna().astype(str):
                        for g in _split_multi(v):
                            mapped = _norm_token_to_label(g, dkey, labels)
                            if mapped and str(mapped).strip().lower() != "unknown":
                                cnt[mapped] += 1
                    return cnt if cnt else None
            return None

        if dkey == "games":
            paths = [
                os.path.join(root, "Games", "steam.csv"),
                os.path.join(root, "Games", "steam_description_data.csv"),
            ]
            cnt = Counter()
            labels_txt = os.path.join(root, "artifacts", dkey, "labels.txt")
            labels: Optional[List[str]] = None
            if os.path.exists(labels_txt):
                with open(labels_txt, "r", encoding="utf-8") as f:
                    labels = [line.strip() for line in f if line.strip()]
            for p in paths:
                if os.path.exists(p):
                    try:
                        df = pd.read_csv(p)
                        for col in ["genre", "genres", "Genre", "Genres"]:
                            if col in df.columns:
                                for v in df[col].dropna().astype(str):
                                    for g in _split_multi(v):
                                        mapped = _norm_token_to_label(g, dkey, labels)
                                        if mapped and str(mapped).strip().lower() != "unknown":
                                            cnt[mapped] += 1
                    except Exception:
                        continue
            return cnt if cnt else None

        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def compute_notebook_style_counts(domain: str) -> Optional[Counter]:
    """
    Mimic each notebook's early preprocessing up to the genre/category frequency plot.
    This intentionally prioritizes faithfulness over optimization.
    """
    root = _project_root()
    dkey = domain.strip().lower()

    def _split_books(text: str) -> List[str]:
        # Books.split_category: split on commas, slashes, pipes, whitespace; strip; dedupe per entry
        parts = re.split(r'[,/|\s]+', str(text))
        parts = [p.strip() for p in parts if p.strip()]
        # remove lone ampersand tokens
        parts = [p for p in parts if p != '&']
        # maintain order but unique within entry
        seen = {}
        for p in parts:
            if p not in seen:
                seen[p] = True
        return list(seen.keys())

    def _split_generic(text: str) -> List[str]:
        # For Movies/Games: split on common separators
        parts = re.split(r"[|,/;]+", str(text))
        return [p.strip() for p in parts if p and p.strip()]

    try:
        if dkey == "books":
            # EXACT mimic of early notebook steps
            csv_path = os.path.join(root, "Books", "BooksDatasetClean.csv")
            if not os.path.exists(csv_path):
                return None
            df = pd.read_csv(csv_path)
            if "Description" not in df.columns or "Category" not in df.columns:
                return None
            df = df.dropna(subset=['Description', 'Category'])

            # split_category from notebook
            def split_category(text: str) -> List[str]:
                if not isinstance(text, str):
                    return []
                parts = re.split(r'[,/|\s]+', text.lower())
                parts = [p.strip() for p in parts if p.strip()]
                # dict.fromkeys preserves order and removes dups
                return list(dict.fromkeys(parts))

            category_list: List[List[str]] = []
            for category_str in df["Category"]:
                category_list.append(split_category(category_str))
            # remove lone '&' tokens
            category_list = [[c for c in sublist if c != '&'] for sublist in category_list]

            from collections import Counter as _Ctr
            EXCLUDE_CATEGORIES = {"fiction", "nonfiction", "general"}
            flat = [g.strip() for sub in category_list for g in sub if g and str(g).strip()]
            filtered = [g for g in flat if g.lower() not in EXCLUDE_CATEGORIES]
            return _Ctr(filtered) if filtered else None

        elif dkey == "movies":
            # EXACT mimic: use 'Genre' column and drop 'unknown' rows first
            csv_path = os.path.join(root, "Movies", "wiki_movie_plots_deduped.csv")
            if not os.path.exists(csv_path):
                return None
            df = pd.read_csv(csv_path)
            if "Genre" not in df.columns:
                return None
            df = df[df["Genre"] != "unknown"]

            def split_genres(text: str) -> List[str]:
                if not isinstance(text, str):
                    return []
                parts = re.split(r'[,/|\s]+', text.lower())
                parts = [p.strip() for p in parts if p.strip()]
                return list(dict.fromkeys(parts))

            genre_list: List[List[str]] = []
            for genre_str in df["Genre"]:
                genre_list.append(split_genres(genre_str))
            all_genres = [g for sublist in genre_list for g in sublist]
            return Counter(all_genres) if all_genres else None

        elif dkey == "games":
            # EXACT mimic: merge description and genres CSVs, then split genres by ';' and lowercase
            desc_path = os.path.join(root, "Games", "steam_description_data.csv")
            data_path = os.path.join(root, "Games", "steam.csv")
            if not (os.path.exists(desc_path) and os.path.exists(data_path)):
                return None
            description_df = pd.read_csv(desc_path)
            data_df = pd.read_csv(data_path)
            if not all(col in description_df.columns for col in ["steam_appid", "detailed_description"]):
                return None
            if "appid" not in data_df.columns or "genres" not in data_df.columns:
                return None
            desc_df_subset = description_df[['steam_appid', 'detailed_description']]
            data_df_subset = data_df[['appid', 'genres']]
            merged_df = pd.merge(desc_df_subset, data_df_subset, left_on='steam_appid', right_on='appid', how='inner')
            df = merged_df.dropna(subset=['detailed_description', 'genres'])

            genre_list_raw: List[List[str]] = []
            for genre_str in df["genres"]:
                temp = str(genre_str).split(';')
                temp = [x.lower() for x in temp]
                genre_list_raw.append(temp)

            all_genres = [g for sublist in genre_list_raw for g in sublist]
            return Counter(all_genres) if all_genres else None

        else:
            return None
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _load_bundle_for_explain(domain_key: str):
    # Load artifacts and TorchScript model once for explanations
    root = _project_root()
    dkey = domain_key.strip().lower()
    bundle_dir = os.path.join(root, "artifacts", dkey)
    labels_path = os.path.join(bundle_dir, "labels.txt")
    preproc_path = os.path.join(bundle_dir, "preprocessor.json")
    model_path = None
    for cand in [os.path.join(bundle_dir, "model.ts"), os.path.join(bundle_dir, "model.pt")]:
        if os.path.exists(cand):
            model_path = cand
            break
    if model_path is None or (not os.path.exists(labels_path)) or (not os.path.exists(preproc_path)):
        raise FileNotFoundError("Missing model.ts/labels.txt/preprocessor.json for explanations.")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    with open(preproc_path, "r", encoding="utf-8") as f:
        preproc = json.load(f)

    # Deferred torch import
    import torch  # type: ignore
    device = "cpu"
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, labels, preproc


def _tokenize(text: str, lowercase: bool) -> List[str]:
    s = text or ""
    if lowercase:
        s = s.lower()
    return re.findall(r"[a-z0-9']+", s)


def _encode_texts(texts: List[str], preproc: Dict[str, Any]) -> np.ndarray:
    word_index = preproc.get("word_index") or preproc.get("vocab") or {}
    max_length = int(preproc.get("max_length", 256))
    lowercase = bool(preproc.get("lowercase", True))
    oov_token_id = int(preproc.get("oov_token_id", 1))

    ids_batch: List[List[int]] = []
    for t in texts:
        toks = _tokenize(t or "", lowercase)
        ids = [int(word_index.get(tok, oov_token_id)) for tok in toks]
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        ids_batch.append(ids)
    return np.asarray(ids_batch, dtype=np.int64)


def _make_predict_fn(domain_key: str):
    # Returns a function: List[str] -> np.ndarray [N, C] of probabilities
    model, labels, preproc = _load_bundle_for_explain(domain_key)

    def _predict(texts: List[str]) -> np.ndarray:
        import torch  # type: ignore
        input_ids = _encode_texts(texts, preproc)
        with torch.no_grad():
            tens = torch.as_tensor(input_ids, device="cpu")
            logits = model(tens)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            logits = logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs.astype(float)

    return _predict, labels, preproc

st.markdown("### Experiment results")
_metrics, _conf_mat, _lbls = load_artifact_results(domain)
if _metrics is not None:
    metrics_df = metrics_to_dataframe(_metrics)
if _conf_mat is not None and _lbls is not None and len(_conf_mat) > 0:
    try:
        conf_mat_df = confusion_matrix_to_dataframe(np.asarray(_conf_mat), _lbls)
    except Exception:
        conf_mat_df = None
    if metrics_df is not None:
        st.markdown("### Metrics")
        st.dataframe(metrics_df, use_container_width=True)

# Per-model comparison (shown even if metrics_df is missing)
model_metrics = load_per_model_metrics(domain)
if model_metrics:
    st.markdown("#### Per-model comparison")
    # Determine best model by preferred metric
    preferred_order = ["f1_weighted", "accuracy", "f1_macro"]
    available = {k for md in model_metrics.values() for k in md.keys()}
    metric_choice = next((m for m in preferred_order if m in available), None)
    if metric_choice is None:
        metric_choice = sorted(list(available))[0] if available else None

    best_model = None
    best_value = None
    if metric_choice:
        for mk, md in model_metrics.items():
            val = md.get(metric_choice)
            if isinstance(val, (int, float)):
                if best_value is None or val > best_value:
                    best_model, best_value = mk, val
        if best_model is not None:
            st.success(f"Best model by {metric_choice}: {best_model.upper()} ({best_value:.4f})")

    all_metric_names = sorted({m for md in model_metrics.values() for m in md.keys()})
    compare_df = pd.DataFrame({"metric": all_metric_names})
    for mk, md in model_metrics.items():
        compare_df[mk.upper()] = [md.get(name, np.nan) for name in all_metric_names]
    st.dataframe(compare_df, use_container_width=True)

# Per-model confusion matrices side-by-side
per_model_conf = load_per_model_confusions(domain)
if per_model_conf:
    st.markdown("#### Per-model confusion matrices")
    cols = st.columns(3)
    order = ["rnn", "lstm", "gru"]
    for c, mk in zip(cols, order):
        with c:
            if mk in per_model_conf:
                mat, lbls = per_model_conf[mk]
                # Shorten overly long labels to avoid squeezing the matrix
                lbls_short = [("mmo" if str(l).strip().lower() == "massively multiplayer" else l) for l in lbls]
                draw_confusion_matrix_heatmap(mat, lbls_short, width=500, dpi=200, title=mk.upper())
            else:
                st.caption(f"{mk.upper()}")
                st.write("—")

# Dataset distribution (Top 10)
st.markdown("### Top 10 Genre/Category Frequency")
nb_counts = compute_notebook_style_counts(domain)
if nb_counts is not None and len(nb_counts) > 0:
    # EXACT plotting behavior per notebook
    dkey = domain.strip().lower()
    counts_for_plot = nb_counts
    if dkey == "games":
        # Apply EXCLUDE_GENRES before selecting top 10 (as in Games notebook)
        EXCLUDE_GENRES = ["early access", "free to play", "indie", "casual"]
        counts_for_plot = Counter({g: c for g, c in nb_counts.items() if g not in EXCLUDE_GENRES})
    elif dkey == "books":
        # For Books: mimic Movies' mapping step within the same plot
        try:
            root = _project_root()
            csv_path = os.path.join(root, "Books", "BooksDatasetClean.csv")
            if os.path.exists(csv_path):
                df_books = pd.read_csv(csv_path)
                if {"Description", "Category"}.issubset(df_books.columns):
                    df_books = df_books.dropna(subset=["Description", "Category"])

                    def split_category_b(text: str) -> List[str]:
                        if not isinstance(text, str):
                            return []
                        parts = re.split(r"[,/|\s]+", text.lower())
                        parts = [p.strip() for p in parts if p.strip()]
                        return list(dict.fromkeys(parts))

                    category_list_b: List[List[str]] = []
                    for category_str in df_books["Category"]:
                        category_list_b.append(split_category_b(category_str))
                    category_list_b = [[c for c in sublist if c != "&"] for sublist in category_list_b]

                    EXCLUDE_CATEGORIES_B = {"fiction", "nonfiction", "general"}
                    flat_b = [g.strip() for sub in category_list_b for g in sub if g and str(g).strip()]
                    filtered_b = [g for g in flat_b if g.lower() not in EXCLUDE_CATEGORIES_B]
                    top10_b = [cat for cat, _ in Counter(filtered_b).most_common(10)]

                    # Apply mapping: keep only top-10; if none remain, map to 'other'
                    mapped_list: List[List[str]] = [[g for g in sub if g in top10_b] for sub in category_list_b]
                    for i in range(len(mapped_list)):
                        if len(mapped_list[i]) == 0:
                            mapped_list[i] = ["other"]
                    counts_for_plot = Counter([g for sub in mapped_list for g in sub])
        except Exception:
            pass
    # Movies: take most_common(10) directly
    top = counts_for_plot.most_common(10)
    genres, freq = zip(*top)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(genres, freq)
    ax.set_xticklabels(genres, rotation=45, ha="right", fontsize=9)
    ax.set_title("Top 10 Genre/Category Frequency")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=600)
    plt.close(fig)
else:
    st.info("No categories available for this domain using notebook-style preprocessing.")

# Samples
st.markdown("### Sample predictions vs actual")
model_keys = ["RNN", "LSTM", "GRU"]
found_any = False
for mk in model_keys:
    sdf = load_samples_table(domain, mk.lower())
    if sdf is not None and not sdf.empty:
        found_any = True
        st.markdown(f"#### {mk} samples")
        display_cols = [x for x in ["text", "true", "pred", "prob_pred", "correct"] if x in sdf.columns]
        if not display_cols:
            display_cols = list(sdf.columns)[:6]
        st.dataframe(sdf[display_cols].head(100), use_container_width=True, height=260)
        download_bytes_from_df(sdf, f"{domain.lower()}_{mk.lower()}_samples.csv", f"Download {mk} samples CSV")
    else:
        st.markdown(f"#### {mk} samples")
        st.write("—")
if not found_any:
    samples_df = load_samples_table(domain)
    if samples_df is not None and not samples_df.empty:
        display_cols = [c for c in ["text", "true", "pred", "prob_pred", "correct"] if c in samples_df.columns]
        if not display_cols:
            display_cols = list(samples_df.columns)[:6]
        st.dataframe(samples_df[display_cols].head(200), use_container_width=True, height=320)
        download_bytes_from_df(samples_df, f"{domain.lower()}_samples.csv", "Download samples CSV")
    else:
        st.info("No saved samples found yet. Once notebooks export samples.csv or samples_{model}.csv, they will appear here.")


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
    lime_enabled = st.checkbox("Run LIME explanation", value=True)
    LIME_FIXED_SAMPLES = 500

go = st.button("Predict Genre", type="primary")

if go:
    if not user_text or not user_text.strip():
        st.warning("Please enter a description.")
    else:
        # Trained weights path: requires standardized bundle artifacts/{domain}/
        try:
            logger.info("Starting prediction for domain=%s, text_len=%d", domain, len(user_text or ""))
            pred_label, model_labels, probabilities = predict_with_weights(domain, user_text)
            logger.info("Prediction complete: pred_label=%s, num_labels=%d", pred_label, len(model_labels))
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

            # Auto-run LIME on predicted class
            if lime_enabled:
                st.markdown("### LIME explanation (predicted class)")
                try:
                    logger.info("Entering LIME explanation block")
                    if LimeTextExplainer is None:
                        logger.warning("LIME not installed or failed to import")
                        st.info("LIME not installed. Please add 'lime' to your environment to enable explanations.")
                    else:
                        logger.debug("Constructing predict function for LIME")
                        predict_fn, labels, _ = _make_predict_fn(domain)
                        logger.debug("Predict function ready. labels_count=%d", len(labels))
                        target_idx = labels.index(pred_label) if pred_label in labels else int(np.argmax(probabilities))
                        logger.info("Target index for LIME: %d (%s); samples=%d", target_idx, labels[target_idx] if 0 <= target_idx < len(labels) else "out-of-range", LIME_FIXED_SAMPLES)
                        explainer = LimeTextExplainer(class_names=labels)
                        logger.debug("LimeTextExplainer created")
                        exp = explainer.explain_instance(
                            user_text or "",
                            predict_fn,
                            labels=[target_idx],
                            num_features=10,
                            num_samples=LIME_FIXED_SAMPLES,
                        )
                        logger.info("LIME explanation computed successfully")
                        weights = exp.as_list(label=target_idx)
                        logger.debug("LIME weights extracted: %d features", len(weights) if isinstance(weights, list) else -1)
                        lime_df = pd.DataFrame(weights, columns=["token/phrase", "weight"])
                        st.dataframe(lime_df, use_container_width=True, height=280)
                        try:
                            st.bar_chart(lime_df.set_index("token/phrase"))
                        except Exception:
                            logger.debug("Bar chart rendering for LIME weights failed; continuing")
                            pass
                except FileNotFoundError as e:
                    logger.warning("Artifacts missing for LIME: %s", e)
                    st.info(str(e))
                except Exception as e:
                    logger.exception("LIME explanation failed with an exception")
                    st.info(f"LIME explanation unavailable: {e}")
            else:
                st.info("LIME explanation disabled.")
        except FileNotFoundError as e:
            logger.warning("Prediction artifacts missing: %s", e)
            st.warning(str(e))
        except Exception as e:
            logger.exception("Prediction with trained weights failed")
            st.error(f"Prediction with trained weights failed: {e}")