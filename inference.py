from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import glob


BOOKS_DEFAULT_LABELS: List[str] = [
    "Fiction",
    "Non-Fiction",
    "Mystery",
    "Romance",
    "Fantasy",
    "Sci-Fi",
    "Thriller",
    "Young Adult",
    "Historical",
    "Horror",
]

MOVIES_DEFAULT_LABELS: List[str] = [
    "Drama",
    "Comedy",
    "Action",
    "Adventure",
    "Romance",
    "Thriller",
    "Horror",
    "Sci-Fi",
    "Fantasy",
    "Animation",
]

GAMES_DEFAULT_LABELS: List[str] = [
    "Action",
    "Adventure",
    "RPG",
    "Strategy",
    "Simulation",
    "Puzzle",
    "Sports",
    "Racing",
    "Indie",
    "Casual",
]


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _safe_unique_labels_from_column(series: pd.Series, max_labels: int = 50) -> List[str]:
    # Split multi-label strings on common separators, normalize, and dedupe
    labels: List[str] = []
    for val in series.dropna().astype(str).tolist():
        parts = re.split(r"[|,/;]+", val)
        for p in parts:
            label = p.strip()
            if label:
                labels.append(label)
    unique = sorted({l for l in labels})
    if not unique:
        return []
    return unique[:max_labels]


def _try_load_labels_from_csv(csv_path: str, candidate_columns: Sequence[str]) -> List[str]:
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    for col in candidate_columns:
        if col in df.columns:
            labels = _safe_unique_labels_from_column(df[col])
            if labels:
                return labels
    # Try case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidate_columns:
        if col.lower() in lower_map:
            labels = _safe_unique_labels_from_column(df[lower_map[col.lower()]])
            if labels:
                return labels
    return []


def load_domain_labels(domain: str, prefer_dataset: bool = True) -> List[str]:
    """
    Load genre labels for a domain. Attempts to read from datasets in the repo,
    otherwise falls back to a default list.
    """
    domain_key = domain.strip().lower()
    root = _project_root()

    if prefer_dataset:
        if domain_key == "books":
            csv_path = os.path.join(root, "Books", "BooksDatasetClean.csv")
            labels = _try_load_labels_from_csv(
                csv_path,
                candidate_columns=["Genre", "Genres", "genre", "genres", "Category", "Categories"],
            )
            if labels:
                return labels
        elif domain_key == "movies":
            csv_path = os.path.join(root, "Movies", "wiki_movie_plots_deduped.csv")
            labels = _try_load_labels_from_csv(
                csv_path, candidate_columns=["Genre", "Genres", "genre", "genres"]
            )
            if labels:
                return labels
        elif domain_key == "games":
            # Try both files that exist in repo
            csv_paths = [
                os.path.join(root, "Games", "steam.csv"),
                os.path.join(root, "Games", "steam_description_data.csv"),
            ]
            candidate_columns = ["genre", "genres", "Genre", "Genres"]
            for path in csv_paths:
                labels = _try_load_labels_from_csv(path, candidate_columns=candidate_columns)
                if labels:
                    return labels

    # Fallback defaults
    if domain_key == "books":
        return BOOKS_DEFAULT_LABELS
    if domain_key == "movies":
        return MOVIES_DEFAULT_LABELS
    if domain_key == "games":
        return GAMES_DEFAULT_LABELS
    return sorted({*BOOKS_DEFAULT_LABELS, *MOVIES_DEFAULT_LABELS, *GAMES_DEFAULT_LABELS})


def _deterministic_seed_from_text(text: str) -> int:
    # Stable, deterministic seed from text for reproducible placeholder predictions
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def placeholder_predict(text: str, labels: Sequence[str]) -> Tuple[str, np.ndarray]:
    """
    Placeholder genre prediction:
    - Starts with uniform scores
    - Boosts scores if label tokens appear in text
    - Adds tiny deterministic noise based on text
    Returns (predicted_label, probabilities ndarray)
    """
    clean_text = (text or "").lower()
    num_labels = len(labels)
    if num_labels == 0:
        raise ValueError("No labels provided to placeholder_predict.")
    scores = np.ones(num_labels, dtype=float)
    token_pattern = re.compile(r"[a-z0-9]+")
    text_tokens = set(token_pattern.findall(clean_text))

    for i, label in enumerate(labels):
        label_tokens = token_pattern.findall(str(label).lower())
        overlap = sum(1 for t in label_tokens if t in text_tokens)
        if overlap > 0:
            scores[i] += 2.0 * overlap

    # Deterministic jitter
    seed = _deterministic_seed_from_text(clean_text)
    rng = np.random.default_rng(seed)
    noise = rng.random(num_labels) * 0.01
    probs = (scores + noise).astype(float)
    probs = probs / probs.sum()
    pred_idx = int(np.argmax(probs))
    return str(labels[pred_idx]), probs


# ----------------------------
# Trained Weights Integration (Scaffold)
# ----------------------------
def get_domain_artifacts(domain: str) -> List[str]:
    """
    Discover candidate model artifacts for a domain.
    Returns list of artifact file paths (e.g., .pt/.pth).
    """
    root = _project_root()
    domain_key = domain.strip().lower()
    patterns: List[str] = []
    if domain_key == "books":
        patterns = [
            os.path.join(root, "Books", "*.pt"),
            os.path.join(root, "Books", "*.pth"),
        ]
    elif domain_key == "movies":
        patterns = [
            os.path.join(root, "Movies", "*.pt"),
            os.path.join(root, "Movies", "*.pth"),
        ]
    elif domain_key == "games":
        patterns = [
            os.path.join(root, "Games", "*.pt"),
            os.path.join(root, "Games", "*.pth"),
        ]
    artifacts: List[str] = []
    for p in patterns:
        artifacts.extend(glob.glob(p))
    # Deterministic order
    artifacts = sorted(artifacts)
    return artifacts


def predict_with_weights(domain: str, text: str) -> Tuple[str, List[str], np.ndarray]:
    """
    Attempt to run inference using trained weights for the domain.
    Current scaffold requires a standardized artifact bundle:
      - model.ts or model.pt (TorchScript or full scripted/traced module)
      - labels.txt (one label per line, in the same order as model outputs)
      - preprocessor.json (tokenizer/vocab and preprocessing config)

    If this standardized bundle is not present, raises a clear exception explaining
    what files are missing. This keeps the Streamlit UI resilient while guiding
    how to export artifacts from notebooks.
    """
    root = _project_root()
    domain_key = domain.strip().lower()
    bundle_dir = os.path.join(root, "artifacts", domain_key)
    model_candidates = [
        os.path.join(bundle_dir, "model.ts"),   # preferred TorchScript
        os.path.join(bundle_dir, "model.pt"),   # scripted module
    ]
    labels_path = os.path.join(bundle_dir, "labels.txt")
    preproc_path = os.path.join(bundle_dir, "preprocessor.json")

    missing: List[str] = []
    model_path: Optional[str] = None
    for cand in model_candidates:
        if os.path.exists(cand):
            model_path = cand
            break
    if model_path is None:
        missing.append("model.ts or model.pt")
    if not os.path.exists(labels_path):
        missing.append("labels.txt")
    if not os.path.exists(preproc_path):
        missing.append("preprocessor.json")

    if missing:
        discovered = get_domain_artifacts(domain)
        raise FileNotFoundError(
            "Trained weights bundle not found. Please export a standardized bundle under "
            f"`artifacts/{domain_key}/` with files: model.ts (or model.pt), labels.txt, preprocessor.json. "
            f"Missing: {', '.join(missing)}. "
            f"Discovered raw artifacts for {domain}: {discovered}"
        )

    # Deferred import to avoid forcing torch dependency unless needed
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("PyTorch is required to use trained weights. Please install `torch`.") from e

    # Load labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if len(labels) < 2:
        raise ValueError("labels.txt must contain at least two labels (one per line).")

    # Load preprocessor
    import json
    with open(preproc_path, "r", encoding="utf-8") as f:
        preproc = json.load(f)

    # Basic expected preprocessor fields (tokenizer/vocab)
    # This is intentionally flexible, but we expect either:
    # - a word_index mapping (token->id) and max_length
    # - or an instruction to use a built-in tokenizer (e.g., whitespace)
    word_index = preproc.get("word_index") or preproc.get("vocab")
    max_length = int(preproc.get("max_length", 256))
    lowercase = bool(preproc.get("lowercase", True))
    oov_token_id = int(preproc.get("oov_token_id", 1))

    if not isinstance(word_index, dict):
        raise ValueError("preprocessor.json must include a 'word_index' (token->id mapping).")

    # Tokenize text
    clean = text or ""
    if lowercase:
        clean = clean.lower()
    tokens = re.findall(r"[a-z0-9']+", clean)
    ids = [int(word_index.get(tok, oov_token_id)) for tok in tokens]
    if len(ids) < max_length:
        ids = ids + [0] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
    input_ids = np.asarray([ids], dtype=np.int64)

    # Load TorchScript model and run
    device = "cpu"
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    with torch.no_grad():
        inp = torch.as_tensor(input_ids, device=device)
        logits = model(inp)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    return pred_label, labels, probs


