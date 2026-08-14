"""Strict loading of the canonical WDBC CSV."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_SHA256

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_wdbc(path: str | Path, require_canonical_hash: bool = True) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if require_canonical_hash and actual_hash != DATASET_SHA256:
        raise ValueError(f"dataset SHA-256 mismatch: expected {DATASET_SHA256}, got {actual_hash}")
    frame = pd.read_csv(path)
    empty_exports = [c for c in frame.columns if str(c).startswith("Unnamed:") and frame[c].isna().all()]
    frame = frame.drop(columns=empty_exports)
    if "diagnosis" not in frame or "id" not in frame:
        raise ValueError("dataset must contain id and diagnosis columns")
    feature_columns = [c for c in frame.columns if c not in {"id", "diagnosis"}]
    if len(frame) != 569 or len(feature_columns) != 30:
        raise ValueError(f"expected 569 rows and 30 predictors; got {len(frame)} and {len(feature_columns)}")
    labels = frame["diagnosis"].astype(str).str.strip().str.upper()
    if labels.value_counts().to_dict() != {"B": 357, "M": 212}:
        raise ValueError("expected exactly 357 benign and 212 malignant observations")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("row ids must be present and unique")
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("predictors must be finite numeric values")
    clean = pd.concat([frame[["id"]].reset_index(drop=True), labels.rename("diagnosis"), numeric.reset_index(drop=True)], axis=1)
    return clean, feature_columns
