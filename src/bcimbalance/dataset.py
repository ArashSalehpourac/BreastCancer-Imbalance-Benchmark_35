"""Dataset registration, hashing, and fail-closed WDBC schema validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_SCHEMA_VERSION = "wdbc-csv/v1"
TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"
OPTIONAL_EMPTY_COLUMNS = ("Unnamed: 32",)
ALLOWED_LABELS = frozenset({"B", "M"})

FEATURE_COLUMNS = (
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
)

CANONICAL_COLUMNS = (ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS)


class DatasetValidationError(ValueError):
    """Raised when the candidate dataset violates the frozen protocol gates."""


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(encoded)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def validate_wdbc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize a WDBC CSV dataframe without mutating the input."""
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DatasetValidationError(f"duplicate column names: {duplicates}")

    df = frame.copy(deep=True)

    for optional in OPTIONAL_EMPTY_COLUMNS:
        if optional in df.columns:
            if not df[optional].isna().all():
                raise DatasetValidationError(
                    f"optional export column {optional!r} is present but is not entirely empty"
                )
            df = df.drop(columns=[optional])

    actual = set(df.columns)
    expected = set(CANONICAL_COLUMNS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise DatasetValidationError(f"schema mismatch; missing={missing}, extra={extra}")

    df = df.loc[:, list(CANONICAL_COLUMNS)]

    if df[ID_COLUMN].isna().any():
        raise DatasetValidationError("row identifier contains missing values")
    if df[ID_COLUMN].duplicated().any():
        duplicate_ids = df.loc[df[ID_COLUMN].duplicated(keep=False), ID_COLUMN].astype(str).tolist()
        raise DatasetValidationError(f"row identifier is not unique: {duplicate_ids[:10]}")

    labels = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    observed = set(labels.unique())
    if observed != ALLOWED_LABELS:
        raise DatasetValidationError(
            f"target labels must be exactly {sorted(ALLOWED_LABELS)} after normalization; observed={sorted(observed)}"
        )
    df[TARGET_COLUMN] = labels

    numeric = df.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise DatasetValidationError(f"predictor columns contain missing/non-numeric values: {bad}")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_positions = np.argwhere(~np.isfinite(values))
        first = bad_positions[0]
        raise DatasetValidationError(
            f"predictor matrix contains non-finite value at row={int(first[0])}, column={FEATURE_COLUMNS[int(first[1])]!r}"
        )
    df.loc[:, list(FEATURE_COLUMNS)] = numeric

    if len(df) < 10:
        raise DatasetValidationError("dataset contains too few rows for the frozen 10-fold protocol")

    class_counts = df[TARGET_COLUMN].value_counts()
    if int(class_counts.min()) < 10:
        raise DatasetValidationError("each diagnosis class must contain at least 10 rows for 10-fold stratification")

    return df


def load_registered_dataset(csv_path: str | os.PathLike[str], expected_sha256: str) -> pd.DataFrame:
    actual = sha256_file(csv_path)
    if actual.lower() != expected_sha256.lower():
        raise DatasetValidationError(
            f"dataset SHA-256 mismatch: expected={expected_sha256.lower()} actual={actual.lower()}"
        )
    frame = pd.read_csv(csv_path)
    return validate_wdbc_frame(frame)


def register_dataset(
    source: str | os.PathLike[str],
    registry_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    canonical_copy: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Register exact source bytes after schema validation; optionally copy them locally."""
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    digest = sha256_file(source_path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise DatasetValidationError(
            f"dataset SHA-256 mismatch: expected={expected_sha256.lower()} actual={digest.lower()}"
        )

    validated = validate_wdbc_frame(pd.read_csv(source_path))

    copied_to = None
    if canonical_copy is not None:
        destination = Path(canonical_copy)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        copied_digest = sha256_file(destination)
        if copied_digest != digest:
            raise DatasetValidationError("copied dataset bytes do not match registered source hash")
        copied_to = str(destination)

    class_counts = validated[TARGET_COLUMN].value_counts().sort_index().to_dict()
    manifest: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "source_filename": source_path.name,
        "source_bytes": source_path.stat().st_size,
        "sha256": digest,
        "n_rows": int(len(validated)),
        "n_features": len(FEATURE_COLUMNS),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "canonical_columns": list(CANONICAL_COLUMNS),
        "canonical_copy": copied_to,
    }
    _atomic_json_write(Path(registry_path), manifest)
    return manifest
