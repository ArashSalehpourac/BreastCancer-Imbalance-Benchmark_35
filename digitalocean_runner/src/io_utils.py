"""Atomic artifacts and validation for restart-safe execution."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

def ensure_output(root: Path) -> None:
    for name in ("tasks", "training_cache", "resampling_audits", "errors"):
        (root / name).mkdir(parents=True, exist_ok=True)

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent); os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def valid_task(path: Path, expected_predictions: int | None = None) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {"identity", "metrics", "predictions", "dataset_sha256"}
        return required <= value.keys() and (expected_predictions is None or len(value["predictions"]) == expected_predictions)
    except (OSError, ValueError, TypeError):
        return False
def valid_cache(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {"X", "y"} <= set(data.files) and len(data["X"]) == len(data["y"]) and data["X"].ndim == 2
    except (OSError, ValueError):
        return False
