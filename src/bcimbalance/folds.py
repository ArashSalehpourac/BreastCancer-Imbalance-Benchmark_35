"""Immutable repeated-stratified outer-fold artifact generation and verification."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .dataset import ID_COLUMN, TARGET_COLUMN
from .protocol import MASTER_SEED, N_REPEATS, N_SPLITS, PROTOCOL_VERSION

SPLIT_SCHEMA_VERSION = "outer-folds/v1"


class SplitValidationError(ValueError):
    """Raised when a split artifact violates the frozen outer-CV invariants."""


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _payload_hash(payload_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload_without_hash)).hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(encoded)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def build_outer_folds(
    frame: pd.DataFrame,
    *,
    dataset_sha256: str,
    master_seed: int = MASTER_SEED,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
) -> dict[str, Any]:
    """Build the frozen shared 5×10 outer-fold artifact from canonical row order."""
    ids = frame[ID_COLUMN].astype(str).tolist()
    labels = frame[TARGET_COLUMN].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise SplitValidationError("row identifiers must be unique before fold generation")

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=master_seed,
    )

    folds: list[dict[str, Any]] = []
    for split_index, (train_idx, test_idx) in enumerate(splitter.split(ids, labels)):
        repeat_index = split_index // n_splits
        fold_index = split_index % n_splits
        train_ids = [ids[int(i)] for i in train_idx]
        test_ids = [ids[int(i)] for i in test_idx]
        train_labels = [labels[int(i)] for i in train_idx]
        test_labels = [labels[int(i)] for i in test_idx]
        folds.append(
            {
                "split_index": split_index,
                "repeat_index": repeat_index,
                "fold_index": fold_index,
                "train_ids": train_ids,
                "test_ids": test_ids,
                "train_class_counts": dict(sorted(Counter(train_labels).items())),
                "test_class_counts": dict(sorted(Counter(test_labels).items())),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_sha256": dataset_sha256.lower(),
        "master_seed": int(master_seed),
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "row_count": len(ids),
        "row_ids": ids,
        "folds": folds,
    }
    verify_split_artifact(payload, require_hash=False)
    split_hash = _payload_hash(payload)
    payload["split_sha256"] = split_hash
    verify_split_artifact(payload, require_hash=True)
    return payload


def verify_split_artifact(artifact: dict[str, Any], *, require_hash: bool = True) -> None:
    row_ids = [str(v) for v in artifact.get("row_ids", [])]
    if not row_ids or len(row_ids) != len(set(row_ids)):
        raise SplitValidationError("split artifact row_ids must be non-empty and unique")

    n_splits = int(artifact.get("n_splits", -1))
    n_repeats = int(artifact.get("n_repeats", -1))
    folds = artifact.get("folds", [])
    if n_splits != N_SPLITS or n_repeats != N_REPEATS:
        raise SplitValidationError(
            f"artifact must implement frozen {N_REPEATS}×{N_SPLITS} design; got {n_repeats}×{n_splits}"
        )
    if len(folds) != n_splits * n_repeats:
        raise SplitValidationError("unexpected number of outer folds")

    all_ids = set(row_ids)
    test_counts_by_repeat: dict[int, Counter[str]] = defaultdict(Counter)
    seen_keys: set[tuple[int, int]] = set()

    for fold in folds:
        repeat = int(fold["repeat_index"])
        fold_index = int(fold["fold_index"])
        key = (repeat, fold_index)
        if key in seen_keys:
            raise SplitValidationError(f"duplicate repeat/fold key: {key}")
        seen_keys.add(key)
        if not (0 <= repeat < n_repeats and 0 <= fold_index < n_splits):
            raise SplitValidationError(f"out-of-range repeat/fold key: {key}")

        train = {str(v) for v in fold["train_ids"]}
        test = {str(v) for v in fold["test_ids"]}
        if train & test:
            raise SplitValidationError(f"train/test overlap in repeat={repeat} fold={fold_index}")
        if train | test != all_ids:
            raise SplitValidationError(f"train/test union does not cover all rows in repeat={repeat} fold={fold_index}")
        if len(train) != len(fold["train_ids"]) or len(test) != len(fold["test_ids"]):
            raise SplitValidationError(f"duplicate IDs inside fold repeat={repeat} fold={fold_index}")
        test_counts_by_repeat[repeat].update(test)

    if seen_keys != {(r, f) for r in range(n_repeats) for f in range(n_splits)}:
        raise SplitValidationError("repeat/fold grid is incomplete")

    for repeat in range(n_repeats):
        counts = test_counts_by_repeat[repeat]
        if set(counts) != all_ids or any(counts[row_id] != 1 for row_id in all_ids):
            raise SplitValidationError(f"each row must appear exactly once as test in repeat {repeat}")

    if require_hash:
        stored_hash = str(artifact.get("split_sha256", "")).lower()
        if len(stored_hash) != 64:
            raise SplitValidationError("missing/invalid split_sha256")
        copy = dict(artifact)
        copy.pop("split_sha256", None)
        computed = _payload_hash(copy)
        if computed != stored_hash:
            raise SplitValidationError(
                f"split artifact hash mismatch: stored={stored_hash} computed={computed}"
            )


def write_split_artifact(path: str | os.PathLike[str], artifact: dict[str, Any]) -> str:
    verify_split_artifact(artifact, require_hash=True)
    destination = Path(path)
    _atomic_json_write(destination, artifact)
    return str(artifact["split_sha256"])


def load_split_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    verify_split_artifact(artifact, require_hash=True)
    return artifact
