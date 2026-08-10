"""Google Colab foundation verification with no scientific model execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .dataset import FEATURE_COLUMNS, TARGET_COLUMN, load_registered_dataset, sha256_file
from .evidence import build_foundation_manifest, write_foundation_manifest, write_json
from .folds import build_outer_folds
from .protocol import MASTER_SEED, PROTOCOL_VERSION
from .seeds import generate_seed_registry

FOUNDATION_BASE_COMMIT = "2e6c405c6dce747fc86e3d72252f7a78831f1771"
COLAB_GATE_SCHEMA_VERSION = "colab-foundation-gate/v1"


class ColabFoundationGateError(ValueError):
    """Raised when a Colab runtime does not match the frozen P1 foundation."""


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ColabFoundationGateError(f"expected JSON object in {path}")
    return payload


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ColabFoundationGateError(f"{name} mismatch: expected={expected!r} actual={actual!r}")


def verify_colab_foundation(
    *,
    dataset_path: str | os.PathLike[str],
    lock_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    git_commit: str,
) -> dict[str, Any]:
    """Verify dataset/split/seed identity and write non-result-bearing evidence.

    This function performs only foundation checks. It does not import, fit, sample,
    train, predict, score, or compare any scientific classifier or imbalance method.
    """
    dataset = Path(dataset_path)
    lock_file = Path(lock_path)
    output = Path(output_dir)

    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not lock_file.is_file():
        raise FileNotFoundError(lock_file)
    if len(git_commit.strip()) < 7:
        raise ColabFoundationGateError("git_commit must identify the checked-out repository revision")

    lock = _load_json(lock_file)
    _require_equal("foundation lock schema", lock.get("schema_version"), "foundation-lock/v1")
    _require_equal("protocol version", str(lock.get("protocol_version")), PROTOCOL_VERSION)
    _require_equal("foundation lock result_bearing", bool(lock.get("result_bearing")), False)

    dataset_lock = lock.get("dataset", {})
    expected_dataset_sha = str(dataset_lock.get("sha256", "")).lower()
    if len(expected_dataset_sha) != 64:
        raise ColabFoundationGateError("foundation lock contains an invalid dataset SHA-256")

    actual_dataset_sha = sha256_file(dataset)
    _require_equal("dataset SHA-256", actual_dataset_sha, expected_dataset_sha)
    _require_equal("dataset bytes", dataset.stat().st_size, int(dataset_lock["bytes"]))

    frame = load_registered_dataset(dataset, expected_dataset_sha)
    class_counts = {
        str(label): int(count)
        for label, count in frame[TARGET_COLUMN].value_counts().sort_index().to_dict().items()
    }
    expected_class_counts = {
        str(label): int(count) for label, count in dict(dataset_lock["class_counts"]).items()
    }
    _require_equal("dataset rows", len(frame), int(dataset_lock["rows"]))
    _require_equal("dataset features", len(FEATURE_COLUMNS), int(dataset_lock["features"]))
    _require_equal("dataset class counts", class_counts, expected_class_counts)

    split_artifact = build_outer_folds(frame, dataset_sha256=actual_dataset_sha)
    split_lock = lock.get("outer_folds", {})
    _require_equal("outer fold count", len(split_artifact["folds"]), int(split_lock["count"]))
    _require_equal("outer fold SHA-256", split_artifact["split_sha256"], str(split_lock["sha256"]).lower())

    seed_registry = generate_seed_registry()
    seed_lock = lock.get("seed_registry", {})
    _require_equal("master seed", MASTER_SEED, int(seed_lock["master_seed"]))
    _require_equal("seed record count", seed_registry["n_records"], int(seed_lock["records"]))
    _require_equal(
        "seed registry SHA-256",
        seed_registry["registry_sha256"],
        str(seed_lock["sha256"]).lower(),
    )

    output.mkdir(parents=True, exist_ok=True)
    lock_sha256 = sha256_file(lock_file)
    report: dict[str, Any] = {
        "schema_version": COLAB_GATE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "foundation_base_commit": FOUNDATION_BASE_COMMIT,
        "git_commit": git_commit.strip(),
        "foundation_lock_sha256": lock_sha256,
        "dataset_path": str(dataset),
        "dataset_sha256": actual_dataset_sha,
        "dataset_bytes": dataset.stat().st_size,
        "dataset_rows": len(frame),
        "dataset_features": len(FEATURE_COLUMNS),
        "dataset_class_counts": class_counts,
        "split_sha256": split_artifact["split_sha256"],
        "outer_folds": len(split_artifact["folds"]),
        "seed_registry_sha256": seed_registry["registry_sha256"],
        "seed_records": seed_registry["n_records"],
        "master_seed": MASTER_SEED,
        "gate": "PASS",
        "result_bearing": False,
    }

    report_receipt = write_json(output / "colab_foundation_gate.json", report)
    manifest = build_foundation_manifest(
        run_id="p1c-colab-foundation-gate",
        git_commit=git_commit.strip(),
        dataset_sha256=actual_dataset_sha,
        split_sha256=split_artifact["split_sha256"],
        seed_registry_sha256=seed_registry["registry_sha256"],
        phase="P1-foundation",
    )
    manifest["subphase"] = "P1C-colab-foundation"
    manifest["foundation_base_commit"] = FOUNDATION_BASE_COMMIT
    manifest["foundation_lock_sha256"] = lock_sha256
    manifest["colab_gate_report_sha256"] = report_receipt.sha256
    write_foundation_manifest(output / "colab_foundation_manifest.json", manifest)
    return report
