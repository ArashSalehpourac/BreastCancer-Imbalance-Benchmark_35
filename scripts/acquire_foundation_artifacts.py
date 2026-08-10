#!/usr/bin/env python3
"""Acquire the pinned historical WDBC snapshot and emit P1 metadata artifacts only.

This script never imports or executes a classifier, resampler, or CTGAN. It is
strictly a dataset/split/seed/provenance operation under Protocol v1.0.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from bcimbalance.dataset import load_registered_dataset, register_dataset
from bcimbalance.evidence import build_foundation_manifest, write_foundation_manifest, write_json
from bcimbalance.folds import build_outer_folds, write_split_artifact
from bcimbalance.seeds import generate_seed_registry, write_seed_registry

SOURCE_REPOSITORY = "rzaroz/BreastCancer"
SOURCE_REVISION = "df2a0919eacd8e98e9242c8f4002a231e6eb57eb"
SOURCE_PATH = "MainDatasets/BreastCancerWisconsin(Diagnostic).csv"
DEFAULT_LOCK_FILE = "data/registry/FOUNDATION_LOCK_v1.json"


def pinned_source_url() -> str:
    encoded_path = urllib.parse.quote(SOURCE_PATH, safe="/")
    return f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_REVISION}/{encoded_path}"


def _download_exact_bytes(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BreastCancer-Imbalance-Benchmark_35/P1-foundation"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310: pinned HTTPS host/revision
        payload = response.read()
    if not payload:
        raise RuntimeError("pinned dataset download returned zero bytes")
    destination.write_bytes(payload)


def _load_lock(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("schema_version") != "foundation-lock/v1":
        raise RuntimeError("foundation lock schema mismatch")
    if lock.get("protocol_version") != "1.0" or bool(lock.get("result_bearing")):
        raise RuntimeError("foundation lock does not match frozen non-result-bearing protocol")
    dataset = lock.get("dataset", {})
    if dataset.get("source_repository") != SOURCE_REPOSITORY:
        raise RuntimeError("foundation lock source repository mismatch")
    if dataset.get("source_revision") != SOURCE_REVISION:
        raise RuntimeError("foundation lock source revision mismatch")
    if dataset.get("source_path") != SOURCE_PATH:
        raise RuntimeError("foundation lock source path mismatch")
    return lock


def _assert_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(f"foundation lock mismatch for {name}: expected={expected!r} actual={actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/p1")
    parser.add_argument("--raw-path", default="data/raw/wdbc.csv")
    parser.add_argument("--lock-file", default=DEFAULT_LOCK_FILE)
    parser.add_argument("--git-commit", default=os.environ.get("FOUNDATION_GIT_COMMIT", "unknown"))
    parser.add_argument("--run-id", default="p1-foundation-acquisition")
    args = parser.parse_args()

    lock = _load_lock(Path(args.lock_file))
    dataset_lock = lock["dataset"]
    folds_lock = lock["outer_folds"]
    seed_lock = lock["seed_registry"]

    output_dir = Path(args.output_dir)
    raw_path = Path(args.raw_path)
    source_url = pinned_source_url()
    _download_exact_bytes(source_url, raw_path)

    dataset_registry = output_dir / "wdbc_dataset.json"
    dataset_manifest = register_dataset(
        raw_path,
        dataset_registry,
        expected_sha256=str(dataset_lock["sha256"]),
        canonical_copy=None,
        source_uri=source_url,
        source_revision=SOURCE_REVISION,
    )
    _assert_equal("dataset bytes", dataset_manifest["source_bytes"], int(dataset_lock["bytes"]))
    _assert_equal("dataset rows", dataset_manifest["n_rows"], int(dataset_lock["rows"]))
    _assert_equal("dataset features", dataset_manifest["n_features"], int(dataset_lock["features"]))
    _assert_equal("dataset class counts", dataset_manifest["class_counts"], dataset_lock["class_counts"])

    frame = load_registered_dataset(raw_path, dataset_manifest["sha256"])
    split_artifact = build_outer_folds(frame, dataset_sha256=dataset_manifest["sha256"])
    split_path = output_dir / "outer_folds_v1.json"
    split_sha256 = write_split_artifact(split_path, split_artifact)
    _assert_equal("split SHA-256", split_sha256, str(folds_lock["sha256"]))
    _assert_equal("outer fold count", len(split_artifact["folds"]), int(folds_lock["count"]))

    seed_registry = generate_seed_registry()
    seed_path = output_dir / "seed_registry_v1.json"
    seed_sha256 = write_seed_registry(seed_path, seed_registry)
    _assert_equal("seed registry SHA-256", seed_sha256, str(seed_lock["sha256"]))
    _assert_equal("seed record count", seed_registry["n_records"], int(seed_lock["records"]))

    manifest = build_foundation_manifest(
        run_id=args.run_id,
        git_commit=args.git_commit,
        dataset_sha256=dataset_manifest["sha256"],
        split_sha256=split_sha256,
        seed_registry_sha256=seed_sha256,
    )
    manifest["foundation_lock"] = str(Path(args.lock_file))
    manifest["dataset_source_repository"] = SOURCE_REPOSITORY
    manifest["dataset_source_revision"] = SOURCE_REVISION
    manifest["dataset_source_path"] = SOURCE_PATH
    manifest["dataset_source_uri"] = source_url
    write_foundation_manifest(output_dir / "foundation_manifest.json", manifest)

    summary = {
        "protocol_version": manifest["protocol_version"],
        "result_bearing": False,
        "foundation_lock": str(Path(args.lock_file)),
        "dataset_sha256": dataset_manifest["sha256"],
        "dataset_bytes": dataset_manifest["source_bytes"],
        "dataset_rows": dataset_manifest["n_rows"],
        "dataset_features": dataset_manifest["n_features"],
        "dataset_class_counts": dataset_manifest["class_counts"],
        "dataset_source_revision": SOURCE_REVISION,
        "split_sha256": split_sha256,
        "outer_folds": len(split_artifact["folds"]),
        "seed_registry_sha256": seed_sha256,
        "seed_records": seed_registry["n_records"],
        "git_commit": args.git_commit,
    }
    write_json(output_dir / "foundation_summary.json", summary)

    print(f"DATASET_SHA256={summary['dataset_sha256']}")
    print(f"DATASET_BYTES={summary['dataset_bytes']}")
    print(f"DATASET_ROWS={summary['dataset_rows']}")
    print(f"DATASET_CLASS_COUNTS={json.dumps(summary['dataset_class_counts'], sort_keys=True)}")
    print(f"SPLIT_SHA256={summary['split_sha256']}")
    print(f"OUTER_FOLDS={summary['outer_folds']}")
    print(f"SEED_REGISTRY_SHA256={summary['seed_registry_sha256']}")
    print(f"SEED_RECORDS={summary['seed_records']}")
    print("FOUNDATION_LOCK=PASS")
    print("RESULT_BEARING=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
