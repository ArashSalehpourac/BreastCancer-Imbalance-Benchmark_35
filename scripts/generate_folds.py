#!/usr/bin/env python3
"""Generate the immutable shared 5×10 outer-fold artifact after dataset registration."""

from __future__ import annotations

import argparse
import json

from bcimbalance.dataset import load_registered_dataset
from bcimbalance.folds import build_outer_folds, write_split_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/raw/wdbc.csv")
    parser.add_argument("--registry", default="data/registry/wdbc_dataset.json")
    parser.add_argument("--output", default="data/splits/outer_folds_v1.json")
    args = parser.parse_args()

    with open(args.registry, "r", encoding="utf-8") as handle:
        registry = json.load(handle)
    dataset_sha256 = str(registry["sha256"])
    frame = load_registered_dataset(args.dataset, dataset_sha256)
    artifact = build_outer_folds(frame, dataset_sha256=dataset_sha256)
    split_hash = write_split_artifact(args.output, artifact)
    print(f"SPLIT_SHA256={split_hash}")
    print(f"OUTER_FOLDS={len(artifact['folds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
