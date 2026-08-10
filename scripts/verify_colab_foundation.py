#!/usr/bin/env python3
"""Verify the frozen P1 foundation in a Colab/Drive runtime only."""

from __future__ import annotations

import argparse

from bcimbalance.colab_gate import verify_colab_foundation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Exact WDBC CSV path, typically on mounted Google Drive")
    parser.add_argument(
        "--lock",
        default="data/registry/FOUNDATION_LOCK_v1.json",
        help="Frozen P1 foundation lock",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Drive directory for non-result-bearing P1C verification evidence",
    )
    parser.add_argument("--git-commit", required=True, help="Checked-out repository HEAD")
    args = parser.parse_args()

    report = verify_colab_foundation(
        dataset_path=args.dataset,
        lock_path=args.lock,
        output_dir=args.output_dir,
        git_commit=args.git_commit,
    )
    print(f"DATASET_SHA256={report['dataset_sha256']}")
    print(f"SPLIT_SHA256={report['split_sha256']}")
    print(f"SEED_REGISTRY_SHA256={report['seed_registry_sha256']}")
    print(f"OUTER_FOLDS={report['outer_folds']}")
    print(f"SEED_RECORDS={report['seed_records']}")
    print("RESULT_BEARING=false")
    print("COLAB_FOUNDATION_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
