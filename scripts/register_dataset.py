#!/usr/bin/env python3
"""Register exact WDBC dataset bytes and create a non-result-bearing dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from bcimbalance.dataset import register_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the candidate WDBC CSV")
    parser.add_argument(
        "--registry",
        default="data/registry/wdbc_dataset.json",
        help="Destination JSON registry manifest",
    )
    parser.add_argument(
        "--copy-to",
        default="data/raw/wdbc.csv",
        help="Local canonical copy path; raw data are gitignored",
    )
    parser.add_argument("--expected-sha256", default=None)
    args = parser.parse_args()

    manifest = register_dataset(
        args.source,
        args.registry,
        expected_sha256=args.expected_sha256,
        canonical_copy=args.copy_to,
    )
    print(f"DATASET_REGISTERED_SHA256={manifest['sha256']}")
    print(f"DATASET_ROWS={manifest['n_rows']}")
    print(f"REGISTRY={Path(args.registry)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
