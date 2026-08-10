#!/usr/bin/env python3
"""Generate the deterministic frozen condition-seed registry; no models are executed."""

from __future__ import annotations

import argparse

from bcimbalance.seeds import generate_seed_registry, write_seed_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/registry/seed_registry_v1.json")
    args = parser.parse_args()

    registry = generate_seed_registry()
    digest = write_seed_registry(args.output, registry)
    print(f"SEED_REGISTRY_SHA256={digest}")
    print(f"SEED_RECORDS={registry['n_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
