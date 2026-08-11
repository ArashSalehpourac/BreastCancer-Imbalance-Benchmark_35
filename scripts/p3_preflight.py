#!/usr/bin/env python3
"""Execute P3 preflight/dry-run only; never execute scientific conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

from bcimbalance.execution_harness import (
    run_preflight,
    write_json_atomic,
    write_sha256_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen execution identities and write a non-result-bearing dry-run plan."
    )
    parser.add_argument("--dataset", required=True, help="Path to canonical LF WDBC CSV")
    parser.add_argument(
        "--config",
        default="config/EXPERIMENT_CONFIG_v1.json",
        help="Frozen P2 experiment configuration",
    )
    parser.add_argument(
        "--foundation-lock",
        default="data/registry/FOUNDATION_LOCK_v1.json",
        help="Frozen P1 foundation lock",
    )
    parser.add_argument(
        "--result-schema",
        default="config/RESULT_EVIDENCE_SCHEMA_v1.json",
        help="Frozen P3 result-evidence schema",
    )
    parser.add_argument("--repo-root", default=".", help="Git repository root")
    parser.add_argument(
        "--expected-git-commit",
        required=True,
        help="Separately authorized exact Git commit; HEAD must equal it",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for non-result-bearing preflight evidence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "pip_freeze.txt"

    report, plan = run_preflight(
        dataset_path=args.dataset,
        config_path=args.config,
        foundation_lock_path=args.foundation_lock,
        result_schema_path=args.result_schema,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        pip_freeze_path=freeze_path,
    )

    report_path = output_dir / "p3_preflight_report.json"
    report_sha = write_json_atomic(report_path, report)
    write_sha256_receipt(report_path, report_sha)

    plan_path = output_dir / "p3_execution_plan.json"
    plan_sha = write_json_atomic(plan_path, plan)
    write_sha256_receipt(plan_path, plan_sha)

    print(f"P3_GIT_COMMIT={report['git_commit']}")
    print(f"P3_CONFIG_SHA256={report['config_sha256']}")
    print(f"P3_FOUNDATION_SHA256={report['foundation_sha256']}")
    print(f"P3_RESULT_SCHEMA_SHA256={report['result_schema_sha256']}")
    print(f"P3_SPLIT_SHA256={report['split_sha256']}")
    print(f"P3_SEED_REGISTRY_SHA256={report['seed_registry_sha256']}")
    print(f"P3_PIP_FREEZE_SHA256={report['pip_freeze_sha256']}")
    print(f"P3_PRIMARY_CONDITIONS={report['primary_conditions']}")
    print(f"P3_PLANNED_EXECUTION_TASKS={report['planned_execution_tasks']}")
    print("P3_PREFLIGHT_GATE=PASS")
    print("P3_DRY_RUN_GATE=PASS")
    print("RESULT_BEARING=false")
    print("SCIENTIFIC_EXECUTION=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
