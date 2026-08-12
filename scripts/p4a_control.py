#!/usr/bin/env python3
"""Run P4A control/preflight only; never execute a scientific condition."""

from __future__ import annotations

import argparse

from bcimbalance.p4_control import run_p4a_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate P4A execution control and write non-result-bearing evidence."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument(
        "--policy",
        default="config/P4A_EXECUTION_CONTROL_POLICY_v1.json",
    )
    parser.add_argument(
        "--config",
        default="config/EXPERIMENT_CONFIG_v1.json",
    )
    parser.add_argument(
        "--foundation-lock",
        default="data/registry/FOUNDATION_LOCK_v1.json",
    )
    parser.add_argument(
        "--result-schema",
        default="config/RESULT_EVIDENCE_SCHEMA_v1.json",
    )
    parser.add_argument(
        "--authorization-ref",
        required=True,
        help="Auditable P4A authorization reference; does not authorize P4B.",
    )
    parser.add_argument(
        "--require-exact-colab-paths",
        action="store_true",
        help="Require the frozen Drive dataset and P4A control-root path policy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_report, manifest, plan = run_p4a_control(
        dataset_path=args.dataset,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        control_root=args.control_root,
        policy_path=args.policy,
        config_path=args.config,
        foundation_lock_path=args.foundation_lock,
        result_schema_path=args.result_schema,
        authorization_ref=args.authorization_ref,
        require_exact_colab_paths=args.require_exact_colab_paths,
    )

    print(f"P4A_GIT_COMMIT={control_report['git_commit']}")
    print(f"P4A_POLICY_SHA256={control_report['policy_sha256']}")
    print(f"P4A_CONTROL_SESSION={control_report['control_session_id']}")
    print(f"P4A_PLANNED_TASKS={plan['execution_task_count']}")
    print(f"P4A_PENDING_TASKS={manifest['checkpoint_summary']['PENDING']}")
    print("P3_PREFLIGHT_GATE=PASS")
    print("P3_DRY_RUN_GATE=PASS")
    print("P4A_CONTROL_GATE=PASS")
    print("P4B_AUTHORIZED=false")
    print("RESULT_BEARING=false")
    print("SCIENTIFIC_EXECUTION=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
