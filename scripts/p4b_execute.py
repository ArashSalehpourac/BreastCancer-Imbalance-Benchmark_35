#!/usr/bin/env python3
"""P4B validation-only gate and future one-task execution entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bcimbalance.execution_harness import load_foundation_lock, load_result_schema, verify_exact_git_commit
from bcimbalance.experiment_config import load_experiment_config
from bcimbalance.p4b_authorization import (
    AUTHORIZED_P4B_BASE,
    EXPECTED_P4A_PIP_FREEZE_SHA256,
    P4BAuthorizationError,
    load_policy,
)
from bcimbalance.scientific_task import run_authorized_task


def validate_only(args: argparse.Namespace) -> int:
    actual_commit = verify_exact_git_commit(args.repo_root, args.expected_git_commit)
    if actual_commit != AUTHORIZED_P4B_BASE and not args.allow_review_head:
        raise P4BAuthorizationError(
            "validation requires authoritative P4B base unless --allow-review-head is explicitly set"
        )

    policy, policy_sha256 = load_policy(args.policy)
    config = load_experiment_config(args.config)
    load_foundation_lock(args.foundation_lock)
    result_schema = load_result_schema(args.result_schema)

    schema_path = Path(args.authorization_schema)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("properties", {}).get("schema_version", {}).get("const") != "p4b-authorization/v1":
        raise P4BAuthorizationError("P4B authorization JSON schema identifier mismatch")
    if schema.get("properties", {}).get("result_bearing_authorized", {}).get("const") is not True:
        raise P4BAuthorizationError("authorization schema must require explicit true authorization")
    if schema.get("properties", {}).get("max_tasks", {}).get("const") != 1:
        raise P4BAuthorizationError("authorization schema must freeze max_tasks=1")

    print("P4B_PACKAGE_VALIDATION=PASS")
    print(f"P4B_REVIEW_GIT_COMMIT={actual_commit}")
    print(f"P4B_POLICY_SHA256={policy_sha256}")
    print(f"P4B_CONFIG_SHA256={config['config_sha256']}")
    print(f"P4B_FOUNDATION_SHA256={policy['frozen_identities']['foundation_sha256']}")
    print(f"P4B_RESULT_SCHEMA_SHA256={result_schema['schema_sha256']}")
    print(f"P4B_ACCEPTED_PIP_FREEZE_SHA256={EXPECTED_P4A_PIP_FREEZE_SHA256}")
    print("P4B_FIRST_RUN_MAX_TASKS=1")
    print("P4B_AUTHORIZED=false")
    print("RESULT_BEARING=false")
    print("SCIENTIFIC_EXECUTION=0")
    return 0


def execute_one(args: argparse.Namespace) -> int:
    if not args.authorization or not args.task_id or not args.run_root:
        raise P4BAuthorizationError(
            "--execute-one requires --authorization, --task-id, and --run-root"
        )
    verify_exact_git_commit(args.repo_root, args.expected_git_commit)
    result = run_authorized_task(
        authorization_path=args.authorization,
        policy_path=args.policy,
        realized_pip_freeze_path=args.pip_freeze,
        expected_git_commit=args.expected_git_commit,
        expected_task_id=args.task_id,
        run_root=args.run_root,
        plan_path=args.plan,
        dataset_path=args.dataset,
        config_path=args.config,
    )
    print("P4B_ONE_TASK_GATE=PASS")
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-one", action="store_true")

    p.add_argument("--repo-root", default=".")
    p.add_argument("--expected-git-commit", required=True)
    p.add_argument("--policy", default="config/P4B_EXECUTION_POLICY_v1.json")
    p.add_argument("--authorization-schema", default="config/P4B_AUTHORIZATION_v1.schema.json")
    p.add_argument("--config", default="config/EXPERIMENT_CONFIG_v1.json")
    p.add_argument("--foundation-lock", default="data/registry/FOUNDATION_LOCK_v1.json")
    p.add_argument("--result-schema", default="config/RESULT_EVIDENCE_SCHEMA_v1.json")
    p.add_argument("--allow-review-head", action="store_true")

    p.add_argument("--authorization")
    p.add_argument("--task-id")
    p.add_argument("--run-root")
    p.add_argument("--pip-freeze")
    p.add_argument("--plan")
    p.add_argument("--dataset")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.validate_only:
        return validate_only(args)

    required = {
        "--pip-freeze": args.pip_freeze,
        "--plan": args.plan,
        "--dataset": args.dataset,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise P4BAuthorizationError("execute-one missing required arguments: " + ", ".join(missing))
    return execute_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
