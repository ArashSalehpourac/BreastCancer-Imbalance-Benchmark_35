#!/usr/bin/env python3
"""P4B validation-only gate and future isolated one-task execution entry point.

The controller validates an exact one-task grant before dispatching a fresh
Python worker. The worker process receives the frozen thread/hash environment
before importing any scientific package. Package validation never dispatches a
scientific worker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def validate_only(args: argparse.Namespace) -> int:
    from bcimbalance.execution_harness import (
        load_foundation_lock,
        load_result_schema,
        verify_exact_git_commit,
    )
    from bcimbalance.experiment_config import load_experiment_config
    from bcimbalance.p4b_authorization import (
        AUTHORIZED_P4B_BASE,
        EXPECTED_P4A_PIP_FREEZE_SHA256,
        P4BAuthorizationError,
        load_policy,
    )

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


def _controller_execute_one(args: argparse.Namespace) -> int:
    from bcimbalance.execution_harness import verify_exact_git_commit
    from bcimbalance.p4b_authorization import P4BAuthorizationError, validate_authorization
    from bcimbalance.scientific_task import find_exact_task, load_frozen_plan

    required = {
        "--authorization": args.authorization,
        "--task-id": args.task_id,
        "--run-root": args.run_root,
        "--pip-freeze": args.pip_freeze,
        "--plan": args.plan,
        "--dataset": args.dataset,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise P4BAuthorizationError("execute-one missing required arguments: " + ", ".join(missing))

    verify_exact_git_commit(args.repo_root, args.expected_git_commit)

    grant = validate_authorization(
        args.authorization,
        policy_path=args.policy,
        expected_git_commit=args.expected_git_commit,
        expected_task_id=args.task_id,
        expected_run_root=args.run_root,
        realized_pip_freeze_path=args.pip_freeze,
    )
    if Path(args.run_root).exists():
        raise P4BAuthorizationError(
            "immutable P4B first-run root already exists; reuse is forbidden"
        )

    plan = load_frozen_plan(args.plan)
    task = find_exact_task(plan, args.task_id)
    seed = int(task["seed"])

    worker_env = os.environ.copy()
    worker_env.update(
        {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": str(seed),
            "P4B_AUTHORIZED_WORKER": "1",
            "P4B_AUTHORIZATION_SHA256": grant.authorization_sha256,
        }
    )

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-execute-one",
        "--repo-root",
        args.repo_root,
        "--expected-git-commit",
        args.expected_git_commit,
        "--policy",
        args.policy,
        "--config",
        args.config,
        "--foundation-lock",
        args.foundation_lock,
        "--result-schema",
        args.result_schema,
        "--authorization-schema",
        args.authorization_schema,
        "--authorization",
        args.authorization,
        "--task-id",
        args.task_id,
        "--run-root",
        args.run_root,
        "--pip-freeze",
        args.pip_freeze,
        "--plan",
        args.plan,
        "--dataset",
        args.dataset,
    ]
    completed = subprocess.run(command, check=False, env=worker_env)
    if completed.returncode != 0:
        raise P4BAuthorizationError(
            f"isolated P4B worker failed with exit code {completed.returncode}"
        )
    return 0


def _worker_execute_one(args: argparse.Namespace) -> int:
    """Scientific worker entry point; environment is checked before science imports."""
    if os.environ.get("P4B_AUTHORIZED_WORKER") != "1":
        raise RuntimeError("P4B worker mode requires controller authorization")
    if not args.task_id or not args.authorization or not args.run_root:
        raise RuntimeError("P4B worker missing exact authorization/task/run-root arguments")

    plan_payload = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    matching = [task for task in plan_payload.get("tasks", []) if task.get("task_id") == args.task_id]
    if len(matching) != 1:
        raise RuntimeError("P4B worker task must resolve exactly once before imports")
    seed = int(matching[0]["seed"])

    required_env = {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": str(seed),
    }
    for key, expected in required_env.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(
                f"P4B worker environment mismatch before scientific imports: {key}"
            )
    if not os.environ.get("P4B_AUTHORIZATION_SHA256"):
        raise RuntimeError("P4B worker authorization receipt is missing")
    if Path(args.run_root).exists():
        raise RuntimeError("immutable P4B first-run root already exists before worker import")

    # Scientific modules are deliberately imported only after the process-level
    # deterministic environment and unused run root have been validated.
    from bcimbalance.execution_harness import verify_exact_git_commit
    from bcimbalance.scientific_task import run_authorized_task

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
    mode.add_argument("--worker-execute-one", action="store_true", help=argparse.SUPPRESS)

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
    if args.worker_execute_one:
        return _worker_execute_one(args)
    return _controller_execute_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
