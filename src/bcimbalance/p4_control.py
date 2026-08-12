"""P4A execution-control layer for the frozen breast-cancer benchmark.

P4A is intentionally non-result-bearing. It validates Colab path policy,
reuses the accepted P3 preflight/dry-run, writes control-only manifests and
receipts, and audits checkpoint/evidence-path semantics. It does not execute
any scientific condition.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution_harness import (
    EXPECTED_EXECUTION_TASKS,
    EXPECTED_FOUNDATION_SHA256,
    EXPECTED_PRIMARY_CONDITIONS,
    EXPECTED_RESULT_SCHEMA_SHA256,
    PreflightError,
    checkpoint_state,
    run_preflight,
    write_json_atomic,
    write_sha256_receipt,
)
from .experiment_config import EXPECTED_CONFIG_SHA256

P4A_SCHEMA_VERSION = "p4a-execution-control/v1"
P4A_POLICY_SCHEMA_VERSION = "p4a-execution-control-policy/v1"
AUTHORIZED_P4_BASE = "57f721960f87ac189851024b477fbcdb771a7ed4"
EXPECTED_P4A_POLICY_SHA256 = "af3212b00c87afbd83032de5a86b1bb933f840a7e6b675e752278e55c2a80c0b"
EXPECTED_DATASET_SHA256 = "27f219231dbb30eecbfc1361407ed641ea01be43316e2c707a1baf82c9795e23"
EXPECTED_SPLIT_SHA256 = "00114e7735fc4eb012ecf248010b18b441cce4a74e0d1540115bbaff543d764a"
EXPECTED_SEED_REGISTRY_SHA256 = "985f5614275ef880213bf775cded0b8e3867fb011f43df817b48d94bb5af73e2"
EXPECTED_PLAN_TASKS = 3600
EXPECTED_OUTER_FOLDS = 50
EXPECTED_REPLICATES = 3

CANONICAL_COLAB_DATASET_PATH = (
    "/content/drive/MyDrive/35/01_Experiment_Evidence/00_Dataset/"
    "wdbc_canonical_lf.csv"
)
P4A_CONTROL_ROOT = (
    "/content/drive/MyDrive/35/01_Experiment_Evidence/P4A_Control_Validation"
)
FUTURE_RESULT_BEARING_ROOT = (
    "/content/drive/MyDrive/35/01_Experiment_Evidence/P4_Result_Bearing_Runs"
)
FUTURE_REQUIRED_SUBDIRECTORIES = (
    "preflight",
    "manifests",
    "splits",
    "raw",
    "checkpoints",
    "logs",
    "receipts",
)
TASK_SOURCE_PREFIX = "evidence/conditions/"
TASK_TARGET_PREFIX = "raw/conditions/"

FORBIDDEN_P4A_CALLS = frozenset(
    {
        "fit",
        "fit_resample",
        "sample",
        "sample_from_conditions",
        "predict",
        "predict_proba",
        "score",
    }
)


class P4AControlError(RuntimeError):
    """Raised when a P4A control gate fails closed."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_control_policy(path: str | os.PathLike[str]) -> dict[str, Any]:
    policy_path = Path(path)
    actual = sha256_file(policy_path)
    if actual != EXPECTED_P4A_POLICY_SHA256:
        raise P4AControlError(
            "P4A policy SHA-256 mismatch: "
            f"expected={EXPECTED_P4A_POLICY_SHA256} actual={actual}"
        )
    with policy_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    expected_scalars = {
        "schema_version": P4A_POLICY_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "result_bearing": False,
        "p4b_authorized": False,
        "authorized_base": AUTHORIZED_P4_BASE,
        "issue_number": 11,
        "canonical_colab_dataset_path": CANONICAL_COLAB_DATASET_PATH,
        "p4a_control_root": P4A_CONTROL_ROOT,
        "future_result_bearing_root": FUTURE_RESULT_BEARING_ROOT,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise P4AControlError(f"P4A policy field mismatch: {key}")

    if tuple(payload.get("future_required_subdirectories", [])) != FUTURE_REQUIRED_SUBDIRECTORIES:
        raise P4AControlError("future required subdirectory policy changed")

    mapping = payload.get("future_task_record_mapping", {})
    if mapping.get("source_prefix") != TASK_SOURCE_PREFIX:
        raise P4AControlError("future task source prefix changed")
    if mapping.get("target_prefix") != TASK_TARGET_PREFIX:
        raise P4AControlError("future task target prefix changed")

    identities = payload.get("frozen_identities", {})
    expected_identities = {
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "seed_registry_sha256": EXPECTED_SEED_REGISTRY_SHA256,
        "p2_config_sha256": EXPECTED_CONFIG_SHA256,
        "p3_result_schema_sha256": EXPECTED_RESULT_SCHEMA_SHA256,
        "p3_merge": AUTHORIZED_P4_BASE,
    }
    for key, expected in expected_identities.items():
        if identities.get(key) != expected:
            raise P4AControlError(f"frozen identity changed: {key}")

    design = payload.get("design", {})
    if design.get("primary_conditions") != EXPECTED_PRIMARY_CONDITIONS:
        raise P4AControlError("primary condition count changed")
    if design.get("outer_folds") != EXPECTED_OUTER_FOLDS:
        raise P4AControlError("outer fold count changed")
    if design.get("replicates") != EXPECTED_REPLICATES:
        raise P4AControlError("replicate count changed")
    if design.get("planned_tasks") != EXPECTED_PLAN_TASKS:
        raise P4AControlError("planned task count changed")
    if float(design.get("threshold", -1)) != 0.50:
        raise P4AControlError("threshold changed")
    if design.get("positive_class") != "M":
        raise P4AControlError("positive class changed")

    return payload


def validate_colab_path_policy(
    dataset_path: str | os.PathLike[str],
    control_root: str | os.PathLike[str],
    *,
    require_exact_colab_paths: bool,
) -> None:
    dataset = str(dataset_path)
    root = str(control_root)

    future_root = Path(FUTURE_RESULT_BEARING_ROOT)
    control = Path(root)
    try:
        control.relative_to(future_root)
    except ValueError:
        pass
    else:
        raise P4AControlError(
            "P4A control output may not be written under the future result-bearing root"
        )

    if require_exact_colab_paths:
        if dataset != CANONICAL_COLAB_DATASET_PATH:
            raise P4AControlError("Colab dataset path differs from frozen canonical path")
        if Path(root).parent != Path(P4A_CONTROL_ROOT):
            raise P4AControlError(
                "Colab P4A control session must be a direct child of the frozen control root"
            )


def build_control_session_id(
    git_commit: str,
    *,
    utc_time: datetime | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise P4AControlError("Git commit must be an exact 40-character lowercase SHA")
    moment = utc_time or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise P4AControlError("UTC time must be timezone-aware")
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"P4A_CONTROL_{stamp}_{git_commit[:12]}"


def build_future_run_id(
    git_commit: str,
    *,
    utc_time: datetime,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise P4AControlError("Git commit must be an exact 40-character lowercase SHA")
    if utc_time.tzinfo is None:
        raise P4AControlError("future run UTC time must be timezone-aware")
    stamp = utc_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"P4_RUN_{stamp}_{git_commit[:12]}"


def validate_future_run_root(run_root: str | os.PathLike[str]) -> str:
    root = Path(run_root)
    if root.parent != Path(FUTURE_RESULT_BEARING_ROOT):
        raise P4AControlError("future run root must be a direct child of the frozen P4 root")
    if not re.fullmatch(r"P4_RUN_\d{8}T\d{6}Z_[0-9a-f]{12}", root.name):
        raise P4AControlError("future run root name does not match the immutable run-id policy")
    return root.name


def required_future_paths(run_root: str | os.PathLike[str]) -> dict[str, str]:
    validate_future_run_root(run_root)
    root = Path(run_root)
    return {name: str(root / name) for name in FUTURE_REQUIRED_SUBDIRECTORIES}


def resolve_task_record_path(
    task: dict[str, Any],
    run_root: str | os.PathLike[str],
) -> Path:
    validate_future_run_root(run_root)
    logical = str(task.get("evidence_path", ""))
    if not logical.startswith(TASK_SOURCE_PREFIX):
        raise P4AControlError("P3 task evidence path does not use the frozen source prefix")
    suffix = logical[len(TASK_SOURCE_PREFIX):]
    if not suffix or Path(suffix).is_absolute() or ".." in Path(suffix).parts:
        raise P4AControlError("unsafe task evidence suffix")
    destination = Path(run_root) / TASK_TARGET_PREFIX / suffix
    raw_root = (Path(run_root) / "raw").resolve()
    try:
        destination.resolve().relative_to(raw_root)
    except ValueError as exc:
        raise P4AControlError("task record path escapes the future raw evidence root") from exc
    return destination


def audit_checkpoint_inventory(
    checkpoint_root: str | os.PathLike[str],
    plan: dict[str, Any],
    *,
    require_all_pending: bool,
) -> dict[str, int]:
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = {str(task["task_id"]): task for task in plan["tasks"]}
    known = set(tasks)

    observed_marker_ids: set[str] = set()
    for path in root.glob("*.json"):
        name = path.name
        if name.endswith(".inprogress.json"):
            observed_marker_ids.add(name[: -len(".inprogress.json")])
        elif name.endswith(".complete.json"):
            observed_marker_ids.add(name[: -len(".complete.json")])
        else:
            raise P4AControlError(f"unrecognized checkpoint file: {name}")

    unknown = observed_marker_ids - known
    if unknown:
        raise P4AControlError(f"checkpoint marker references unknown task: {sorted(unknown)[0]}")

    counts = {"PENDING": 0, "INCOMPLETE": 0, "COMPLETE": 0}
    for task_id in tasks:
        try:
            state = checkpoint_state(root, task_id)
        except PreflightError as exc:
            raise P4AControlError(str(exc)) from exc
        counts[state] += 1

    if sum(counts.values()) != EXPECTED_EXECUTION_TASKS:
        raise P4AControlError("checkpoint inventory does not cover all frozen tasks")
    if require_all_pending and counts != {
        "PENDING": EXPECTED_EXECUTION_TASKS,
        "INCOMPLETE": 0,
        "COMPLETE": 0,
    }:
        raise P4AControlError(
            "P4A validation requires every scientific task to remain PENDING"
        )
    return counts


def validate_preflight_report(
    report: dict[str, Any],
    plan: dict[str, Any],
    *,
    expected_git_commit: str,
) -> None:
    expected = {
        "status": "PASS",
        "result_bearing": False,
        "scientific_execution": 0,
        "git_commit": expected_git_commit,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "seed_registry_sha256": EXPECTED_SEED_REGISTRY_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "result_schema_sha256": EXPECTED_RESULT_SCHEMA_SHA256,
        "outer_folds": EXPECTED_OUTER_FOLDS,
        "primary_conditions": EXPECTED_PRIMARY_CONDITIONS,
        "planned_execution_tasks": EXPECTED_EXECUTION_TASKS,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise P4AControlError(f"P3 preflight report mismatch: {key}")

    if plan.get("result_bearing") is not False:
        raise P4AControlError("P3 execution plan must remain non-result-bearing")
    if plan.get("execution_task_count") != EXPECTED_EXECUTION_TASKS:
        raise P4AControlError("P3 execution plan task count changed")
    if plan.get("primary_condition_count") != EXPECTED_PRIMARY_CONDITIONS:
        raise P4AControlError("P3 execution plan condition count changed")
    if report.get("plan_sha256") != plan.get("plan_sha256"):
        raise P4AControlError("P3 preflight report/plan hash cross-link mismatch")


def build_control_manifest(
    *,
    session_id: str,
    authorization_ref: str,
    git_commit: str,
    preflight_report: dict[str, Any],
    checkpoint_summary: dict[str, int],
    policy_sha256: str,
) -> dict[str, Any]:
    if not authorization_ref.strip():
        raise P4AControlError("P4A authorization reference is required")
    manifest = {
        "schema_version": P4A_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "phase": "P4A",
        "control_session_id": session_id,
        "authorization_ref": authorization_ref,
        "git_commit": git_commit,
        "authorized_p4_base": AUTHORIZED_P4_BASE,
        "policy_sha256": policy_sha256,
        "dataset_sha256": preflight_report["dataset_sha256"],
        "foundation_sha256": preflight_report["foundation_sha256"],
        "split_sha256": preflight_report["split_sha256"],
        "seed_registry_sha256": preflight_report["seed_registry_sha256"],
        "p2_config_sha256": preflight_report["config_sha256"],
        "p3_result_schema_sha256": preflight_report["result_schema_sha256"],
        "pip_freeze_sha256": preflight_report["pip_freeze_sha256"],
        "plan_sha256": preflight_report["plan_sha256"],
        "planned_tasks": preflight_report["planned_execution_tasks"],
        "checkpoint_summary": checkpoint_summary,
        "future_result_bearing_root": FUTURE_RESULT_BEARING_ROOT,
        "future_required_subdirectories": list(FUTURE_REQUIRED_SUBDIRECTORIES),
        "task_record_mapping": {
            "source_prefix": TASK_SOURCE_PREFIX,
            "target_prefix": TASK_TARGET_PREFIX,
        },
        "p4b_authorized": False,
        "result_bearing": False,
        "scientific_execution": 0,
    }
    return manifest


def run_p4a_control(
    *,
    dataset_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    expected_git_commit: str,
    control_root: str | os.PathLike[str],
    policy_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    foundation_lock_path: str | os.PathLike[str],
    result_schema_path: str | os.PathLike[str],
    authorization_ref: str,
    require_exact_colab_paths: bool,
    utc_time: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_colab_path_policy(
        dataset_path,
        control_root,
        require_exact_colab_paths=require_exact_colab_paths,
    )
    load_control_policy(policy_path)

    root = Path(control_root)
    preflight_dir = root / "preflight"
    manifests_dir = root / "manifests"
    checkpoints_dir = root / "checkpoints"
    logs_dir = root / "logs"
    receipts_dir = root / "receipts"
    for directory in (preflight_dir, manifests_dir, checkpoints_dir, logs_dir, receipts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    preflight_report, plan = run_preflight(
        dataset_path=dataset_path,
        config_path=config_path,
        foundation_lock_path=foundation_lock_path,
        result_schema_path=result_schema_path,
        repo_root=repo_root,
        expected_git_commit=expected_git_commit,
        pip_freeze_path=preflight_dir / "pip_freeze.txt",
    )
    validate_preflight_report(
        preflight_report,
        plan,
        expected_git_commit=expected_git_commit,
    )

    report_path = preflight_dir / "p3_preflight_report.json"
    report_sha = write_json_atomic(report_path, preflight_report)
    write_sha256_receipt(report_path, report_sha)

    plan_path = preflight_dir / "p3_execution_plan.json"
    plan_sha = write_json_atomic(plan_path, plan)
    write_sha256_receipt(plan_path, plan_sha)

    checkpoint_summary = audit_checkpoint_inventory(
        checkpoints_dir,
        plan,
        require_all_pending=True,
    )

    session_id = build_control_session_id(expected_git_commit, utc_time=utc_time)
    manifest = build_control_manifest(
        session_id=session_id,
        authorization_ref=authorization_ref,
        git_commit=expected_git_commit,
        preflight_report=preflight_report,
        checkpoint_summary=checkpoint_summary,
        policy_sha256=EXPECTED_P4A_POLICY_SHA256,
    )

    manifest_path = manifests_dir / "p4a_control_manifest.json"
    manifest_sha = write_json_atomic(manifest_path, manifest)
    write_sha256_receipt(manifest_path, manifest_sha)

    path_policy_report = {
        "schema_version": "p4a-path-policy/v1",
        "result_bearing": False,
        "scientific_execution": 0,
        "canonical_colab_dataset_path": CANONICAL_COLAB_DATASET_PATH,
        "p4a_control_root": P4A_CONTROL_ROOT,
        "future_result_bearing_root": FUTURE_RESULT_BEARING_ROOT,
        "future_required_subdirectories": list(FUTURE_REQUIRED_SUBDIRECTORIES),
        "task_record_mapping": {
            "source_prefix": TASK_SOURCE_PREFIX,
            "target_prefix": TASK_TARGET_PREFIX,
        },
        "future_run_id_pattern": "P4_RUN_YYYYMMDDTHHMMSSZ_<12-char-exact-git-sha>",
        "future_run_root_created": False,
        "p4b_authorized": False,
    }
    path_report_path = manifests_dir / "p4a_path_policy_report.json"
    path_report_sha = write_json_atomic(path_report_path, path_policy_report)
    write_sha256_receipt(path_report_path, path_report_sha)

    receipt_index = {
        "schema_version": "p4a-receipt-index/v1",
        "result_bearing": False,
        "scientific_execution": 0,
        "files": {
            "p3_preflight_report.json": report_sha,
            "p3_execution_plan.json": plan_sha,
            "p4a_control_manifest.json": manifest_sha,
            "p4a_path_policy_report.json": path_report_sha,
            "pip_freeze.txt": preflight_report["pip_freeze_sha256"],
        },
    }
    receipt_index_path = receipts_dir / "p4a_receipt_index.json"
    receipt_index_sha = write_json_atomic(receipt_index_path, receipt_index)
    write_sha256_receipt(receipt_index_path, receipt_index_sha)

    control_report = {
        "schema_version": P4A_SCHEMA_VERSION,
        "status": "PASS",
        "result_bearing": False,
        "scientific_execution": 0,
        "p4b_authorized": False,
        "control_session_id": session_id,
        "git_commit": expected_git_commit,
        "policy_sha256": EXPECTED_P4A_POLICY_SHA256,
        "preflight_report_file_sha256": report_sha,
        "execution_plan_file_sha256": plan_sha,
        "control_manifest_file_sha256": manifest_sha,
        "path_policy_file_sha256": path_report_sha,
        "receipt_index_file_sha256": receipt_index_sha,
        "checkpoint_summary": checkpoint_summary,
    }
    return control_report, manifest, plan
