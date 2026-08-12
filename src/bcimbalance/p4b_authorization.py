"""Fail-closed P4B authorization and run-manifest controls.

Importing or validating this module is non-result-bearing. A scientific task may
proceed only after a one-task authorization record passes every frozen identity
check and the realized environment receipt matches the accepted P4A receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution_harness import write_json_atomic, write_sha256_receipt
from .p4_control import FUTURE_REQUIRED_SUBDIRECTORIES, FUTURE_RESULT_BEARING_ROOT, validate_future_run_root

P4B_POLICY_SCHEMA_VERSION = "p4b-execution-policy/v1"
P4B_AUTH_SCHEMA_VERSION = "p4b-authorization/v1"
AUTHORIZED_P4B_BASE = "ffd7a29c8c05d17ac4e20dae9db75492d1ed1dff"

EXPECTED_DATASET_SHA256 = "27f219231dbb30eecbfc1361407ed641ea01be43316e2c707a1baf82c9795e23"
EXPECTED_FOUNDATION_SHA256 = "49291df347b8aa8453a68cf06296642f3939bfc3fe51ce452d3620d9c228e030"
EXPECTED_SPLIT_SHA256 = "00114e7735fc4eb012ecf248010b18b441cce4a74e0d1540115bbaff543d764a"
EXPECTED_SEED_REGISTRY_SHA256 = "985f5614275ef880213bf775cded0b8e3867fb011f43df817b48d94bb5af73e2"
EXPECTED_P2_CONFIG_SHA256 = "71e82edad62dcf06382cf85dcd87e642049e80668eaa7dbd9913d7a5a5bb7dc9"
EXPECTED_P3_RESULT_SCHEMA_SHA256 = "7819dd71c49d3c5c686ca76079a92f0b5399596d0de71d2964eca8ece8af4686"
EXPECTED_P4A_POLICY_SHA256 = "af3212b00c87afbd83032de5a86b1bb933f840a7e6b675e752278e55c2a80c0b"
EXPECTED_P4A_PIP_FREEZE_SHA256 = "9bdf50ed4d756c753579eb0e4ced6031299ffd6dab1d885f03dfdf58e9978a4e"
EXPECTED_PLAN_SHA256 = "3696604d936e103426fd0527cf8a37b1ec619043fcb32121cb55fe119b94c137"

EXPECTED_IDENTITIES = {
    "dataset_sha256": EXPECTED_DATASET_SHA256,
    "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
    "split_sha256": EXPECTED_SPLIT_SHA256,
    "seed_registry_sha256": EXPECTED_SEED_REGISTRY_SHA256,
    "p2_config_sha256": EXPECTED_P2_CONFIG_SHA256,
    "p3_result_schema_sha256": EXPECTED_P3_RESULT_SCHEMA_SHA256,
    "p4a_policy_sha256": EXPECTED_P4A_POLICY_SHA256,
    "accepted_p4a_pip_freeze_sha256": EXPECTED_P4A_PIP_FREEZE_SHA256,
    "execution_plan_sha256": EXPECTED_PLAN_SHA256,
}


class P4BAuthorizationError(RuntimeError):
    """Raised before scientific execution when P4B authorization is invalid."""


@dataclass(frozen=True)
class AuthorizationGrant:
    exact_git_commit: str
    task_id: str
    run_id: str
    run_root: str
    authorization_ref: str
    authorized_utc: str
    authorization_sha256: str
    p4b_policy_sha256: str
    pip_freeze_sha256: str


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    policy_path = Path(path)
    digest = sha256_file(policy_path)
    with policy_path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)

    expected_scalars = {
        "schema_version": P4B_POLICY_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "issue_number": 13,
        "parent_control_issue": 11,
        "authorized_base": AUTHORIZED_P4B_BASE,
        "implementation_state": "DISABLED_PENDING_EXACT_AUTHORIZATION",
        "validation_only": True,
        "result_bearing_authorized": False,
        "future_result_bearing_root": FUTURE_RESULT_BEARING_ROOT,
    }
    for key, value in expected_scalars.items():
        if policy.get(key) != value:
            raise P4BAuthorizationError(f"P4B policy field mismatch: {key}")

    if tuple(policy.get("required_subdirectories", [])) != FUTURE_REQUIRED_SUBDIRECTORIES:
        raise P4BAuthorizationError("P4B required run subdirectories changed")
    if policy.get("authorization", {}).get("first_run_max_tasks") != 1:
        raise P4BAuthorizationError("P4B first-run authorization must remain max_tasks=1")

    identities = policy.get("frozen_identities", {})
    for key, expected in EXPECTED_IDENTITIES.items():
        if identities.get(key) != expected:
            raise P4BAuthorizationError(f"P4B frozen identity changed: {key}")

    design = policy.get("design", {})
    expected_design = {
        "planned_tasks": 3600,
        "primary_conditions": 24,
        "outer_folds": 50,
        "replicates": 3,
        "threshold": 0.5,
        "positive_class": "M",
    }
    for key, expected in expected_design.items():
        if design.get(key) != expected:
            raise P4BAuthorizationError(f"P4B design field changed: {key}")
    return policy, digest


def _parse_authorized_utc(value: object) -> str:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise P4BAuthorizationError("authorized_utc is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise P4BAuthorizationError("authorized_utc must include a timezone")
    return text


def validate_authorization(
    authorization_path: str | os.PathLike[str],
    *,
    policy_path: str | os.PathLike[str],
    expected_git_commit: str,
    expected_task_id: str,
    expected_run_root: str,
    realized_pip_freeze_path: str | os.PathLike[str],
) -> AuthorizationGrant:
    """Validate one exact future result-bearing grant.

    This function performs no scientific computation and must be called before
    any model/resampler/generator/prediction/metric operation.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", expected_git_commit):
        raise P4BAuthorizationError("expected Git commit must be an exact lowercase 40-character SHA")
    if not expected_task_id:
        raise P4BAuthorizationError("exact task ID is required")

    policy, policy_sha256 = load_policy(policy_path)
    auth_path = Path(authorization_path)
    if not auth_path.is_file():
        raise P4BAuthorizationError("P4B authorization artifact is absent")
    authorization_sha256 = sha256_file(auth_path)
    with auth_path.open("r", encoding="utf-8") as handle:
        auth = json.load(handle)

    allowed_keys = {
        "schema_version", "result_bearing_authorized", "exact_git_commit", "task_ids",
        "max_tasks", "run_id", "run_root", "authorization_ref", "authorized_utc",
        "p4b_policy_sha256", "dataset_sha256", "foundation_sha256", "split_sha256",
        "seed_registry_sha256", "p2_config_sha256", "p3_result_schema_sha256",
        "p4a_policy_sha256", "pip_freeze_sha256", "execution_plan_sha256",
    }
    if set(auth) != allowed_keys:
        raise P4BAuthorizationError("authorization fields differ from frozen P4B schema")
    if auth.get("schema_version") != P4B_AUTH_SCHEMA_VERSION:
        raise P4BAuthorizationError("authorization schema version mismatch")
    if auth.get("result_bearing_authorized") is not True:
        raise P4BAuthorizationError("result-bearing authorization flag is not true")
    if auth.get("exact_git_commit") != expected_git_commit:
        raise P4BAuthorizationError("authorization Git commit mismatch")
    task_ids = auth.get("task_ids")
    if task_ids != [expected_task_id] or auth.get("max_tasks") != 1:
        raise P4BAuthorizationError("authorization must bind exactly one requested task with max_tasks=1")
    if auth.get("p4b_policy_sha256") != policy_sha256:
        raise P4BAuthorizationError("authorization is not bound to the exact P4B policy bytes")

    expected_auth_identities = {
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "seed_registry_sha256": EXPECTED_SEED_REGISTRY_SHA256,
        "p2_config_sha256": EXPECTED_P2_CONFIG_SHA256,
        "p3_result_schema_sha256": EXPECTED_P3_RESULT_SCHEMA_SHA256,
        "p4a_policy_sha256": EXPECTED_P4A_POLICY_SHA256,
        "pip_freeze_sha256": EXPECTED_P4A_PIP_FREEZE_SHA256,
        "execution_plan_sha256": EXPECTED_PLAN_SHA256,
    }
    for key, expected in expected_auth_identities.items():
        if auth.get(key) != expected:
            raise P4BAuthorizationError(f"authorization frozen identity mismatch: {key}")

    actual_freeze = sha256_file(realized_pip_freeze_path)
    if actual_freeze != EXPECTED_P4A_PIP_FREEZE_SHA256:
        raise P4BAuthorizationError(
            "realized pip-freeze receipt differs from the accepted P4A environment"
        )

    run_root = str(expected_run_root)
    run_id = validate_future_run_root(run_root)
    if auth.get("run_root") != run_root or auth.get("run_id") != run_id:
        raise P4BAuthorizationError("authorization run identity/root mismatch")
    if not str(auth.get("authorization_ref", "")).strip():
        raise P4BAuthorizationError("authorization reference is required")
    authorized_utc = _parse_authorized_utc(auth.get("authorized_utc"))

    return AuthorizationGrant(
        exact_git_commit=expected_git_commit,
        task_id=expected_task_id,
        run_id=run_id,
        run_root=run_root,
        authorization_ref=str(auth["authorization_ref"]),
        authorized_utc=authorized_utc,
        authorization_sha256=authorization_sha256,
        p4b_policy_sha256=policy_sha256,
        pip_freeze_sha256=actual_freeze,
    )


def build_preexecution_manifest(
    grant: AuthorizationGrant,
    *,
    task: dict[str, Any],
    start_utc: datetime | None = None,
) -> dict[str, Any]:
    moment = start_utc or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise P4BAuthorizationError("run start time must be timezone-aware")
    return {
        "schema_version": "p4b-run-manifest/v1",
        "protocol_version": "1.0",
        "phase": "P4B",
        "run_id": grant.run_id,
        "git_commit": grant.exact_git_commit,
        "authorization_ref": grant.authorization_ref,
        "authorization_sha256": grant.authorization_sha256,
        "authorized_utc": grant.authorized_utc,
        "p4b_policy_sha256": grant.p4b_policy_sha256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "seed_registry_sha256": EXPECTED_SEED_REGISTRY_SHA256,
        "p2_config_sha256": EXPECTED_P2_CONFIG_SHA256,
        "p3_result_schema_sha256": EXPECTED_P3_RESULT_SCHEMA_SHA256,
        "p4a_policy_sha256": EXPECTED_P4A_POLICY_SHA256,
        "pip_freeze_sha256": grant.pip_freeze_sha256,
        "execution_plan_sha256": EXPECTED_PLAN_SHA256,
        "master_seed": 20260810,
        "planned_tasks_total": 3600,
        "authorized_task_ids": [grant.task_id],
        "max_tasks": 1,
        "task": task,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "hardware_summary": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "utc_start": moment.astimezone(timezone.utc).isoformat(),
        "result_bearing": True,
        "result_bearing_authorized": True,
    }


def write_preexecution_manifest(
    run_root: str | os.PathLike[str],
    manifest: dict[str, Any],
) -> tuple[Path, str]:
    root = Path(run_root)
    destination = root / "manifests" / "p4b_run_manifest.json"
    if destination.exists():
        raise P4BAuthorizationError("pre-execution run manifest already exists")
    digest = write_json_atomic(destination, manifest)
    write_sha256_receipt(destination, digest)
    return destination, digest
