"""P3 preflight and dry-run infrastructure for the frozen primary benchmark.

This module is intentionally non-result-bearing. It validates immutable inputs,
realized software, Git identity, and the complete execution plan. It does not
fit classifiers, execute resamplers, fit/sample CTGAN, predict, calculate
scientific metrics, or perform statistics.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from .dataset import load_registered_dataset, sha256_file
from .experiment_config import EXPECTED_CONFIG_SHA256, EXPECTED_PINS, load_experiment_config
from .folds import build_outer_folds
from .seeds import generate_seed_registry, verify_seed_registry

P3_SCHEMA_VERSION = "execution-preflight/v1"
PLAN_SCHEMA_VERSION = "execution-plan/v1"
CHECKPOINT_SCHEMA_VERSION = "condition-checkpoint/v1"
RESULT_SCHEMA_VERSION = "result-evidence/v1"

AUTHORIZED_P3_BASE = "8c6ff728375de033b32fac8b32240a22e98401e1"
EXPECTED_FOUNDATION_SHA256 = "49291df347b8aa8453a68cf06296642f3939bfc3fe51ce452d3620d9c228e030"
EXPECTED_RESULT_SCHEMA_SHA256 = "7819dd71c49d3c5c686ca76079a92f0b5399596d0de71d2964eca8ece8af4686"
EXPECTED_OUTER_FOLDS = 50
EXPECTED_EXECUTION_TASKS = 3600
EXPECTED_PRIMARY_CONDITIONS = 24
EXPECTED_REPLICATES = 3

FORBIDDEN_SCIENTIFIC_CALLS = frozenset(
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


class PreflightError(RuntimeError):
    """Raised when any P3 fail-closed preflight gate does not pass."""


def _canonical_hash(payload: dict[str, Any], *, omit: str | None = None) -> str:
    copy = dict(payload)
    if omit is not None:
        copy.pop(omit, None)
    encoded = json.dumps(
        copy, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_json_atomic(path: str | os.PathLike[str], payload: dict[str, Any]) -> str:
    destination = Path(path)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    _atomic_write_bytes(destination, encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_text_atomic(path: str | os.PathLike[str], text: str) -> str:
    destination = Path(path)
    encoded = text.encode("utf-8")
    _atomic_write_bytes(destination, encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_sha256_receipt(path: str | os.PathLike[str], digest: str) -> Path:
    target = Path(path)
    receipt = target.with_name(target.name + ".sha256")
    write_text_atomic(receipt, f"{digest}  {target.name}\n")
    return receipt


def load_foundation_lock(path: str | os.PathLike[str]) -> dict[str, Any]:
    lock_path = Path(path)
    actual = sha256_file(lock_path)
    if actual != EXPECTED_FOUNDATION_SHA256:
        raise PreflightError(
            "foundation lock SHA-256 mismatch: "
            f"expected={EXPECTED_FOUNDATION_SHA256} actual={actual}"
        )
    with lock_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "foundation-lock/v1":
        raise PreflightError("foundation lock schema mismatch")
    if payload.get("protocol_version") != "1.0":
        raise PreflightError("foundation lock protocol version mismatch")
    if payload.get("result_bearing") is not False:
        raise PreflightError("foundation lock must remain non-result-bearing")
    return payload


def load_result_schema(path: str | os.PathLike[str]) -> dict[str, Any]:
    schema_path = Path(path)
    with schema_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise PreflightError("result evidence schema version mismatch")
    if payload.get("protocol_version") != "1.0":
        raise PreflightError("result evidence schema protocol mismatch")
    if payload.get("artifact_result_bearing") is not False:
        raise PreflightError("P3 result-schema artifact must be non-result-bearing")
    if payload.get("describes_result_bearing_evidence") is not True:
        raise PreflightError("result evidence schema purpose flag mismatch")
    stored = str(payload.get("schema_sha256", ""))
    computed = _canonical_hash(payload, omit="schema_sha256")
    if stored != EXPECTED_RESULT_SCHEMA_SHA256 or computed != stored:
        raise PreflightError(
            f"result evidence schema hash mismatch: stored={stored} computed={computed}"
        )
    prediction_columns = payload["required_artifacts"]["predictions"]["required_columns"]
    required_prediction_columns = [
        "run_id",
        "protocol_version",
        "git_commit",
        "dataset_sha256",
        "row_id",
        "repeat",
        "fold",
        "method",
        "classifier",
        "replicate",
        "seed",
        "y_true",
        "p_malignant",
        "threshold",
        "y_pred",
    ]
    if prediction_columns != required_prediction_columns:
        raise PreflightError("prediction evidence column schema changed")
    return payload


def verify_direct_pins() -> dict[str, str]:
    realized: dict[str, str] = {}
    for distribution, expected in EXPECTED_PINS.items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise PreflightError(f"required package is not installed: {distribution}") from exc
        if actual != expected:
            raise PreflightError(
                f"package pin mismatch for {distribution}: expected={expected} actual={actual}"
            )
        realized[distribution] = actual
    return dict(sorted(realized.items()))


def capture_pip_freeze(path: str | os.PathLike[str]) -> tuple[str, list[str]]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )
    lines = sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    normalized = "\n".join(lines) + "\n"
    digest = write_text_atomic(path, normalized)
    write_sha256_receipt(path, digest)
    return digest, lines


def current_git_commit(repo_root: str | os.PathLike[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PreflightError("unable to resolve repository HEAD") from exc


def verify_exact_git_commit(
    repo_root: str | os.PathLike[str],
    expected_git_commit: str,
) -> str:
    actual = current_git_commit(repo_root)
    if actual != expected_git_commit:
        raise PreflightError(
            f"Git commit mismatch: expected={expected_git_commit} actual={actual}"
        )
    return actual


def _validate_foundation_cross_links(
    foundation: dict[str, Any],
    config: dict[str, Any],
) -> None:
    config_foundation = config["foundation_lock"]
    checks = {
        "dataset_sha256": foundation["dataset"]["sha256"],
        "split_sha256": foundation["outer_folds"]["sha256"],
        "seed_registry_sha256": foundation["seed_registry"]["sha256"],
        "master_seed": foundation["seed_registry"]["master_seed"],
    }
    for key, actual in checks.items():
        if config_foundation.get(key) != actual:
            raise PreflightError(f"foundation/config cross-link mismatch: {key}")
    if config_foundation.get("sha256") != EXPECTED_FOUNDATION_SHA256:
        raise PreflightError("P2 configuration references the wrong foundation lock")


def build_execution_plan(
    config: dict[str, Any],
    seed_registry: dict[str, Any],
    *,
    evidence_root: str = "evidence/conditions",
) -> dict[str, Any]:
    verify_seed_registry(seed_registry)
    if seed_registry.get("registry_sha256") != config["foundation_lock"]["seed_registry_sha256"]:
        raise PreflightError("seed registry hash differs from P2 configuration")

    methods = list(config["primary_design"]["methods"])
    classifiers = list(config["primary_design"]["classifiers"])
    if len(methods) * len(classifiers) != EXPECTED_PRIMARY_CONDITIONS:
        raise PreflightError("primary method×classifier condition count is not 24")
    if int(config["primary_design"]["replicates"]) != EXPECTED_REPLICATES:
        raise PreflightError("replicate count differs from frozen P2 configuration")

    primary_conditions = [
        {
            "primary_condition_id": f"{method}__{classifier}",
            "method": method,
            "classifier": classifier,
        }
        for method in methods
        for classifier in classifiers
    ]

    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for record in seed_registry["records"]:
        repeat_index = int(record["repeat_index"])
        fold_index = int(record["fold_index"])
        method = str(record["method"])
        classifier = str(record["model"])
        replicate_index = int(record["replicate_index"])
        seed = int(record["seed"])

        task_id = (
            f"r{repeat_index:02d}-f{fold_index:02d}-"
            f"{method}-{classifier}-rep{replicate_index:02d}"
        )
        relative_path = (
            f"{evidence_root}/{method}/{classifier}/"
            f"repeat_{repeat_index:02d}/fold_{fold_index:02d}/"
            f"replicate_{replicate_index:02d}/{task_id}.json"
        )
        if task_id in seen_ids or relative_path in seen_paths:
            raise PreflightError(f"duplicate planned task or evidence path: {task_id}")
        seen_ids.add(task_id)
        seen_paths.add(relative_path)
        tasks.append(
            {
                "task_id": task_id,
                "primary_condition_id": f"{method}__{classifier}",
                "repeat_index": repeat_index,
                "fold_index": fold_index,
                "method": method,
                "classifier": classifier,
                "replicate_index": replicate_index,
                "seed": seed,
                "evidence_path": relative_path,
            }
        )

    if len(tasks) != EXPECTED_EXECUTION_TASKS:
        raise PreflightError(
            f"planned task count must be {EXPECTED_EXECUTION_TASKS}; got {len(tasks)}"
        )
    if len(seen_ids) != EXPECTED_EXECUTION_TASKS:
        raise PreflightError("planned task identifiers are not unique")

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "result_bearing": False,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
        "result_schema_sha256": EXPECTED_RESULT_SCHEMA_SHA256,
        "primary_condition_count": len(primary_conditions),
        "execution_task_count": len(tasks),
        "primary_conditions": primary_conditions,
        "tasks": tasks,
    }
    plan["plan_sha256"] = _canonical_hash(plan)
    return plan


def checkpoint_state(
    checkpoint_root: str | os.PathLike[str],
    task_id: str,
) -> str:
    root = Path(checkpoint_root)
    complete = root / f"{task_id}.complete.json"
    inprogress = root / f"{task_id}.inprogress.json"
    if complete.exists() and inprogress.exists():
        raise PreflightError(f"ambiguous checkpoint state for {task_id}")
    if complete.exists():
        return "COMPLETE"
    if inprogress.exists():
        return "INCOMPLETE"
    return "PENDING"


def write_inprogress_checkpoint(
    checkpoint_root: str | os.PathLike[str],
    task: dict[str, Any],
    *,
    git_commit: str,
) -> Path:
    root = Path(checkpoint_root)
    task_id = str(task["task_id"])
    state = checkpoint_state(root, task_id)
    if state != "PENDING":
        raise PreflightError(
            f"cannot begin task {task_id}; checkpoint state is already {state}"
        )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "result_bearing": False,
        "task_id": task_id,
        "git_commit": git_commit,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
        "state": "INPROGRESS",
    }
    destination = root / f"{task_id}.inprogress.json"
    write_json_atomic(destination, payload)
    return destination


def run_preflight(
    *,
    dataset_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    foundation_lock_path: str | os.PathLike[str],
    result_schema_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    expected_git_commit: str,
    pip_freeze_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    git_commit = verify_exact_git_commit(repo_root, expected_git_commit)
    config = load_experiment_config(config_path)
    if config.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise PreflightError("P2 configuration digest changed")

    foundation = load_foundation_lock(foundation_lock_path)
    _validate_foundation_cross_links(foundation, config)
    result_schema = load_result_schema(result_schema_path)

    expected_dataset = foundation["dataset"]["sha256"]
    frame = load_registered_dataset(dataset_path, expected_dataset)
    if len(frame) != int(foundation["dataset"]["rows"]):
        raise PreflightError("dataset row count differs from foundation lock")

    split_artifact = build_outer_folds(
        frame,
        dataset_sha256=expected_dataset,
        master_seed=int(foundation["seed_registry"]["master_seed"]),
    )
    if len(split_artifact["folds"]) != EXPECTED_OUTER_FOLDS:
        raise PreflightError("outer fold count differs from frozen 50-fold design")
    if split_artifact["split_sha256"] != foundation["outer_folds"]["sha256"]:
        raise PreflightError("regenerated split hash differs from foundation lock")

    seed_registry = generate_seed_registry(
        master_seed=int(foundation["seed_registry"]["master_seed"])
    )
    verify_seed_registry(seed_registry)
    if seed_registry["registry_sha256"] != foundation["seed_registry"]["sha256"]:
        raise PreflightError("regenerated seed registry differs from foundation lock")

    realized_pins = verify_direct_pins()
    pip_freeze_sha256, pip_freeze_lines = capture_pip_freeze(pip_freeze_path)
    plan = build_execution_plan(config, seed_registry)

    report: dict[str, Any] = {
        "schema_version": P3_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "result_bearing": False,
        "status": "PASS",
        "authorized_p3_base": AUTHORIZED_P3_BASE,
        "git_commit": git_commit,
        "dataset_sha256": expected_dataset,
        "dataset_rows": len(frame),
        "split_sha256": split_artifact["split_sha256"],
        "outer_folds": len(split_artifact["folds"]),
        "seed_registry_sha256": seed_registry["registry_sha256"],
        "seed_records": int(seed_registry["n_records"]),
        "config_sha256": config["config_sha256"],
        "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
        "result_schema_sha256": result_schema["schema_sha256"],
        "direct_package_pins": realized_pins,
        "pip_freeze_sha256": pip_freeze_sha256,
        "pip_freeze_line_count": len(pip_freeze_lines),
        "primary_conditions": plan["primary_condition_count"],
        "planned_execution_tasks": plan["execution_task_count"],
        "plan_sha256": plan["plan_sha256"],
        "scientific_execution": 0,
    }
    return report, plan
