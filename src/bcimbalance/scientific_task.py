"""One-task scientific executor for future P4B authorization.

The public entry point is fail-closed: authorization is validated before any
scientific package call or result-bearing run directory is created. CI and
implementation review use validation-only paths and never call the internal
scientific execution function.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN, load_registered_dataset
from .execution_harness import checkpoint_state, write_json_atomic, write_sha256_receipt
from .experiment_config import load_experiment_config
from .folds import build_outer_folds
from .p4_control import FUTURE_REQUIRED_SUBDIRECTORIES, required_future_paths, resolve_task_record_path
from .p4b_authorization import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_PLAN_SHA256,
    P4BAuthorizationError,
    AuthorizationGrant,
    build_preexecution_manifest,
    validate_authorization,
    write_preexecution_manifest,
)

TASK_EVIDENCE_SCHEMA_VERSION = "p4b-task-evidence/v1"
TASK_CHECKPOINT_SCHEMA_VERSION = "p4b-task-checkpoint/v1"


class ScientificTaskError(RuntimeError):
    """Raised when an authorized one-task execution fails closed."""


def load_frozen_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("schema_version") != "execution-plan/v1":
        raise ScientificTaskError("execution-plan schema mismatch")
    if plan.get("result_bearing") is not False:
        raise ScientificTaskError("frozen P3 plan artifact must remain non-result-bearing")
    if plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        raise ScientificTaskError("canonical execution-plan SHA-256 mismatch")
    if plan.get("execution_task_count") != 3600 or len(plan.get("tasks", [])) != 3600:
        raise ScientificTaskError("frozen execution plan must contain exactly 3600 tasks")
    return plan


def find_exact_task(plan: dict[str, Any], task_id: str) -> dict[str, Any]:
    matches = [task for task in plan["tasks"] if task.get("task_id") == task_id]
    if len(matches) != 1:
        raise ScientificTaskError(f"exact task ID must resolve once; got {len(matches)}")
    return dict(matches[0])


def _prepare_runtime(seed: int) -> None:
    for key in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise ScientificTaskError("CUDA is forbidden for the frozen primary P4B execution")


def _replace_seed(value: Any, seed: int) -> Any:
    if value == "$SEED":
        return seed
    if isinstance(value, list):
        return [_replace_seed(item, seed) for item in value]
    if isinstance(value, dict):
        return {key: _replace_seed(item, seed) for key, item in value.items()}
    return value


def _build_classifier(name: str, config: dict[str, Any], seed: int) -> Any:
    params = _replace_seed(dict(config["classifiers"][name]["parameters"]), seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    if name == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(**params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    raise ScientificTaskError(f"unknown frozen classifier: {name}")


def _apply_imbalance_method(
    method: str,
    X_train: Any,
    y_train: Any,
    *,
    train_labels: Any,
    feature_names: list[str],
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    import pandas as pd

    before_b = int((train_labels == "B").sum())
    before_m = int((train_labels == "M").sum())
    audit: dict[str, Any] = {
        "method": method,
        "training_only": True,
        "before": {"B": before_b, "M": before_m},
        "synthetic_rows": 0,
    }
    if method == "baseline":
        audit["after"] = {"B": before_b, "M": before_m}
        return X_train, y_train, audit

    if method == "adasyn":
        from imblearn.over_sampling import ADASYN
        params = _replace_seed(config["imbalance_methods"][method]["parameters"], seed)
        sampler = ADASYN(**params)
        X_out, y_out = sampler.fit_resample(X_train, y_train)
    elif method == "borderline_smote":
        from imblearn.over_sampling import BorderlineSMOTE
        params = _replace_seed(config["imbalance_methods"][method]["parameters"], seed)
        sampler = BorderlineSMOTE(**params)
        X_out, y_out = sampler.fit_resample(X_train, y_train)
    elif method == "smote":
        from imblearn.over_sampling import SMOTE
        params = _replace_seed(config["imbalance_methods"][method]["parameters"], seed)
        sampler = SMOTE(**params)
        X_out, y_out = sampler.fit_resample(X_train, y_train)
    elif method == "smote_tomek":
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import TomekLinks
        outer = _replace_seed(config["imbalance_methods"][method]["parameters"], seed)
        smote_spec = outer.pop("smote")
        tomek_spec = outer.pop("tomek")
        inner_smote = SMOTE(**_replace_seed(smote_spec["parameters"], seed))
        tomek = TomekLinks(**tomek_spec["parameters"])
        sampler = SMOTETomek(smote=inner_smote, tomek=tomek, **outer)
        X_out, y_out = sampler.fit_resample(X_train, y_train)
    elif method == "ctgan":
        from sdv.metadata import SingleTableMetadata
        from sdv.sampling import Condition
        from sdv.single_table import CTGANSynthesizer

        needed = max(0, before_b - before_m)
        if needed == 0:
            audit["after"] = {"B": before_b, "M": before_m}
            return X_train, y_train, audit

        training = pd.DataFrame(X_train, columns=feature_names)
        training.insert(0, "diagnosis", train_labels.reset_index(drop=True).astype(str))
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=training)
        metadata.update_column(column_name="diagnosis", sdtype="categorical")

        ctgan_params = _replace_seed(config["imbalance_methods"]["ctgan"]["parameters"], seed)
        synthesizer = CTGANSynthesizer(metadata, **ctgan_params)
        synthesizer.fit(training)

        conditional = config["imbalance_methods"]["ctgan"]["conditional_sampling"]
        condition = Condition(num_rows=needed, column_values={"diagnosis": "M"})
        sampled = synthesizer.sample_from_conditions(
            conditions=[condition],
            **conditional["sample_from_conditions"],
        )
        if len(sampled) != needed:
            raise ScientificTaskError(
                f"CTGAN returned {len(sampled)} rows but exact conditional count is {needed}"
            )
        if set(sampled["diagnosis"].astype(str)) != {"M"}:
            raise ScientificTaskError("CTGAN conditional sample contains a non-M diagnosis")

        synthetic_X = sampled.loc[:, feature_names].to_numpy(dtype=float)
        X_out = np.vstack([np.asarray(X_train, dtype=float), synthetic_X])
        y_out = np.concatenate([np.asarray(y_train, dtype=int), np.ones(needed, dtype=int)])
        audit["synthetic_rows"] = int(needed)
        audit["conditional_column_values"] = {"diagnosis": "M"}
    else:
        raise ScientificTaskError(f"unknown frozen imbalance method: {method}")

    after_b = int((np.asarray(y_out) == 0).sum())
    after_m = int((np.asarray(y_out) == 1).sum())
    audit["after"] = {"B": after_b, "M": after_m}
    audit["rows_after"] = int(len(y_out))
    return X_out, y_out, audit


def _compute_metrics(y_true: Any, probabilities: Any, predictions: Any) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    npv = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    return {
        "malignant_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "specificity": specificity,
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "npv": npv,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _execute_scientific_task(
    *,
    task: dict[str, Any],
    dataset_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    run_id: str,
    git_commit: str,
) -> dict[str, Any]:
    """Execute exactly one already-authorized frozen task."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    seed = int(task["seed"])
    _prepare_runtime(seed)
    config = load_experiment_config(config_path)
    frame = load_registered_dataset(dataset_path, EXPECTED_DATASET_SHA256)
    split_artifact = build_outer_folds(
        frame,
        dataset_sha256=EXPECTED_DATASET_SHA256,
        master_seed=20260810,
    )
    fold_matches = [
        fold for fold in split_artifact["folds"]
        if int(fold["repeat_index"]) == int(task["repeat_index"])
        and int(fold["fold_index"]) == int(task["fold_index"])
    ]
    if len(fold_matches) != 1:
        raise ScientificTaskError("task repeat/fold does not resolve exactly one frozen outer fold")
    fold = fold_matches[0]

    by_id = frame.set_index(frame[ID_COLUMN].astype(str), drop=False)
    train = by_id.loc[[str(v) for v in fold["train_ids"]]].copy()
    test = by_id.loc[[str(v) for v in fold["test_ids"]]].copy()
    if set(train[ID_COLUMN].astype(str)) & set(test[ID_COLUMN].astype(str)):
        raise ScientificTaskError("held-out test IDs overlap training IDs")

    feature_names = list(FEATURE_COLUMNS)
    X_train_raw = train.loc[:, feature_names].to_numpy(dtype=float)
    X_test_raw = test.loc[:, feature_names].to_numpy(dtype=float)
    train_labels = train[TARGET_COLUMN].astype(str).reset_index(drop=True)
    test_labels = test[TARGET_COLUMN].astype(str).reset_index(drop=True)
    y_train = (train_labels == "M").astype(int).to_numpy()
    y_test = (test_labels == "M").astype(int).to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    method = str(task["method"])
    classifier_name = str(task["classifier"])
    resample_start = time.perf_counter()
    X_fit, y_fit, resampling_audit = _apply_imbalance_method(
        method,
        X_train,
        y_train,
        train_labels=train_labels,
        feature_names=feature_names,
        config=config,
        seed=seed,
    )
    resampling_seconds = time.perf_counter() - resample_start

    model = _build_classifier(classifier_name, config, seed)
    fit_start = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    probabilities = np.asarray(model.predict_proba(X_test), dtype=float)[:, 1]
    threshold = float(config["primary_design"]["decision_threshold"])
    predictions = (probabilities >= threshold).astype(int)
    predict_seconds = time.perf_counter() - predict_start

    metrics = _compute_metrics(y_test, probabilities, predictions)
    prediction_records = []
    for row_id, y_true, p_m, y_pred in zip(
        test[ID_COLUMN].astype(str).tolist(),
        y_test.tolist(),
        probabilities.tolist(),
        predictions.tolist(),
    ):
        prediction_records.append(
            {
                "run_id": run_id,
                "protocol_version": "1.0",
                "git_commit": git_commit,
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "row_id": row_id,
                "repeat": int(task["repeat_index"]),
                "fold": int(task["fold_index"]),
                "method": method,
                "classifier": classifier_name,
                "replicate": int(task["replicate_index"]),
                "seed": seed,
                "y_true": int(y_true),
                "p_malignant": float(p_m),
                "threshold": threshold,
                "y_pred": int(y_pred),
            }
        )

    return {
        "schema_version": TASK_EVIDENCE_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "result_bearing": True,
        "scientific_execution": 1,
        "run_id": run_id,
        "git_commit": git_commit,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "task": task,
        "split_sha256": split_artifact["split_sha256"],
        "train_row_count": int(len(train)),
        "test_row_count": int(len(test)),
        "train_ids": train[ID_COLUMN].astype(str).tolist(),
        "test_ids": test[ID_COLUMN].astype(str).tolist(),
        "preprocessing": {
            "scaler": "sklearn.preprocessing.StandardScaler",
            "fit_scope": "current_outer_training_fold_only",
            "test_transform": "transform_only_using_training_fitted_scaler",
        },
        "resampling_audit": resampling_audit,
        "metrics": metrics,
        "timings_seconds": {
            "resampling_or_generation": float(resampling_seconds),
            "model_fit": float(fit_seconds),
            "prediction": float(predict_seconds),
        },
        "predictions": prediction_records,
    }


def _write_inprogress_marker(
    checkpoint_root: Path,
    *,
    task: dict[str, Any],
    grant: AuthorizationGrant,
) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    task_id = str(task["task_id"])
    if checkpoint_state(checkpoint_root, task_id) != "PENDING":
        raise ScientificTaskError(f"task {task_id} is not PENDING")
    marker = checkpoint_root / f"{task_id}.inprogress.json"
    payload = {
        "schema_version": TASK_CHECKPOINT_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "state": "INPROGRESS",
        "result_bearing": True,
        "task_id": task_id,
        "seed": int(task["seed"]),
        "git_commit": grant.exact_git_commit,
        "authorization_sha256": grant.authorization_sha256,
        "run_id": grant.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(marker, payload)
    return marker


def _finalize_complete_marker(
    inprogress: Path,
    *,
    task: dict[str, Any],
    grant: AuthorizationGrant,
    evidence_path: Path,
    evidence_sha256: str,
) -> Path:
    complete = inprogress.with_name(inprogress.name.replace(".inprogress.json", ".complete.json"))
    payload = {
        "schema_version": TASK_CHECKPOINT_SCHEMA_VERSION,
        "protocol_version": "1.0",
        "state": "COMPLETE",
        "result_bearing": True,
        "task_id": str(task["task_id"]),
        "seed": int(task["seed"]),
        "git_commit": grant.exact_git_commit,
        "authorization_sha256": grant.authorization_sha256,
        "run_id": grant.run_id,
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_sha256,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(inprogress, payload)
    os.replace(inprogress, complete)
    return complete


def run_authorized_task(
    *,
    authorization_path: str | os.PathLike[str],
    policy_path: str | os.PathLike[str],
    realized_pip_freeze_path: str | os.PathLike[str],
    expected_git_commit: str,
    expected_task_id: str,
    run_root: str | os.PathLike[str],
    plan_path: str | os.PathLike[str],
    dataset_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run one task only after the exact P4B grant passes.

    IMPORTANT: the authorization gate is intentionally first. No run directory,
    model, resampler, generator, prediction, or metric operation occurs before
    it returns successfully.
    """
    grant = validate_authorization(
        authorization_path,
        policy_path=policy_path,
        expected_git_commit=expected_git_commit,
        expected_task_id=expected_task_id,
        expected_run_root=str(run_root),
        realized_pip_freeze_path=realized_pip_freeze_path,
    )

    plan = load_frozen_plan(plan_path)
    task = find_exact_task(plan, expected_task_id)
    paths = required_future_paths(str(run_root))
    root = Path(run_root)
    if root.exists():
        manifest_path = root / "manifests" / "p4b_run_manifest.json"
        if manifest_path.exists():
            raise ScientificTaskError("immutable P4B run root already has a run manifest")
    for name in FUTURE_REQUIRED_SUBDIRECTORIES:
        Path(paths[name]).mkdir(parents=True, exist_ok=True)

    manifest = build_preexecution_manifest(grant, task=task)
    manifest_path, manifest_sha256 = write_preexecution_manifest(root, manifest)

    split_snapshot_path = root / "splits" / f"{expected_task_id}.json"
    if split_snapshot_path.exists():
        raise ScientificTaskError("task split snapshot already exists")

    checkpoint_root = root / "checkpoints"
    if checkpoint_state(checkpoint_root, expected_task_id) != "PENDING":
        raise ScientificTaskError("authorized task is not PENDING")
    inprogress = _write_inprogress_marker(checkpoint_root, task=task, grant=grant)

    try:
        evidence = _execute_scientific_task(
            task=task,
            dataset_path=dataset_path,
            config_path=config_path,
            run_id=grant.run_id,
            git_commit=grant.exact_git_commit,
        )
        split_snapshot = {
            "schema_version": "p4b-task-split/v1",
            "protocol_version": "1.0",
            "task_id": expected_task_id,
            "split_sha256": evidence["split_sha256"],
            "train_ids": evidence["train_ids"],
            "test_ids": evidence["test_ids"],
            "result_bearing": True,
        }
        split_sha = write_json_atomic(split_snapshot_path, split_snapshot)
        write_sha256_receipt(split_snapshot_path, split_sha)

        destination = resolve_task_record_path(task, root)
        if destination.exists() or destination.with_name(destination.name + ".sha256").exists():
            raise ScientificTaskError("task evidence path already exists; overwrite forbidden")
        evidence["authorization_sha256"] = grant.authorization_sha256
        evidence["p4b_policy_sha256"] = grant.p4b_policy_sha256
        evidence["run_manifest_sha256"] = manifest_sha256
        evidence_sha256 = write_json_atomic(destination, evidence)
        write_sha256_receipt(destination, evidence_sha256)
        complete = _finalize_complete_marker(
            inprogress,
            task=task,
            grant=grant,
            evidence_path=destination,
            evidence_sha256=evidence_sha256,
        )
    except Exception:
        raise

    return {
        "P4B_TASK_EXECUTION": "COMPLETE",
        "task_id": expected_task_id,
        "run_id": grant.run_id,
        "evidence_path": str(destination),
        "evidence_sha256": evidence_sha256,
        "checkpoint_path": str(complete),
        "manifest_path": str(manifest_path),
        "result_bearing": True,
        "scientific_execution": 1,
    }
