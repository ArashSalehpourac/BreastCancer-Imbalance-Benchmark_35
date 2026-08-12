"""One-task scientific executor for a future explicitly authorized P4B run.

The module contains result-bearing operations, but the public entry point is
fail-closed: an exact one-task authorization is validated before any run root
is created or any scientific operation is called. The supported launcher also
starts this module only inside an isolated worker whose deterministic process
environment was set before scientific imports.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import resource
import shutil
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
    sha256_file,
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
    required = {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": str(seed),
    }
    for key, expected in required.items():
        if os.environ.get(key) != expected:
            raise ScientificTaskError(
                f"deterministic worker environment was not established before import: {key}"
            )

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


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Resolve a frozen scientific implementation without weakening P1 scope guards."""
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def _build_classifier(name: str, config: dict[str, Any], seed: int) -> Any:
    params = _replace_seed(dict(config["classifiers"][name]["parameters"]), seed)
    implementations = {
        "rf": ("sklearn.ensemble", "RandomForestClassifier"),
        "adaboost": ("sklearn.ensemble", "AdaBoostClassifier"),
        "xgboost": ("xgboost", "XGBClassifier"),
        "lightgbm": ("lightgbm", "LGBMClassifier"),
    }
    if name not in implementations:
        raise ScientificTaskError(f"unknown frozen classifier: {name}")
    module_name, symbol_name = implementations[name]
    classifier_cls = _load_symbol(module_name, symbol_name)
    return classifier_cls(**params)


def _ctgan_quality_diagnostics(
    real_training: Any,
    synthetic: Any,
    feature_names: list[str],
) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import ks_2samp

    real = np.asarray(real_training, dtype=float)
    synth = np.asarray(synthetic, dtype=float)
    if synth.ndim != 2 or synth.shape[1] != real.shape[1]:
        raise ScientificTaskError("CTGAN synthetic feature matrix shape mismatch")
    if not np.isfinite(synth).all():
        raise ScientificTaskError("CTGAN synthetic features contain NaN/Inf values")

    real_rows = {tuple(float(v) for v in row) for row in real}
    exact_duplicates = sum(tuple(float(v) for v in row) in real_rows for row in synth)

    train_min = real.min(axis=0)
    train_max = real.max(axis=0)
    exceed = (synth < train_min) | (synth > train_max)

    per_feature: dict[str, Any] = {}
    for index, feature in enumerate(feature_names):
        real_col = real[:, index]
        synth_col = synth[:, index]
        real_std = float(real_col.std(ddof=0))
        synth_std = float(synth_col.std(ddof=0))
        ks = ks_2samp(real_col, synth_col, alternative="two-sided", method="auto")
        per_feature[feature] = {
            "real_mean": float(real_col.mean()),
            "synthetic_mean": float(synth_col.mean()),
            "mean_difference": float(synth_col.mean() - real_col.mean()),
            "standardized_mean_difference": (
                float((synth_col.mean() - real_col.mean()) / real_std)
                if real_std > 0
                else None
            ),
            "real_std": real_std,
            "synthetic_std": synth_std,
            "std_ratio": float(synth_std / real_std) if real_std > 0 else None,
            "ks_statistic": float(ks.statistic),
            "range_exceedance_rate": float(exceed[:, index].mean()),
        }

    if len(synth):
        differences = synth[:, None, :] - real[None, :, :]
        nearest = np.sqrt(np.sum(differences * differences, axis=2)).min(axis=1)
        nearest_summary = {
            "min": float(np.min(nearest)),
            "q05": float(np.quantile(nearest, 0.05)),
            "median": float(np.median(nearest)),
            "mean": float(np.mean(nearest)),
            "q95": float(np.quantile(nearest, 0.95)),
            "max": float(np.max(nearest)),
        }
    else:
        nearest_summary = None

    return {
        "exact_duplicate_count_against_real_training": int(exact_duplicates),
        "exact_duplicate_rate_against_real_training": (
            float(exact_duplicates / len(synth)) if len(synth) else 0.0
        ),
        "feature_range_exceedance_rate": float(exceed.mean()) if len(synth) else 0.0,
        "per_feature_distributional_discrepancy": per_feature,
        "nearest_neighbor_distance_to_real_training": nearest_summary,
    }


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
        audit["rows_after"] = int(len(y_train))
        return X_train, y_train, audit

    if method in {"adasyn", "borderline_smote", "smote"}:
        mapping = {
            "adasyn": ("imblearn.over_sampling", "ADASYN"),
            "borderline_smote": ("imblearn.over_sampling", "BorderlineSMOTE"),
            "smote": ("imblearn.over_sampling", "SMOTE"),
        }
        module_name, symbol_name = mapping[method]
        sampler_cls = _load_symbol(module_name, symbol_name)
        params = _replace_seed(config["imbalance_methods"][method]["parameters"], seed)
        sampler = sampler_cls(**params)
        X_out, y_out = getattr(sampler, "fit_resample")(X_train, y_train)
    elif method == "smote_tomek":
        combo_cls = _load_symbol("imblearn.combine", "SMOTETomek")
        smote_cls = _load_symbol("imblearn.over_sampling", "SMOTE")
        tomek_cls = _load_symbol("imblearn.under_sampling", "TomekLinks")
        outer = _replace_seed(config["imbalance_methods"][method]["parameters"], seed)
        smote_spec = outer.pop("smote")
        tomek_spec = outer.pop("tomek")
        inner_smote = smote_cls(**_replace_seed(smote_spec["parameters"], seed))
        tomek = tomek_cls(**tomek_spec["parameters"])
        sampler = combo_cls(smote=inner_smote, tomek=tomek, **outer)
        X_out, y_out = getattr(sampler, "fit_resample")(X_train, y_train)
    elif method == "ctgan":
        from sdv.metadata import SingleTableMetadata
        from sdv.sampling import Condition
        from sdv.single_table import CTGANSynthesizer

        needed = max(0, before_b - before_m)
        audit["requested_synthetic_count"] = int(needed)
        if needed == 0:
            audit["returned_synthetic_count"] = 0
            audit["after"] = {"B": before_b, "M": before_m}
            audit["rows_after"] = int(len(y_train))
            audit["quality_diagnostics"] = None
            return X_train, y_train, audit

        expected_columns = ["diagnosis", *feature_names]
        training = pd.DataFrame(X_train, columns=feature_names)
        training.insert(0, "diagnosis", train_labels.reset_index(drop=True).astype(str))
        if list(training.columns) != expected_columns:
            raise ScientificTaskError("CTGAN training frame column order mismatch")

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
        audit["returned_synthetic_count"] = int(len(sampled))
        if len(sampled) != needed:
            raise ScientificTaskError(
                f"CTGAN returned {len(sampled)} rows but exact conditional count is {needed}"
            )
        if list(sampled.columns) != expected_columns:
            raise ScientificTaskError("CTGAN returned missing/reordered columns")
        if set(sampled["diagnosis"].astype(str)) != {"M"}:
            raise ScientificTaskError("CTGAN conditional sample contains a non-M diagnosis")

        synthetic_X = sampled.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(synthetic_X).all():
            raise ScientificTaskError("CTGAN synthetic data failed finite numeric conversion")
        audit["quality_diagnostics"] = _ctgan_quality_diagnostics(
            X_train,
            synthetic_X,
            feature_names,
        )
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
        "malignant_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "specificity": specificity,
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "benign_precision": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "benign_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
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
        fold
        for fold in split_artifact["folds"]
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
    realized_classifier_parameters = _json_safe(model.get_params(deep=False))

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
        "classifier_parameters_realized": realized_classifier_parameters,
        "metrics": metrics,
        "timings_seconds": {
            "resampling_or_generation": float(resampling_seconds),
            "model_fit": float(fit_seconds),
            "prediction": float(predict_seconds),
        },
        "peak_process_memory": {
            "ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "unit": "KiB_on_Linux",
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


def _copy_with_receipt(source: str | os.PathLike[str], destination: Path) -> str:
    if destination.exists() or destination.with_name(destination.name + ".sha256").exists():
        raise ScientificTaskError(f"immutable provenance destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = sha256_file(destination)
    write_sha256_receipt(destination, digest)
    return digest


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
    """Run exactly one task after the exact grant passes.

    Authorization is intentionally the first gate. The immutable first-run root
    must not exist before the grant, and no model/resampler/generator operation
    is reachable until after the grant and frozen task have been validated.
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
    root = Path(run_root)
    if root.exists():
        raise ScientificTaskError("immutable P4B first-run root already exists; reuse is forbidden")

    config = load_experiment_config(config_path)
    paths = required_future_paths(str(run_root))
    for name in FUTURE_REQUIRED_SUBDIRECTORIES:
        Path(paths[name]).mkdir(parents=True, exist_ok=False)

    # Preserve exact authorization/environment/plan inputs before scientific work.
    auth_copy = root / "receipts" / "p4b_authorization.json"
    auth_copy_sha = _copy_with_receipt(authorization_path, auth_copy)
    if auth_copy_sha != grant.authorization_sha256:
        raise ScientificTaskError("copied authorization receipt changed bytes")
    freeze_copy = root / "preflight" / "pip_freeze.txt"
    freeze_copy_sha = _copy_with_receipt(realized_pip_freeze_path, freeze_copy)
    if freeze_copy_sha != grant.pip_freeze_sha256:
        raise ScientificTaskError("copied realized environment receipt changed bytes")
    plan_copy = root / "preflight" / "p3_execution_plan.json"
    _copy_with_receipt(plan_path, plan_copy)
    policy_copy = root / "receipts" / "P4B_EXECUTION_POLICY_v1.json"
    policy_copy_sha = _copy_with_receipt(policy_path, policy_copy)
    if policy_copy_sha != grant.p4b_policy_sha256:
        raise ScientificTaskError("copied P4B policy changed bytes")

    manifest = build_preexecution_manifest(grant, task=task)
    manifest.update(
        {
            "software_versions": config["software"]["direct_pins"],
            "classifier_parameters": config["classifiers"],
            "resampler_parameters": config["imbalance_methods"],
            "ctgan_parameters": config["imbalance_methods"]["ctgan"],
            "validation_gates": {
                "authorization": "PASS",
                "exact_git_commit": "PASS",
                "frozen_task": "PASS",
                "unused_run_root": "PASS",
                "accepted_environment_receipt": "PASS",
            },
            "utc_end": None,
        }
    )
    manifest_path, manifest_sha256 = write_preexecution_manifest(root, manifest)

    split_snapshot_path = root / "splits" / f"{expected_task_id}.json"
    checkpoint_root = root / "checkpoints"
    if checkpoint_state(checkpoint_root, expected_task_id) != "PENDING":
        raise ScientificTaskError("authorized task is not PENDING")
    inprogress = _write_inprogress_marker(checkpoint_root, task=task, grant=grant)

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
