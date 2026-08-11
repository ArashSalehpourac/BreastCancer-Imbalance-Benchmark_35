"""Fail-closed validation for the P2 experiment configuration lock.

This module validates configuration only. It must not fit models, execute
resamplers, sample CTGAN, predict, calculate metrics, or perform statistics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CONFIG_SHA256 = "71e82edad62dcf06382cf85dcd87e642049e80668eaa7dbd9913d7a5a5bb7dc9"
EXPECTED_BASE = "bb37920a134d38edc72c6b7ba2ce97d7c93b96d2"
EXPECTED_DATASET = "27f219231dbb30eecbfc1361407ed641ea01be43316e2c707a1baf82c9795e23"
EXPECTED_SPLIT = "00114e7735fc4eb012ecf248010b18b441cce4a74e0d1540115bbaff543d764a"
EXPECTED_SEEDS = "985f5614275ef880213bf775cded0b8e3867fb011f43df817b48d94bb5af73e2"
EXPECTED_FOUNDATION_LOCK = "49291df347b8aa8453a68cf06296642f3939bfc3fe51ce452d3620d9c228e030"

EXPECTED_METHODS = [
    "baseline", "adasyn", "borderline_smote", "smote", "smote_tomek", "ctgan"
]
EXPECTED_CLASSIFIERS = ["rf", "adaboost", "xgboost", "lightgbm"]
EXPECTED_PINS = {
    "numpy": "2.0.2",
    "pandas": "2.2.2",
    "scikit-learn": "1.9.0",
    "imbalanced-learn": "0.14.2",
    "xgboost": "3.3.0",
    "lightgbm": "4.6.0",
    "sdv": "1.36.3",
    "ctgan": "0.12.1",
}


class ExperimentConfigError(ValueError):
    """Raised when P2 configuration differs from the frozen candidate."""


def canonical_config_sha256(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("config_sha256", None)
    encoded = json.dumps(
        copy, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_experiment_config(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "experiment-config/v1":
        raise ExperimentConfigError("schema version mismatch")
    if payload.get("protocol_version") != "1.0":
        raise ExperimentConfigError("protocol version mismatch")
    if payload.get("config_version") != "1.0":
        raise ExperimentConfigError("config version mismatch")
    if payload.get("status") != "FROZEN_CANDIDATE_P2":
        raise ExperimentConfigError("unexpected P2 status")
    if payload.get("result_bearing") is not False:
        raise ExperimentConfigError("P2 must remain non-result-bearing")
    if payload.get("authorized_base_commit") != EXPECTED_BASE:
        raise ExperimentConfigError("authorized base commit mismatch")

    stored_hash = str(payload.get("config_sha256", ""))
    computed_hash = canonical_config_sha256(payload)
    if stored_hash != EXPECTED_CONFIG_SHA256 or computed_hash != stored_hash:
        raise ExperimentConfigError(
            f"configuration hash mismatch: stored={stored_hash} computed={computed_hash}"
        )

    foundation = payload.get("foundation_lock", {})
    expected_foundation = {
        "sha256": EXPECTED_FOUNDATION_LOCK,
        "dataset_sha256": EXPECTED_DATASET,
        "split_sha256": EXPECTED_SPLIT,
        "seed_registry_sha256": EXPECTED_SEEDS,
        "master_seed": 20260810,
    }
    for key, value in expected_foundation.items():
        if foundation.get(key) != value:
            raise ExperimentConfigError(f"foundation identity mismatch: {key}")

    design = payload.get("primary_design", {})
    if design.get("methods") != EXPECTED_METHODS:
        raise ExperimentConfigError("method order/set mismatch")
    if design.get("classifiers") != EXPECTED_CLASSIFIERS:
        raise ExperimentConfigError("classifier order/set mismatch")
    if design.get("condition_count") != 24:
        raise ExperimentConfigError("primary condition count must be 24")
    if design.get("replicates") != 3:
        raise ExperimentConfigError("replicate count must be 3")
    if design.get("decision_threshold") != 0.5:
        raise ExperimentConfigError("decision threshold must remain 0.50")
    if design.get("positive_class") != "M":
        raise ExperimentConfigError("malignant class must remain positive")
    if design.get("primary_metric") != "malignant_recall":
        raise ExperimentConfigError("primary endpoint mismatch")
    if design.get("class_weighting") != "forbidden":
        raise ExperimentConfigError("class weighting must remain forbidden")
    if design.get("preprocessing", {}).get("fit_scope") != "current_outer_training_fold_only":
        raise ExperimentConfigError("preprocessing leakage guard mismatch")

    classifiers = payload.get("classifiers", {})
    if sorted(classifiers) != sorted(EXPECTED_CLASSIFIERS):
        raise ExperimentConfigError("exact four classifiers are required")
    for name, specification in classifiers.items():
        params = specification.get("parameters", {})
        if params.get("class_weight", None) is not None:
            raise ExperimentConfigError(f"class weighting detected for {name}")
        if "random_state" in params and params["random_state"] != "$SEED":
            raise ExperimentConfigError(f"{name} random_state must use condition seed")

    methods = payload.get("imbalance_methods", {})
    if sorted(methods) != sorted(EXPECTED_METHODS):
        raise ExperimentConfigError("exact six imbalance methods are required")
    if any(spec.get("training_only") is not True for spec in methods.values()):
        raise ExperimentConfigError("every imbalance method must be training-only")

    for name in ("adasyn", "borderline_smote", "smote"):
        if methods[name]["parameters"].get("sampling_strategy") != 1.0:
            raise ExperimentConfigError(f"{name} must target a 1:1 ratio")

    tomek = methods["smote_tomek"]["parameters"]["tomek"]
    if tomek.get("class") != "imblearn.under_sampling.TomekLinks":
        raise ExperimentConfigError("TomekLinks implementation mapping changed")
    if tomek.get("parameters", {}).get("sampling_strategy") != "all":
        raise ExperimentConfigError("TomekLinks must remove both members of links")

    ctgan = methods["ctgan"]
    cp = ctgan.get("parameters", {})
    for key, value in {"epochs": 400, "batch_size": 50, "pac": 10}.items():
        if cp.get(key) != value:
            raise ExperimentConfigError(f"CTGAN {key} mismatch")
    if cp.get("enable_gpu") is not False:
        raise ExperimentConfigError("primary CTGAN execution is CPU-only")
    conditional = ctgan.get("conditional_sampling", {})
    if conditional.get("column_values") != {"diagnosis": "M"}:
        raise ExperimentConfigError("CTGAN conditional target mismatch")
    if conditional.get("forbid_unconditional_sample_then_label_overwrite") is not True:
        raise ExperimentConfigError("forbidden historical CTGAN pattern not blocked")

    pins = payload.get("software", {}).get("direct_pins", {})
    if pins != EXPECTED_PINS:
        raise ExperimentConfigError("direct scientific package pins changed")

    deterministic = payload.get("determinism", {})
    if deterministic.get("seed_source") != "condition seed from frozen P1 seed registry":
        raise ExperimentConfigError("seed source mismatch")
    if deterministic.get("torch_policy", {}).get("cuda_allowed") is not False:
        raise ExperimentConfigError("CUDA must remain disabled for primary CTGAN")


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_experiment_config(payload)
    return payload
