"""Deterministic seed derivation and collision-audited registry generation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import (
    MASTER_SEED,
    METHOD_CODES,
    MODEL_CODES,
    N_REPEATS,
    N_REPLICATES,
    N_SPLITS,
    PROTOCOL_VERSION,
)

SEED_SCHEMA_VERSION = "condition-seeds/v1"


class SeedRegistryError(ValueError):
    """Raised when the deterministic seed registry violates the frozen key space."""


def derive_condition_seed(
    repeat_index: int,
    fold_index: int,
    method: str,
    model: str,
    replicate_index: int,
    *,
    master_seed: int = MASTER_SEED,
) -> int:
    if method not in METHOD_CODES:
        raise SeedRegistryError(f"unknown method: {method}")
    if model not in MODEL_CODES:
        raise SeedRegistryError(f"unknown model: {model}")
    if not 0 <= repeat_index < N_REPEATS:
        raise SeedRegistryError(f"repeat_index out of range: {repeat_index}")
    if not 0 <= fold_index < N_SPLITS:
        raise SeedRegistryError(f"fold_index out of range: {fold_index}")
    if not 0 <= replicate_index < N_REPLICATES:
        raise SeedRegistryError(f"replicate_index out of range: {replicate_index}")

    sequence = np.random.SeedSequence(
        [
            int(master_seed),
            int(repeat_index),
            int(fold_index),
            int(METHOD_CODES[method]),
            int(MODEL_CODES[model]),
            int(replicate_index),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_seed_registry(*, master_seed: int = MASTER_SEED) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seeds_seen: dict[int, tuple[int, int, str, str, int]] = {}

    for repeat_index in range(N_REPEATS):
        for fold_index in range(N_SPLITS):
            for method in METHOD_CODES:
                for model in MODEL_CODES:
                    for replicate_index in range(N_REPLICATES):
                        seed = derive_condition_seed(
                            repeat_index,
                            fold_index,
                            method,
                            model,
                            replicate_index,
                            master_seed=master_seed,
                        )
                        key = (repeat_index, fold_index, method, model, replicate_index)
                        if seed in seeds_seen:
                            raise SeedRegistryError(
                                f"seed collision: seed={seed}, key={key}, previous={seeds_seen[seed]}"
                            )
                        seeds_seen[seed] = key
                        records.append(
                            {
                                "repeat_index": repeat_index,
                                "fold_index": fold_index,
                                "method": method,
                                "method_code": METHOD_CODES[method],
                                "model": model,
                                "model_code": MODEL_CODES[model],
                                "replicate_index": replicate_index,
                                "seed": seed,
                            }
                        )

    payload: dict[str, Any] = {
        "schema_version": SEED_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "master_seed": int(master_seed),
        "n_records": len(records),
        "records": records,
    }
    payload["registry_sha256"] = _canonical_hash(payload)
    return payload


def verify_seed_registry(payload: dict[str, Any]) -> None:
    if int(payload.get("master_seed", -1)) != MASTER_SEED:
        raise SeedRegistryError("seed registry master seed does not match frozen protocol")
    expected_count = N_REPEATS * N_SPLITS * len(METHOD_CODES) * len(MODEL_CODES) * N_REPLICATES
    if int(payload.get("n_records", -1)) != expected_count:
        raise SeedRegistryError("seed registry record count is incomplete")

    records = payload.get("records", [])
    if len(records) != expected_count:
        raise SeedRegistryError("seed registry records list is incomplete")

    observed_seeds: set[int] = set()
    observed_keys: set[tuple[int, int, str, str, int]] = set()
    for record in records:
        key = (
            int(record["repeat_index"]),
            int(record["fold_index"]),
            str(record["method"]),
            str(record["model"]),
            int(record["replicate_index"]),
        )
        expected_seed = derive_condition_seed(*key)
        if int(record["seed"]) != expected_seed:
            raise SeedRegistryError(f"seed mismatch for key={key}")
        if key in observed_keys:
            raise SeedRegistryError(f"duplicate seed key={key}")
        if expected_seed in observed_seeds:
            raise SeedRegistryError(f"duplicate seed value={expected_seed}")
        observed_keys.add(key)
        observed_seeds.add(expected_seed)

    stored = str(payload.get("registry_sha256", ""))
    copy = dict(payload)
    copy.pop("registry_sha256", None)
    computed = _canonical_hash(copy)
    if stored != computed:
        raise SeedRegistryError(f"registry hash mismatch: stored={stored} computed={computed}")


def write_seed_registry(path: str | os.PathLike[str], payload: dict[str, Any]) -> str:
    verify_seed_registry(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as tmp:
        tmp.write(encoded)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, destination)
    return str(payload["registry_sha256"])
