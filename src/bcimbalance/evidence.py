"""Atomic machine-readable evidence writers and provenance manifests."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .protocol import MASTER_SEED, PROTOCOL_VERSION

EVIDENCE_SCHEMA_VERSION = "evidence/v1"
RUN_MANIFEST_SCHEMA_VERSION = "run-manifest/v1"


class EvidenceError(ValueError):
    """Raised when evidence cannot be written or validated safely."""


@dataclass(frozen=True)
class EvidenceReceipt:
    path: str
    sha256: str
    bytes: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _atomic_write(path: Path, payload: bytes) -> EvidenceReceipt:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    digest = _sha256_bytes(payload)

    sidecar = path.with_name(path.name + ".sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(sidecar_payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        sidecar_tmp = Path(tmp.name)
    os.replace(sidecar_tmp, sidecar)
    return EvidenceReceipt(path=str(path), sha256=digest, bytes=len(payload))


def write_json(path: str | os.PathLike[str], payload: Any) -> EvidenceReceipt:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    return _atomic_write(Path(path), encoded)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> EvidenceReceipt:
    buffer = io.StringIO()
    for row in rows:
        buffer.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        buffer.write("\n")
    return _atomic_write(Path(path), buffer.getvalue().encode("utf-8"))


def write_csv_rows(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> EvidenceReceipt:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return _atomic_write(Path(path), buffer.getvalue().encode("utf-8"))


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("numpy", "pandas", "scikit-learn"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def build_foundation_manifest(
    *,
    run_id: str,
    git_commit: str,
    dataset_sha256: str,
    split_sha256: str | None,
    seed_registry_sha256: str | None,
    phase: str = "P1-foundation",
) -> dict[str, Any]:
    if not run_id.strip():
        raise EvidenceError("run_id must be non-empty")
    if len(git_commit.strip()) < 7:
        raise EvidenceError("git_commit must be a commit identifier")
    if not _is_sha256(dataset_sha256):
        raise EvidenceError("dataset_sha256 must be a 64-character hexadecimal SHA-256")
    for name, value in (("split_sha256", split_sha256), ("seed_registry_sha256", seed_registry_sha256)):
        if value is not None and not _is_sha256(value):
            raise EvidenceError(f"{name} must be a 64-character hexadecimal SHA-256 when present")

    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dataset_sha256": dataset_sha256.lower(),
        "split_sha256": split_sha256.lower() if split_sha256 else None,
        "seed_registry_sha256": seed_registry_sha256.lower() if seed_registry_sha256 else None,
        "master_seed": MASTER_SEED,
        "result_bearing": False,
        "environment": environment_snapshot(),
    }


def validate_foundation_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "evidence_schema_version",
        "protocol_version",
        "phase",
        "run_id",
        "created_utc",
        "git_commit",
        "dataset_sha256",
        "master_seed",
        "result_bearing",
        "environment",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise EvidenceError(f"manifest missing fields: {missing}")
    if manifest["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION:
        raise EvidenceError("manifest schema version mismatch")
    if manifest["evidence_schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceError("manifest evidence schema version mismatch")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise EvidenceError("manifest protocol version mismatch")
    if str(manifest["phase"]) != "P1-foundation":
        raise EvidenceError("P1 foundation manifest phase mismatch")
    if int(manifest["master_seed"]) != MASTER_SEED:
        raise EvidenceError("manifest master seed mismatch")
    if bool(manifest["result_bearing"]):
        raise EvidenceError("P1 foundation manifests must never be result-bearing")
    if not _is_sha256(manifest["dataset_sha256"]):
        raise EvidenceError("manifest dataset SHA-256 is invalid")
    for name in ("split_sha256", "seed_registry_sha256"):
        value = manifest.get(name)
        if value is not None and not _is_sha256(value):
            raise EvidenceError(f"manifest {name} is invalid")


def write_foundation_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> EvidenceReceipt:
    validate_foundation_manifest(manifest)
    return write_json(path, dict(manifest))
