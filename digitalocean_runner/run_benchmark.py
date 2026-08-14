#!/usr/bin/env python3
"""Run the clean, leakage-safe, restartable WDBC benchmark."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (DATASET_SHA256, Design, METHOD_CODES, MODEL_CODES,
                        seed_everything, seed_for_model, seed_for_training)
from src.data import load_wdbc
from src.io_utils import atomic_json, atomic_npz, ensure_output, valid_cache, valid_task
from src.metrics import calculate_metrics
from src.models import make_model
from src.resampling import resample

def parse_names(value: str, allowed: dict[str, int], label: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    bad = set(names) - set(allowed)
    if bad or not names: raise argparse.ArgumentTypeError(f"invalid {label}: {sorted(bad)}")
    return names

def task_name(repeat, fold, method, model, replicate) -> str:
    return f"r{repeat:02d}_f{fold:02d}_{method}_{model}_rep{replicate:02d}.json"

def cache_name(repeat, fold, method, replicate) -> str:
    return f"r{repeat:02d}_f{fold:02d}_{method}_rep{replicate:02d}.npz"

def audit_name(repeat, fold, method, replicate) -> str:
    return f"r{repeat:02d}_f{fold:02d}_{method}_rep{replicate:02d}.json"

def selected_identities(design: Design, methods: list[str], models: list[str]):
    for repeat in range(design.repeats):
        for fold in range(design.splits):
            for method in methods:
                for replicate in range(design.replicates):
                    for model in models:
                        yield repeat, fold, method, model, replicate

def print_status(output: Path, total: int = 3600) -> None:
    ensure_output(output)
    completed = sum(valid_task(path) for path in (output / "tasks").glob("*.json"))
    caches = sum(valid_cache(path) for path in (output / "training_cache").glob("*.npz"))
    errors = len(list((output / "errors").glob("*.json")))
    print(f"completed={completed}")
    print(f"pending={max(0, total - completed)}")
    print(f"cache count={caches}")
    print(f"error count={errors}")

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset")
    p.add_argument("--output", default="results")
    p.add_argument("--resume", action="store_true", help="Resume safely; valid artifacts are always skipped")
    p.add_argument("--status", action="store_true", help="Report artifact counts without running")
    p.add_argument("--max-tasks", type=int)
    p.add_argument("--methods", default=",".join(METHOD_CODES))
    p.add_argument("--models", default=",".join(MODEL_CODES))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--ctgan-epochs", type=int, default=400)
    return p

def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if args.status:
        print_status(output); return 0
    if not args.dataset: raise SystemExit("--dataset is required unless --status is used")
    if min(args.repeats, args.splits, args.replicates, args.ctgan_epochs) < 1:
        raise SystemExit("repeats, splits, replicates, and ctgan epochs must be positive")
    if args.max_tasks is not None and args.max_tasks < 0: raise SystemExit("--max-tasks must be nonnegative")
    try:
        methods = parse_names(args.methods, METHOD_CODES, "methods")
        models = parse_names(args.models, MODEL_CODES, "models")
    except argparse.ArgumentTypeError as exc: raise SystemExit(str(exc)) from exc
    design = Design(args.repeats, args.splits, args.replicates, args.ctgan_epochs)
    ensure_output(output)
    frame, feature_names = load_wdbc(args.dataset)
    X = frame[feature_names].to_numpy(dtype=float)
    y = frame["diagnosis"].map({"B": 0, "M": 1}).to_numpy(dtype=int)
    row_ids = frame["id"].astype(str).to_numpy()
    config = {"dataset_sha256": DATASET_SHA256, "repeats": design.repeats, "splits": design.splits,
              "replicates": design.replicates, "ctgan_epochs": design.ctgan_epochs,
              "methods": methods, "models": models, "threshold": 0.5}
    config_path = output / "run_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        structural = ("dataset_sha256", "repeats", "splits", "replicates", "ctgan_epochs")
        if any(prior.get(k) != config[k] for k in structural):
            raise RuntimeError("output directory belongs to an incompatible run design; choose another --output")
    else: atomic_json(config_path, config)
    splits = list(RepeatedStratifiedKFold(n_splits=design.splits, n_repeats=design.repeats,
                                          random_state=20260810).split(X, y))
    identities = list(selected_identities(design, methods, models))
    total = len(identities)
    completed = sum(valid_task(output / "tasks" / task_name(*identity)) for identity in identities)
    started = time.monotonic(); executed = 0
    for repeat, fold, method, model_name, replicate in identities:
        task_path = output / "tasks" / task_name(repeat, fold, method, model_name, replicate)
        split_index = repeat * design.splits + fold
        train_idx, test_idx = splits[split_index]
        if valid_task(task_path, len(test_idx)):
            continue
        if args.max_tasks is not None and executed >= args.max_tasks: break
        cache_path = output / "training_cache" / cache_name(repeat, fold, method, replicate)
        try:
            scaler = StandardScaler().fit(X[train_idx])
            X_train = scaler.transform(X[train_idx]); X_test = scaler.transform(X[test_idx])
            training_seed = seed_for_training(repeat, fold, method, replicate)
            if valid_cache(cache_path):
                with np.load(cache_path, allow_pickle=False) as cached:
                    X_realized, y_realized = cached["X"], cached["y"]
            else:
                if method == "ctgan":
                    print(f"CTGAN TRAIN START repeat={repeat} fold={fold} replicate={replicate}", flush=True)
                before_b, before_m = int((y[train_idx] == 0).sum()), int((y[train_idx] == 1).sum())
                X_realized, y_realized = resample(method, X_train, y[train_idx], training_seed,
                                                   feature_names, design.ctgan_epochs)
                if method == "ctgan":
                    print(f"CTGAN TRAIN COMPLETE repeat={repeat} fold={fold} replicate={replicate}", flush=True)
                atomic_npz(cache_path, X=np.asarray(X_realized, dtype=float), y=np.asarray(y_realized, dtype=np.int8))
                atomic_json(output / "resampling_audits" / audit_name(repeat, fold, method, replicate), {
                    "repeat": repeat, "fold": fold, "method": method, "replicate": replicate,
                    "seed": training_seed, "train_rows_before": len(train_idx), "train_rows_after": len(y_realized),
                    "benign_before": before_b, "malignant_before": before_m,
                    "benign_after": int((y_realized == 0).sum()), "malignant_after": int((y_realized == 1).sum())})
            model_seed = seed_for_model(repeat, fold, method, model_name, replicate)
            seed_everything(model_seed)
            model = make_model(model_name, model_seed); model.fit(X_realized, y_realized)
            probabilities = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
            predictions = (probabilities >= 0.5).astype(int)
            metrics = calculate_metrics(y[test_idx], probabilities)
            records = [{"row_id": str(rid), "repeat": repeat, "fold": fold, "method": method,
                        "classifier": model_name, "replicate": replicate, "y_true": int(truth),
                        "p_malignant": float(prob), "y_pred": int(pred)}
                       for rid, truth, prob, pred in zip(row_ids[test_idx], y[test_idx], probabilities, predictions)]
            atomic_json(task_path, {"identity": {"repeat": repeat, "fold": fold, "method": method,
                        "classifier": model_name, "replicate": replicate}, "dataset_sha256": DATASET_SHA256,
                        "training_seed": training_seed, "classifier_seed": model_seed,
                        "metrics": metrics, "predictions": records})
            error_path = output / "errors" / task_path.name
            if error_path.exists(): error_path.unlink()
            completed += 1; executed += 1
            print(f"completed={completed}/{total} elapsed={time.monotonic()-started:.1f}s repeat={repeat} fold={fold} method={method} classifier={model_name} replicate={replicate}", flush=True)
        except Exception as exc:
            atomic_json(output / "errors" / task_path.name, {"identity": [repeat, fold, method, model_name, replicate],
                        "error": repr(exc), "traceback": traceback.format_exc()})
            raise
    print(f"run stopped: completed={completed}/{total} newly_completed={executed}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
