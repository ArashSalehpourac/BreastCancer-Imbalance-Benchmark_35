from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import seed_for_model, seed_for_training
from src.data import load_wdbc
from src.io_utils import atomic_json, atomic_npz, valid_cache, valid_task
from src.metrics import calculate_metrics
from src.resampling import resample

def fixture_frame() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    y = np.array(["B"] * 357 + ["M"] * 212)
    values = rng.normal(size=(569, 30)) + (y == "M")[:, None]
    frame = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(30)])
    frame.insert(0, "diagnosis", y); frame.insert(0, "id", np.arange(1000, 1569))
    return frame

def test_loader_schema_and_counts(tmp_path):
    path = tmp_path / "wdbc.csv"; fixture_frame().to_csv(path, index=False, lineterminator="\n")
    frame, features = load_wdbc(path, require_canonical_hash=False)
    assert frame.shape == (569, 32); assert len(features) == 30
    assert frame.diagnosis.value_counts().to_dict() == {"B": 357, "M": 212}

def test_seed_domains_are_stable_and_separate():
    assert seed_for_training(0, 0, "smote", 0) == seed_for_training(0, 0, "smote", 0)
    assert seed_for_model(0, 0, "smote", "rf", 0) != seed_for_model(0, 0, "smote", "xgboost", 0)

def test_smote_and_metrics():
    rng = np.random.default_rng(3); X = rng.normal(size=(40, 4)); y = np.array([0] * 30 + [1] * 10)
    Xr, yr = resample("smote", X, y, 44, ["a", "b", "c", "d"], 2)
    assert Xr.shape == (60, 4); assert np.bincount(yr).tolist() == [30, 30]
    metrics = calculate_metrics([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9])
    assert (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]) == (2, 1, 1, 0)

def test_atomic_artifacts_validate(tmp_path):
    cache = tmp_path / "cache.npz"; atomic_npz(cache, X=np.ones((2, 3)), y=np.array([0, 1])); assert valid_cache(cache)
    task = tmp_path / "task.json"; atomic_json(task, {"identity": {}, "metrics": {}, "predictions": [{"x": 1}], "dataset_sha256": "x"})
    assert valid_task(task, 1); assert json.loads(task.read_text())["predictions"][0]["x"] == 1
