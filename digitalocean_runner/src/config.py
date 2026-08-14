"""Frozen scientific defaults and deterministic seed helpers."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

MASTER_SEED = 20260810
DATASET_SHA256 = "27f219231dbb30eecbfc1361407ed641ea01be43316e2c707a1baf82c9795e23"
METHOD_CODES = {"baseline": 0, "adasyn": 1, "borderline_smote": 2, "smote": 3, "smote_tomek": 4, "ctgan": 5}
MODEL_CODES = {"rf": 0, "adaboost": 1, "xgboost": 2, "lightgbm": 3}

@dataclass(frozen=True)
class Design:
    repeats: int = 5
    splits: int = 10
    replicates: int = 3
    ctgan_epochs: int = 400

def seed_for_training(repeat: int, fold: int, method: str, replicate: int) -> int:
    return int(np.random.SeedSequence([MASTER_SEED, repeat, fold, METHOD_CODES[method], replicate]).generate_state(1, dtype=np.uint32)[0])

def seed_for_model(repeat: int, fold: int, method: str, model: str, replicate: int) -> int:
    return int(np.random.SeedSequence([MASTER_SEED, repeat, fold, METHOD_CODES[method], MODEL_CODES[model], replicate]).generate_state(1, dtype=np.uint32)[0])

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

def constrain_cpu() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
