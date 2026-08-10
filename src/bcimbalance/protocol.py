"""Frozen Protocol v1.0 constants used by non-result-bearing foundation code."""

PROTOCOL_VERSION = "1.0"
MASTER_SEED = 20260810
N_SPLITS = 10
N_REPEATS = 5
N_REPLICATES = 3

METHOD_CODES = {
    "baseline": 0,
    "adasyn": 1,
    "borderline_smote": 2,
    "smote": 3,
    "smote_tomek": 4,
    "ctgan": 5,
}

MODEL_CODES = {
    "rf": 0,
    "adaboost": 1,
    "xgboost": 2,
    "lightgbm": 3,
}
