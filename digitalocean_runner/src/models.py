"""Classifier factory with fixed, untuned parameters."""
from __future__ import annotations

def make_model(name: str, seed: int):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, criterion="gini", max_features="sqrt", bootstrap=True, class_weight=None, n_jobs=1, random_state=seed)
    if name == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1, objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cpu", reg_alpha=0, reg_lambda=1, n_jobs=1, random_state=seed, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(boosting_type="gbdt", objective="binary", num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, min_child_samples=20, min_child_weight=0.001, min_split_gain=0, subsample=1, subsample_freq=0, colsample_bytree=1, reg_alpha=0, reg_lambda=0, class_weight=None, n_jobs=1, deterministic=True, force_col_wise=True, device_type="cpu", random_state=seed, verbosity=-1)
    raise ValueError(f"unknown classifier: {name}")
