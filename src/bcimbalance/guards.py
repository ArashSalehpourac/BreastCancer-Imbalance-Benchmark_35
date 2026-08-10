"""Leakage guards and typed train/test partitions for future fold-local pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from .dataset import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN


class LeakageError(ValueError):
    """Raised when held-out and training rows are mixed or incompletely partitioned."""


@dataclass(frozen=True)
class TrainingPartition:
    ids: tuple[str, ...]
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class TestingPartition:
    ids: tuple[str, ...]
    X: pd.DataFrame
    y: pd.Series


def assert_train_test_disjoint(train_ids: Iterable[str], test_ids: Iterable[str]) -> None:
    train = tuple(str(v) for v in train_ids)
    test = tuple(str(v) for v in test_ids)
    if len(train) != len(set(train)):
        raise LeakageError("training partition contains duplicate row IDs")
    if len(test) != len(set(test)):
        raise LeakageError("test partition contains duplicate row IDs")
    overlap = set(train) & set(test)
    if overlap:
        sample = sorted(overlap)[:10]
        raise LeakageError(f"training/test partitions overlap: {sample}")


def partition_fold(
    frame: pd.DataFrame,
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
) -> tuple[TrainingPartition, TestingPartition]:
    """Create isolated training and testing views from a validated canonical dataframe."""
    assert_train_test_disjoint(train_ids, test_ids)

    lookup = frame.copy(deep=False)
    lookup_ids = lookup[ID_COLUMN].astype(str)
    if lookup_ids.duplicated().any():
        raise LeakageError("source dataframe row IDs are not unique")

    all_ids = set(lookup_ids)
    requested = set(str(v) for v in train_ids) | set(str(v) for v in test_ids)
    unknown = sorted(requested - all_ids)
    if unknown:
        raise LeakageError(f"fold references unknown row IDs: {unknown[:10]}")
    if requested != all_ids:
        missing = sorted(all_ids - requested)
        raise LeakageError(f"fold does not partition every dataset row: {missing[:10]}")

    indexed = lookup.assign(__row_id=lookup_ids).set_index("__row_id", drop=False)
    train_key = [str(v) for v in train_ids]
    test_key = [str(v) for v in test_ids]

    train_frame = indexed.loc[train_key]
    test_frame = indexed.loc[test_key]

    training = TrainingPartition(
        ids=tuple(train_key),
        X=train_frame.loc[:, list(FEATURE_COLUMNS)].copy(deep=True),
        y=train_frame[TARGET_COLUMN].copy(deep=True),
    )
    testing = TestingPartition(
        ids=tuple(test_key),
        X=test_frame.loc[:, list(FEATURE_COLUMNS)].copy(deep=True),
        y=test_frame[TARGET_COLUMN].copy(deep=True),
    )
    return training, testing


def fit_transform_training_only(transformer, training: TrainingPartition, testing: TestingPartition):
    """Fit a preprocessing object only on training X, then transform both partitions."""
    assert_train_test_disjoint(training.ids, testing.ids)
    fitted = transformer.fit(training.X)
    X_train = fitted.transform(training.X)
    X_test = fitted.transform(testing.X)
    return fitted, X_train, X_test
