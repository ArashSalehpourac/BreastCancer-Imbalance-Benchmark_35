from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bcimbalance.dataset import (
    DatasetValidationError,
    FEATURE_COLUMNS,
    load_registered_dataset,
    register_dataset,
    sha256_file,
    validate_wdbc_frame,
)
from bcimbalance.evidence import (
    build_foundation_manifest,
    validate_foundation_manifest,
    write_foundation_manifest,
    write_json,
)
from bcimbalance.folds import build_outer_folds, load_split_artifact, verify_split_artifact, write_split_artifact
from bcimbalance.guards import LeakageError, assert_train_test_disjoint, fit_transform_training_only, partition_fold
from bcimbalance.seeds import generate_seed_registry, verify_seed_registry, write_seed_registry


def make_valid_frame(n_b: int = 60, n_m: int = 40) -> pd.DataFrame:
    n = n_b + n_m
    data: dict[str, object] = {
        "id": [100000 + i for i in range(n)],
        "diagnosis": ["B"] * n_b + ["M"] * n_m,
    }
    base = np.linspace(1.0, 10.0, n)
    for index, column in enumerate(FEATURE_COLUMNS):
        data[column] = base + index * 0.01
    frame = pd.DataFrame(data)
    frame["Unnamed: 32"] = np.nan
    return frame


class DatasetFoundationTests(unittest.TestCase):
    def test_valid_schema_is_canonicalized(self):
        validated = validate_wdbc_frame(make_valid_frame())
        self.assertEqual(len(validated), 100)
        self.assertNotIn("Unnamed: 32", validated.columns)
        self.assertEqual(set(validated["diagnosis"]), {"B", "M"})
        self.assertEqual(len(FEATURE_COLUMNS), 30)

    def test_schema_gates_reject_invalid_inputs(self):
        duplicate = make_valid_frame()
        duplicate.loc[1, "id"] = duplicate.loc[0, "id"]
        with self.assertRaises(DatasetValidationError):
            validate_wdbc_frame(duplicate)

        bad_label = make_valid_frame()
        bad_label.loc[0, "diagnosis"] = "X"
        with self.assertRaises(DatasetValidationError):
            validate_wdbc_frame(bad_label)

        nonfinite = make_valid_frame()
        nonfinite.loc[0, FEATURE_COLUMNS[0]] = np.inf
        with self.assertRaises(DatasetValidationError):
            validate_wdbc_frame(nonfinite)

        nonempty_export = make_valid_frame()
        nonempty_export.loc[0, "Unnamed: 32"] = 1
        with self.assertRaises(DatasetValidationError):
            validate_wdbc_frame(nonempty_export)

    def test_registration_hash_and_copy_are_verified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.csv"
            registry = root / "registry.json"
            canonical = root / "raw" / "wdbc.csv"
            make_valid_frame().to_csv(source, index=False)
            expected = sha256_file(source)

            manifest = register_dataset(
                source,
                registry,
                expected_sha256=expected,
                canonical_copy=canonical,
            )
            self.assertEqual(manifest["sha256"], expected)
            self.assertEqual(sha256_file(canonical), expected)
            self.assertEqual(len(load_registered_dataset(canonical, expected)), 100)

            with canonical.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(DatasetValidationError):
                load_registered_dataset(canonical, expected)


class SplitFoundationTests(unittest.TestCase):
    def test_frozen_5x10_split_is_deterministic_and_valid(self):
        frame = validate_wdbc_frame(make_valid_frame())
        dataset_hash = "a" * 64
        first = build_outer_folds(frame, dataset_sha256=dataset_hash)
        second = build_outer_folds(frame, dataset_sha256=dataset_hash)
        self.assertEqual(first["split_sha256"], second["split_sha256"])
        self.assertEqual(len(first["folds"]), 50)
        verify_split_artifact(first)

        for repeat in range(5):
            counts = {row_id: 0 for row_id in first["row_ids"]}
            for fold in first["folds"]:
                if fold["repeat_index"] == repeat:
                    for row_id in fold["test_ids"]:
                        counts[row_id] += 1
            self.assertTrue(all(value == 1 for value in counts.values()))

    def test_split_artifact_round_trip_and_hash_guard(self):
        frame = validate_wdbc_frame(make_valid_frame())
        artifact = build_outer_folds(frame, dataset_sha256="b" * 64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "outer_folds.json"
            digest = write_split_artifact(path, artifact)
            loaded = load_split_artifact(path)
            self.assertEqual(digest, loaded["split_sha256"])

            loaded["folds"][0]["test_ids"][0] = "tampered"
            with self.assertRaises(Exception):
                verify_split_artifact(loaded)


class SeedFoundationTests(unittest.TestCase):
    def test_seed_registry_is_complete_deterministic_and_collision_free(self):
        first = generate_seed_registry()
        second = generate_seed_registry()
        self.assertEqual(first["registry_sha256"], second["registry_sha256"])
        self.assertEqual(first["n_records"], 3600)
        self.assertEqual(len({row["seed"] for row in first["records"]}), 3600)
        verify_seed_registry(first)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seeds.json"
            digest = write_seed_registry(path, first)
            self.assertEqual(digest, first["registry_sha256"])
            self.assertTrue(path.exists())


class LeakageGuardTests(unittest.TestCase):
    def test_overlap_is_rejected(self):
        with self.assertRaises(LeakageError):
            assert_train_test_disjoint(["1", "2"], ["2", "3"])

    def test_preprocessing_fit_receives_training_rows_only(self):
        frame = validate_wdbc_frame(make_valid_frame())
        train_ids = frame["id"].astype(str).tolist()[:90]
        test_ids = frame["id"].astype(str).tolist()[90:]
        training, testing = partition_fold(frame, train_ids=train_ids, test_ids=test_ids)
        scaler, X_train, X_test = fit_transform_training_only(StandardScaler(), training, testing)
        self.assertEqual(X_train.shape, (90, 30))
        self.assertEqual(X_test.shape, (10, 30))
        expected_mean = training.X.to_numpy(dtype=float).mean(axis=0)
        np.testing.assert_allclose(scaler.mean_, expected_mean)
        self.assertTrue(set(training.ids).isdisjoint(testing.ids))


class EvidenceFoundationTests(unittest.TestCase):
    def test_atomic_evidence_and_manifest_receipts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            receipt = write_json(root / "artifact.json", {"b": 2, "a": 1})
            self.assertEqual(len(receipt.sha256), 64)
            self.assertTrue((root / "artifact.json.sha256").exists())

            manifest = build_foundation_manifest(
                run_id="p1-test",
                git_commit="abcdef1234567890",
                dataset_sha256="c" * 64,
                split_sha256="d" * 64,
                seed_registry_sha256="e" * 64,
            )
            validate_foundation_manifest(manifest)
            manifest_receipt = write_foundation_manifest(root / "manifest.json", manifest)
            self.assertEqual(len(manifest_receipt.sha256), 64)
            with (root / "manifest.json").open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertFalse(loaded["result_bearing"])
            self.assertEqual(loaded["phase"], "P1-foundation")


if __name__ == "__main__":
    unittest.main()
