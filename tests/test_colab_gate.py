from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bcimbalance.colab_gate import ColabFoundationGateError, verify_colab_foundation
from bcimbalance.dataset import FEATURE_COLUMNS, sha256_file, validate_wdbc_frame
from bcimbalance.folds import build_outer_folds
from bcimbalance.protocol import MASTER_SEED, PROTOCOL_VERSION
from bcimbalance.seeds import generate_seed_registry


def make_fixture_frame(n_b: int = 60, n_m: int = 40) -> pd.DataFrame:
    n = n_b + n_m
    data: dict[str, object] = {
        "id": [700000 + i for i in range(n)],
        "diagnosis": ["B"] * n_b + ["M"] * n_m,
    }
    base = np.linspace(2.0, 12.0, n)
    for index, column in enumerate(FEATURE_COLUMNS):
        data[column] = base + index * 0.005
    frame = pd.DataFrame(data)
    frame["Unnamed: 32"] = np.nan
    return frame


def write_fixture_lock(csv_path: Path, lock_path: Path) -> dict[str, object]:
    frame = validate_wdbc_frame(pd.read_csv(csv_path))
    dataset_hash = sha256_file(csv_path)
    splits = build_outer_folds(frame, dataset_sha256=dataset_hash)
    seeds = generate_seed_registry()
    lock: dict[str, object] = {
        "schema_version": "foundation-lock/v1",
        "protocol_version": PROTOCOL_VERSION,
        "result_bearing": False,
        "dataset": {
            "name": "fixture",
            "sha256": dataset_hash,
            "bytes": csv_path.stat().st_size,
            "rows": len(frame),
            "features": len(FEATURE_COLUMNS),
            "class_counts": {
                str(k): int(v)
                for k, v in frame["diagnosis"].value_counts().sort_index().to_dict().items()
            },
        },
        "outer_folds": {
            "design": "5x10 repeated stratified",
            "count": len(splits["folds"]),
            "sha256": splits["split_sha256"],
        },
        "seed_registry": {
            "master_seed": MASTER_SEED,
            "records": seeds["n_records"],
            "sha256": seeds["registry_sha256"],
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


class ColabFoundationGateTests(unittest.TestCase):
    def test_gate_passes_only_when_dataset_split_and_seed_identity_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "wdbc.csv"
            lock = root / "FOUNDATION_LOCK_v1.json"
            output = root / "evidence"
            make_fixture_frame().to_csv(dataset, index=False)
            expected = write_fixture_lock(dataset, lock)

            report = verify_colab_foundation(
                dataset_path=dataset,
                lock_path=lock,
                output_dir=output,
                git_commit="abcdef1234567890",
            )
            self.assertEqual(report["gate"], "PASS")
            self.assertFalse(report["result_bearing"])
            self.assertEqual(report["dataset_sha256"], expected["dataset"]["sha256"])
            self.assertEqual(report["outer_folds"], 50)
            self.assertEqual(report["seed_records"], 3600)
            self.assertTrue((output / "colab_foundation_gate.json").exists())
            self.assertTrue((output / "colab_foundation_manifest.json").exists())
            self.assertTrue((output / "colab_foundation_gate.json.sha256").exists())
            self.assertTrue((output / "colab_foundation_manifest.json.sha256").exists())

    def test_gate_fails_closed_after_dataset_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "wdbc.csv"
            lock = root / "FOUNDATION_LOCK_v1.json"
            output = root / "evidence"
            make_fixture_frame().to_csv(dataset, index=False)
            write_fixture_lock(dataset, lock)
            with dataset.open("ab") as handle:
                handle.write(b"tamper")

            with self.assertRaises(ColabFoundationGateError):
                verify_colab_foundation(
                    dataset_path=dataset,
                    lock_path=lock,
                    output_dir=output,
                    git_commit="abcdef1234567890",
                )
            self.assertFalse((output / "colab_foundation_gate.json").exists())

    def test_colab_notebook_code_contains_no_result_bearing_scientific_calls(self):
        notebook_path = Path("colab/P1C_COLAB_FOUNDATION_GATE.ipynb")
        if not notebook_path.exists():
            self.skipTest("notebook not created yet")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        prohibited_modules = {"imblearn", "xgboost", "lightgbm", "ctgan", "sklearn.ensemble"}
        prohibited_symbols = {
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "XGBClassifier",
            "LGBMClassifier",
            "ADASYN",
            "BorderlineSMOTE",
            "SMOTE",
            "SMOTETomek",
            "CTGAN",
            "fit_resample",
        }
        violations: list[str] = []
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            tree = ast.parse(source, filename=f"notebook-cell-{index}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(alias.name == module or alias.name.startswith(module + ".") for module in prohibited_modules):
                            violations.append(f"cell {index}: prohibited import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if any(module == banned or module.startswith(banned + ".") for banned in prohibited_modules):
                        violations.append(f"cell {index}: prohibited import-from {module}")
                    for alias in node.names:
                        if alias.name in prohibited_symbols:
                            violations.append(f"cell {index}: prohibited symbol import {alias.name}")
                elif isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id in prohibited_symbols:
                        violations.append(f"cell {index}: prohibited call {func.id}")
                    elif isinstance(func, ast.Attribute) and func.attr in prohibited_symbols:
                        violations.append(f"cell {index}: prohibited call .{func.attr}")
        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
