from __future__ import annotations

import ast
import copy
import inspect
import json
import unittest
from importlib import metadata
from pathlib import Path

from bcimbalance.experiment_config import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_PINS,
    ExperimentConfigError,
    canonical_config_sha256,
    load_experiment_config,
    validate_experiment_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "EXPERIMENT_CONFIG_v1.json"


class P2ConfigLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_experiment_config(CONFIG_PATH)

    def test_config_hash_and_non_result_bearing_status(self) -> None:
        self.assertFalse(self.config["result_bearing"])
        self.assertEqual(self.config["config_sha256"], EXPECTED_CONFIG_SHA256)
        self.assertEqual(canonical_config_sha256(self.config), EXPECTED_CONFIG_SHA256)

    def test_frozen_factorial_design(self) -> None:
        design = self.config["primary_design"]
        self.assertEqual(len(design["methods"]), 6)
        self.assertEqual(len(design["classifiers"]), 4)
        self.assertEqual(design["condition_count"], 24)
        self.assertEqual(design["replicates"], 3)
        self.assertEqual(design["decision_threshold"], 0.5)
        self.assertEqual(design["positive_class"], "M")
        self.assertEqual(design["primary_metric"], "malignant_recall")
        self.assertEqual(design["class_weighting"], "forbidden")

    def test_exact_tomek_mapping_and_ctgan_condition(self) -> None:
        methods = self.config["imbalance_methods"]
        tomek = methods["smote_tomek"]["parameters"]["tomek"]
        self.assertEqual(tomek["class"], "imblearn.under_sampling.TomekLinks")
        self.assertEqual(tomek["parameters"]["sampling_strategy"], "all")
        ctgan = methods["ctgan"]
        self.assertEqual(ctgan["parameters"]["epochs"], 400)
        self.assertEqual(ctgan["parameters"]["batch_size"], 50)
        self.assertEqual(ctgan["parameters"]["pac"], 10)
        self.assertFalse(ctgan["parameters"]["enable_gpu"])
        self.assertEqual(ctgan["conditional_sampling"]["column_values"], {"diagnosis": "M"})
        self.assertTrue(ctgan["conditional_sampling"]["forbid_unconditional_sample_then_label_overwrite"])

    def test_tampering_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.config)
        tampered["primary_design"]["decision_threshold"] = 0.4
        with self.assertRaises(ExperimentConfigError):
            validate_experiment_config(tampered)

    def test_direct_package_versions_are_exactly_pinned(self) -> None:
        for distribution, expected in EXPECTED_PINS.items():
            self.assertEqual(metadata.version(distribution), expected, distribution)

    def test_classifier_constructor_compatibility_without_fit(self) -> None:
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
        from xgboost import XGBClassifier

        classes = {
            "rf": RandomForestClassifier,
            "adaboost": AdaBoostClassifier,
            "xgboost": XGBClassifier,
            "lightgbm": LGBMClassifier,
        }
        for name, cls in classes.items():
            params = copy.deepcopy(self.config["classifiers"][name]["parameters"])
            if params.get("random_state") == "$SEED":
                params["random_state"] = 123456789
            estimator = cls(**params)
            realized = estimator.get_params(deep=False)
            for key, value in params.items():
                self.assertIn(key, realized, f"{name}:{key}")
                self.assertEqual(realized[key], value, f"{name}:{key}")

    def test_resampler_constructor_compatibility_without_execution(self) -> None:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
        from imblearn.under_sampling import TomekLinks

        seed = 123456789
        ADASYN(sampling_strategy=1.0, n_neighbors=5, random_state=seed)
        BorderlineSMOTE(
            sampling_strategy=1.0,
            k_neighbors=5,
            m_neighbors=10,
            kind="borderline-1",
            random_state=seed,
        )
        SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=seed)
        inner_smote = SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=seed)
        tomek = TomekLinks(sampling_strategy="all", n_jobs=1)
        combo = SMOTETomek(
            sampling_strategy=1.0,
            random_state=seed,
            smote=inner_smote,
            tomek=tomek,
            n_jobs=1,
        )
        self.assertEqual(combo.tomek.sampling_strategy, "all")

    def test_ctgan_api_compatibility_without_fit_or_sample(self) -> None:
        from sdv.sampling import Condition
        from sdv.single_table import CTGANSynthesizer

        init_params = inspect.signature(CTGANSynthesizer.__init__).parameters
        expected = set(self.config["imbalance_methods"]["ctgan"]["parameters"])
        self.assertTrue(expected.issubset(init_params), sorted(expected - set(init_params)))
        condition_params = inspect.signature(Condition.__init__).parameters
        self.assertIn("num_rows", condition_params)
        self.assertIn("column_values", condition_params)

    def test_p2_implementation_has_no_result_bearing_calls(self) -> None:
        forbidden = {
            "fit",
            "fit_resample",
            "sample",
            "sample_from_conditions",
            "predict",
            "predict_proba",
            "score",
        }
        path = ROOT / "src" / "bcimbalance" / "experiment_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        observed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    observed.add(func.attr)
                elif isinstance(func, ast.Name):
                    observed.add(func.id)
        self.assertFalse(forbidden & observed, sorted(forbidden & observed))

    def test_schema_file_is_valid_json_and_names_config_schema(self) -> None:
        schema_path = ROOT / "config" / "EXPERIMENT_CONFIG_v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], "experiment-config/v1")
        self.assertEqual(schema["properties"]["result_bearing"]["const"], False)


if __name__ == "__main__":
    unittest.main()
