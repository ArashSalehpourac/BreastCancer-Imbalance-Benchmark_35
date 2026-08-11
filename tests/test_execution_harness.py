from __future__ import annotations

import ast
import copy
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from bcimbalance.execution_harness import (
    AUTHORIZED_P3_BASE,
    EXPECTED_EXECUTION_TASKS,
    EXPECTED_FOUNDATION_SHA256,
    EXPECTED_PRIMARY_CONDITIONS,
    EXPECTED_RESULT_SCHEMA_SHA256,
    FORBIDDEN_SCIENTIFIC_CALLS,
    PreflightError,
    build_execution_plan,
    checkpoint_state,
    load_result_schema,
    verify_direct_pins,
    write_inprogress_checkpoint,
)
from bcimbalance.experiment_config import EXPECTED_CONFIG_SHA256, load_experiment_config
from bcimbalance.seeds import SeedRegistryError, generate_seed_registry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "EXPERIMENT_CONFIG_v1.json"
RESULT_SCHEMA = ROOT / "config" / "RESULT_EVIDENCE_SCHEMA_v1.json"
P3_CI = os.environ.get("P3_EXECUTION_INFRA_CI") == "1"


class P3ExecutionInfrastructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_experiment_config(CONFIG)
        cls.seed_registry = generate_seed_registry()
        cls.plan = build_execution_plan(cls.config, cls.seed_registry)

    def test_authorized_base_and_frozen_hashes(self) -> None:
        self.assertEqual(
            AUTHORIZED_P3_BASE,
            "8c6ff728375de033b32fac8b32240a22e98401e1",
        )
        self.assertEqual(
            EXPECTED_CONFIG_SHA256,
            "71e82edad62dcf06382cf85dcd87e642049e80668eaa7dbd9913d7a5a5bb7dc9",
        )
        self.assertEqual(
            EXPECTED_FOUNDATION_SHA256,
            "49291df347b8aa8453a68cf06296642f3939bfc3fe51ce452d3620d9c228e030",
        )
        self.assertEqual(
            EXPECTED_RESULT_SCHEMA_SHA256,
            "7819dd71c49d3c5c686ca76079a92f0b5399596d0de71d2964eca8ece8af4686",
        )

    def test_result_schema_is_frozen_and_protocol_derived(self) -> None:
        schema = load_result_schema(RESULT_SCHEMA)
        self.assertFalse(schema["artifact_result_bearing"])
        self.assertTrue(schema["describes_result_bearing_evidence"])
        self.assertTrue(schema["no_manual_transcription"])
        self.assertEqual(schema["schema_sha256"], EXPECTED_RESULT_SCHEMA_SHA256)
        self.assertEqual(
            schema["required_artifacts"]["run_manifest"]["path"],
            "evidence/manifests/run_manifest.json",
        )
        self.assertEqual(
            schema["required_artifacts"]["predictions"]["path"],
            "evidence/raw/predictions.csv",
        )
        self.assertEqual(
            schema["required_artifacts"]["primary_tests"]["path"],
            "evidence/stats/primary_tests.csv",
        )

    def test_result_schema_tampering_fails_closed(self) -> None:
        payload = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
        payload["required_artifacts"]["predictions"]["required_columns"][-1] = "tampered"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PreflightError):
                load_result_schema(path)

    def test_execution_plan_is_complete_unique_and_non_result_bearing(self) -> None:
        plan = self.plan
        self.assertFalse(plan["result_bearing"])
        self.assertEqual(plan["primary_condition_count"], EXPECTED_PRIMARY_CONDITIONS)
        self.assertEqual(plan["execution_task_count"], EXPECTED_EXECUTION_TASKS)
        self.assertEqual(len(plan["tasks"]), EXPECTED_EXECUTION_TASKS)
        self.assertEqual(
            len({task["task_id"] for task in plan["tasks"]}),
            EXPECTED_EXECUTION_TASKS,
        )
        self.assertEqual(
            len({task["evidence_path"] for task in plan["tasks"]}),
            EXPECTED_EXECUTION_TASKS,
        )
        counts = Counter(task["primary_condition_id"] for task in plan["tasks"])
        self.assertEqual(len(counts), EXPECTED_PRIMARY_CONDITIONS)
        self.assertEqual(set(counts.values()), {150})

    def test_execution_plan_is_deterministic(self) -> None:
        second = build_execution_plan(self.config, generate_seed_registry())
        self.assertEqual(second["plan_sha256"], self.plan["plan_sha256"])
        self.assertEqual(second["tasks"], self.plan["tasks"])

    def test_checkpoint_state_is_fail_closed(self) -> None:
        task = self.plan["tasks"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(checkpoint_state(root, task["task_id"]), "PENDING")
            marker = write_inprogress_checkpoint(root, task, git_commit="a" * 40)
            self.assertTrue(marker.is_file())
            self.assertEqual(checkpoint_state(root, task["task_id"]), "INCOMPLETE")
            with self.assertRaises(PreflightError):
                write_inprogress_checkpoint(root, task, git_commit="a" * 40)

            complete = root / f"{task['task_id']}.complete.json"
            complete.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(PreflightError):
                checkpoint_state(root, task["task_id"])

    @unittest.skipUnless(P3_CI, "exact P2 scientific packages are installed only in P3 CI")
    def test_realized_direct_package_pins_match_p2_lock(self) -> None:
        realized = verify_direct_pins()
        self.assertEqual(realized["scikit-learn"], "1.9.0")
        self.assertEqual(realized["imbalanced-learn"], "0.14.2")
        self.assertEqual(realized["xgboost"], "3.3.0")
        self.assertEqual(realized["lightgbm"], "4.6.0")
        self.assertEqual(realized["sdv"], "1.36.3")
        self.assertEqual(realized["ctgan"], "0.12.1")

    def test_p3_python_sources_contain_no_scientific_execution_calls(self) -> None:
        paths = [
            ROOT / "src" / "bcimbalance" / "execution_harness.py",
            ROOT / "scripts" / "p3_preflight.py",
        ]
        observed: set[str] = set()
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute):
                    observed.add(func.attr)
                elif isinstance(func, ast.Name):
                    observed.add(func.id)
        self.assertFalse(
            FORBIDDEN_SCIENTIFIC_CALLS & observed,
            sorted(FORBIDDEN_SCIENTIFIC_CALLS & observed),
        )

    def test_plan_tampering_fails_by_seed_registry_cross_link(self) -> None:
        tampered = copy.deepcopy(self.seed_registry)
        tampered["registry_sha256"] = "0" * 64
        with self.assertRaises(SeedRegistryError):
            build_execution_plan(self.config, tampered)


if __name__ == "__main__":
    unittest.main()
