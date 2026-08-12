from __future__ import annotations

import ast
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from bcimbalance.execution_harness import build_execution_plan
from bcimbalance.experiment_config import load_experiment_config
from bcimbalance.p4_control import (
    AUTHORIZED_P4_BASE,
    CANONICAL_COLAB_DATASET_PATH,
    EXPECTED_P4A_POLICY_SHA256,
    FORBIDDEN_P4A_CALLS,
    FUTURE_REQUIRED_SUBDIRECTORIES,
    FUTURE_RESULT_BEARING_ROOT,
    P4A_CONTROL_ROOT,
    P4AControlError,
    audit_checkpoint_inventory,
    build_control_manifest,
    build_control_session_id,
    build_future_run_id,
    load_control_policy,
    required_future_paths,
    resolve_task_record_path,
    validate_colab_path_policy,
    validate_future_run_root,
    validate_preflight_report,
)
from bcimbalance.seeds import generate_seed_registry

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "P4A_EXECUTION_CONTROL_POLICY_v1.json"
CONFIG = ROOT / "config" / "EXPERIMENT_CONFIG_v1.json"
NOTEBOOK = ROOT / "colab" / "P4A_EXECUTION_CONTROL_GATE.ipynb"


class P4AExecutionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_control_policy(POLICY)
        cls.config = load_experiment_config(CONFIG)
        cls.plan = build_execution_plan(cls.config, generate_seed_registry())

    def test_policy_hash_and_frozen_base(self) -> None:
        self.assertEqual(
            EXPECTED_P4A_POLICY_SHA256,
            "af3212b00c87afbd83032de5a86b1bb933f840a7e6b675e752278e55c2a80c0b",
        )
        self.assertEqual(
            AUTHORIZED_P4_BASE,
            "57f721960f87ac189851024b477fbcdb771a7ed4",
        )
        self.assertFalse(self.policy["result_bearing"])
        self.assertFalse(self.policy["p4b_authorized"])
        self.assertEqual(self.policy["design"]["planned_tasks"], 3600)

    def test_exact_colab_path_policy(self) -> None:
        root = f"{P4A_CONTROL_ROOT}/P4A_CONTROL_20260812T080000Z_57f721960f87"
        validate_colab_path_policy(
            CANONICAL_COLAB_DATASET_PATH,
            root,
            require_exact_colab_paths=True,
        )
        with self.assertRaises(P4AControlError):
            validate_colab_path_policy(
                "/content/drive/MyDrive/35/wrong.csv",
                root,
                require_exact_colab_paths=True,
            )
        with self.assertRaises(P4AControlError):
            validate_colab_path_policy(
                CANONICAL_COLAB_DATASET_PATH,
                f"{FUTURE_RESULT_BEARING_ROOT}/P4_RUN_20260812T080000Z_57f721960f87",
                require_exact_colab_paths=False,
            )

    def test_control_and_future_ids_are_exact_and_utc(self) -> None:
        moment = datetime(2026, 8, 12, 8, 0, 0, tzinfo=timezone.utc)
        control = build_control_session_id(AUTHORIZED_P4_BASE, utc_time=moment)
        future = build_future_run_id(AUTHORIZED_P4_BASE, utc_time=moment)
        self.assertEqual(control, "P4A_CONTROL_20260812T080000Z_57f721960f87")
        self.assertEqual(future, "P4_RUN_20260812T080000Z_57f721960f87")
        run_root = f"{FUTURE_RESULT_BEARING_ROOT}/{future}"
        self.assertEqual(validate_future_run_root(run_root), future)
        self.assertEqual(
            tuple(required_future_paths(run_root)),
            FUTURE_REQUIRED_SUBDIRECTORIES,
        )

    def test_task_record_mapping_is_unique_and_inside_raw(self) -> None:
        run_root = (
            f"{FUTURE_RESULT_BEARING_ROOT}/"
            "P4_RUN_20260812T080000Z_57f721960f87"
        )
        paths = [resolve_task_record_path(task, run_root) for task in self.plan["tasks"]]
        self.assertEqual(len(paths), 3600)
        self.assertEqual(len(set(map(str, paths))), 3600)
        raw_root = (Path(run_root) / "raw").resolve()
        for path in paths[:20] + paths[-20:]:
            path.resolve().relative_to(raw_root)

    def test_checkpoint_inventory_requires_all_tasks_pending_in_p4a(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            counts = audit_checkpoint_inventory(tmp, self.plan, require_all_pending=True)
            self.assertEqual(counts, {"PENDING": 3600, "INCOMPLETE": 0, "COMPLETE": 0})
            task_id = self.plan["tasks"][0]["task_id"]
            Path(tmp, f"{task_id}.inprogress.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(P4AControlError):
                audit_checkpoint_inventory(tmp, self.plan, require_all_pending=True)

    def test_checkpoint_inventory_rejects_unknown_or_ambiguous_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "unknown.inprogress.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(P4AControlError):
                audit_checkpoint_inventory(tmp, self.plan, require_all_pending=False)
        with tempfile.TemporaryDirectory() as tmp:
            task_id = self.plan["tasks"][0]["task_id"]
            Path(tmp, f"{task_id}.inprogress.json").write_text("{}\n", encoding="utf-8")
            Path(tmp, f"{task_id}.complete.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(P4AControlError):
                audit_checkpoint_inventory(tmp, self.plan, require_all_pending=False)

    def test_preflight_report_is_fail_closed(self) -> None:
        report = {
            "status": "PASS",
            "result_bearing": False,
            "scientific_execution": 0,
            "git_commit": AUTHORIZED_P4_BASE,
            "dataset_sha256": self.policy["frozen_identities"]["dataset_sha256"],
            "foundation_sha256": self.policy["frozen_identities"]["foundation_sha256"],
            "split_sha256": self.policy["frozen_identities"]["split_sha256"],
            "seed_registry_sha256": self.policy["frozen_identities"]["seed_registry_sha256"],
            "config_sha256": self.policy["frozen_identities"]["p2_config_sha256"],
            "result_schema_sha256": self.policy["frozen_identities"]["p3_result_schema_sha256"],
            "outer_folds": 50,
            "primary_conditions": 24,
            "planned_execution_tasks": 3600,
            "plan_sha256": self.plan["plan_sha256"],
            "pip_freeze_sha256": "a" * 64,
        }
        validate_preflight_report(report, self.plan, expected_git_commit=AUTHORIZED_P4_BASE)
        tampered = copy.deepcopy(report)
        tampered["scientific_execution"] = 1
        with self.assertRaises(P4AControlError):
            validate_preflight_report(
                tampered,
                self.plan,
                expected_git_commit=AUTHORIZED_P4_BASE,
            )

    def test_manifest_can_only_describe_non_result_bearing_p4a(self) -> None:
        preflight = {
            "dataset_sha256": self.policy["frozen_identities"]["dataset_sha256"],
            "foundation_sha256": self.policy["frozen_identities"]["foundation_sha256"],
            "split_sha256": self.policy["frozen_identities"]["split_sha256"],
            "seed_registry_sha256": self.policy["frozen_identities"]["seed_registry_sha256"],
            "config_sha256": self.policy["frozen_identities"]["p2_config_sha256"],
            "result_schema_sha256": self.policy["frozen_identities"]["p3_result_schema_sha256"],
            "pip_freeze_sha256": "b" * 64,
            "plan_sha256": self.plan["plan_sha256"],
            "planned_execution_tasks": 3600,
        }
        manifest = build_control_manifest(
            session_id="P4A_CONTROL_20260812T080000Z_57f721960f87",
            authorization_ref="GitHub Issue #11 P4A implementation authorization",
            git_commit=AUTHORIZED_P4_BASE,
            preflight_report=preflight,
            checkpoint_summary={"PENDING": 3600, "INCOMPLETE": 0, "COMPLETE": 0},
            policy_sha256=EXPECTED_P4A_POLICY_SHA256,
        )
        self.assertFalse(manifest["p4b_authorized"])
        self.assertFalse(manifest["result_bearing"])
        self.assertEqual(manifest["scientific_execution"], 0)
        self.assertEqual(manifest["planned_tasks"], 3600)

    def test_p4a_python_sources_have_no_scientific_execution_calls(self) -> None:
        paths = [
            ROOT / "src" / "bcimbalance" / "p4_control.py",
            ROOT / "scripts" / "p4a_control.py",
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
        self.assertFalse(FORBIDDEN_P4A_CALLS & observed, sorted(FORBIDDEN_P4A_CALLS & observed))

    def test_colab_notebook_is_control_only(self) -> None:
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        for forbidden in FORBIDDEN_P4A_CALLS:
            self.assertNotIn(f".{forbidden}(", code)
        self.assertIn(CANONICAL_COLAB_DATASET_PATH, code)
        self.assertIn("P4B_AUTHORIZED=false", code)
        self.assertIn("SCIENTIFIC_EXECUTION=0", code)


if __name__ == "__main__":
    unittest.main()
