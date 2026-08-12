from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bcimbalance.p4b_authorization import P4BAuthorizationError
from bcimbalance.scientific_task import run_authorized_task

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "P4B_EXECUTION_POLICY_v1.json"
SCRIPT = ROOT / "scripts" / "p4b_execute.py"


class P4BScientificGateTests(unittest.TestCase):
    def test_missing_authorization_blocks_before_scientific_executor_and_run_root_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_auth = tmp_path / "missing.json"
            freeze = tmp_path / "freeze.txt"
            freeze.write_text("mock\n", encoding="utf-8")
            plan = tmp_path / "plan.json"
            plan.write_text("{}\n", encoding="utf-8")
            dataset = tmp_path / "dataset.csv"
            dataset.write_text("not-read\n", encoding="utf-8")
            run_root = (
                "/content/drive/MyDrive/35/01_Experiment_Evidence/"
                "P4_Result_Bearing_Runs/P4_RUN_20260812T120000Z_ffd7a29c8c05"
            )

            with patch("bcimbalance.scientific_task._execute_scientific_task") as executor:
                with self.assertRaises(P4BAuthorizationError):
                    run_authorized_task(
                        authorization_path=missing_auth,
                        policy_path=POLICY,
                        realized_pip_freeze_path=freeze,
                        expected_git_commit="ffd7a29c8c05d17ac4e20dae9db75492d1ed1dff",
                        expected_task_id="r00-f00-baseline-rf-rep00",
                        run_root=run_root,
                        plan_path=plan,
                        dataset_path=dataset,
                        config_path=ROOT / "config" / "EXPERIMENT_CONFIG_v1.json",
                    )
                executor.assert_not_called()

    def test_public_entry_point_orders_authorization_before_scientific_execution(self) -> None:
        source = inspect.getsource(run_authorized_task)
        authorization_position = source.index("validate_authorization(")
        path_creation_position = source.index("required_future_paths(")
        scientific_position = source.index("_execute_scientific_task(")
        self.assertLess(authorization_position, path_creation_position)
        self.assertLess(authorization_position, scientific_position)

    def test_controller_dispatches_isolated_worker_with_preimport_environment(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"--worker-execute-one"', text)
        self.assertIn('"P4B_AUTHORIZED_WORKER": "1"', text)
        self.assertIn('"PYTHONHASHSEED": str(seed)', text)
        self.assertIn('"MKL_NUM_THREADS": "1"', text)
        self.assertIn('"OPENBLAS_NUM_THREADS": "1"', text)
        self.assertIn("Scientific modules are deliberately imported only after", text)
        self.assertIn("immutable P4B first-run root already exists", text)

    def test_worker_checks_environment_before_scientific_import_textually(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        worker_start = text.index("def _worker_execute_one")
        worker = text[worker_start:]
        env_check = worker.index('required_env = {')
        scientific_import = worker.index("from bcimbalance.scientific_task import run_authorized_task")
        self.assertLess(env_check, scientific_import)
        self.assertIn('os.environ.get("P4B_AUTHORIZED_WORKER") != "1"', worker[:scientific_import])
        self.assertIn("Path(args.run_root).exists()", worker[:scientific_import])

    def test_validation_modules_contain_no_scientific_calls(self) -> None:
        forbidden = {
            "fit", "fit_resample", "sample", "sample_from_conditions",
            "predict", "predict_proba", "score",
        }
        for relative in (
            "src/bcimbalance/p4b_authorization.py",
            "scripts/p4b_execute.py",
        ):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            observed = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        observed.add(node.func.attr)
                    elif isinstance(node.func, ast.Name):
                        observed.add(node.func.id)
            self.assertFalse(forbidden & observed, f"{relative}: {sorted(forbidden & observed)}")

    def test_scientific_calls_exist_only_in_future_executor_module(self) -> None:
        text = (ROOT / "src" / "bcimbalance" / "scientific_task.py").read_text(encoding="utf-8")
        self.assertIn("model.fit(", text)
        self.assertIn("fit_resample(", text)
        self.assertIn("sample_from_conditions(", text)
        self.assertIn("predict_proba(", text)
        self.assertIn("_compute_metrics(", text)
        self.assertIn("validate_authorization(", text)


if __name__ == "__main__":
    unittest.main()
