from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bcimbalance.p4b_authorization import (
    AUTHORIZED_P4B_BASE,
    EXPECTED_DATASET_SHA256,
    EXPECTED_FOUNDATION_SHA256,
    EXPECTED_P2_CONFIG_SHA256,
    EXPECTED_P3_RESULT_SCHEMA_SHA256,
    EXPECTED_P4A_PIP_FREEZE_SHA256,
    EXPECTED_P4A_POLICY_SHA256,
    EXPECTED_PLAN_SHA256,
    EXPECTED_SEED_REGISTRY_SHA256,
    EXPECTED_SPLIT_SHA256,
    P4BAuthorizationError,
    load_policy,
    validate_authorization,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "P4B_EXECUTION_POLICY_v1.json"
SCHEMA = ROOT / "config" / "P4B_AUTHORIZATION_v1.schema.json"


def physical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class P4BAuthorizationTests(unittest.TestCase):
    def test_policy_is_disabled_and_preserves_frozen_identities(self) -> None:
        policy, digest = load_policy(POLICY)
        self.assertEqual(digest, physical_sha(POLICY))
        self.assertEqual(policy["authorized_base"], AUTHORIZED_P4B_BASE)
        self.assertFalse(policy["result_bearing_authorized"])
        self.assertTrue(policy["validation_only"])
        self.assertEqual(policy["authorization"]["first_run_max_tasks"], 1)
        self.assertEqual(policy["design"]["planned_tasks"], 3600)
        self.assertEqual(policy["design"]["threshold"], 0.5)
        self.assertEqual(policy["design"]["positive_class"], "M")

    def test_authorization_schema_requires_exact_one_task(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        props = schema["properties"]
        self.assertTrue(props["result_bearing_authorized"]["const"])
        self.assertEqual(props["max_tasks"]["const"], 1)
        self.assertEqual(props["task_ids"]["minItems"], 1)
        self.assertEqual(props["task_ids"]["maxItems"], 1)
        self.assertFalse(schema["additionalProperties"])

    def test_absent_authorization_fails_before_environment_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            freeze = Path(tmp) / "freeze.txt"
            freeze.write_text("not-used\n", encoding="utf-8")
            run_root = (
                "/content/drive/MyDrive/35/01_Experiment_Evidence/"
                "P4_Result_Bearing_Runs/P4_RUN_20260812T120000Z_ffd7a29c8c05"
            )
            with self.assertRaisesRegex(P4BAuthorizationError, "artifact is absent"):
                validate_authorization(
                    missing,
                    policy_path=POLICY,
                    expected_git_commit=AUTHORIZED_P4B_BASE,
                    expected_task_id="r00-f00-baseline-rf-rep00",
                    expected_run_root=run_root,
                    realized_pip_freeze_path=freeze,
                )

    def _valid_mock_authorization(self, task_id: str, run_root: str) -> dict:
        return {
            "schema_version": "p4b-authorization/v1",
            "result_bearing_authorized": True,
            "exact_git_commit": AUTHORIZED_P4B_BASE,
            "task_ids": [task_id],
            "max_tasks": 1,
            "run_id": Path(run_root).name,
            "run_root": run_root,
            "authorization_ref": "MOCK TEST AUTHORIZATION — NON-SCIENTIFIC",
            "authorized_utc": "2026-08-12T12:00:00+00:00",
            "p4b_policy_sha256": physical_sha(POLICY),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "foundation_sha256": EXPECTED_FOUNDATION_SHA256,
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "seed_registry_sha256": EXPECTED_SEED_REGISTRY_SHA256,
            "p2_config_sha256": EXPECTED_P2_CONFIG_SHA256,
            "p3_result_schema_sha256": EXPECTED_P3_RESULT_SCHEMA_SHA256,
            "p4a_policy_sha256": EXPECTED_P4A_POLICY_SHA256,
            "pip_freeze_sha256": EXPECTED_P4A_PIP_FREEZE_SHA256,
            "execution_plan_sha256": EXPECTED_PLAN_SHA256,
        }

    def test_broader_than_one_task_is_rejected(self) -> None:
        task_id = "r00-f00-baseline-rf-rep00"
        run_root = (
            "/content/drive/MyDrive/35/01_Experiment_Evidence/"
            "P4_Result_Bearing_Runs/P4_RUN_20260812T120000Z_ffd7a29c8c05"
        )
        with tempfile.TemporaryDirectory() as tmp:
            auth_path = Path(tmp) / "auth.json"
            freeze = Path(tmp) / "freeze.txt"
            freeze.write_text("mock\n", encoding="utf-8")
            auth = self._valid_mock_authorization(task_id, run_root)
            auth["task_ids"] = [task_id, "r00-f00-baseline-rf-rep01"]
            auth_path.write_text(json.dumps(auth), encoding="utf-8")
            with self.assertRaisesRegex(P4BAuthorizationError, "exactly one"):
                validate_authorization(
                    auth_path,
                    policy_path=POLICY,
                    expected_git_commit=AUTHORIZED_P4B_BASE,
                    expected_task_id=task_id,
                    expected_run_root=run_root,
                    realized_pip_freeze_path=freeze,
                )

    def test_mock_exact_grant_validates_without_scientific_execution(self) -> None:
        task_id = "r00-f00-baseline-rf-rep00"
        run_root = (
            "/content/drive/MyDrive/35/01_Experiment_Evidence/"
            "P4_Result_Bearing_Runs/P4_RUN_20260812T120000Z_ffd7a29c8c05"
        )
        with tempfile.TemporaryDirectory() as tmp:
            auth_path = Path(tmp) / "auth.json"
            freeze = Path(tmp) / "freeze.txt"
            freeze.write_text("mock-pip-freeze\n", encoding="utf-8")
            auth_path.write_text(
                json.dumps(self._valid_mock_authorization(task_id, run_root)),
                encoding="utf-8",
            )

            policy_digest = physical_sha(POLICY)
            auth_digest = hashlib.sha256(auth_path.read_bytes()).hexdigest()

            def fake_sha(path) -> str:
                p = Path(path)
                if p == POLICY:
                    return policy_digest
                if p == auth_path:
                    return auth_digest
                if p == freeze:
                    return EXPECTED_P4A_PIP_FREEZE_SHA256
                raise AssertionError(f"unexpected mocked hash path: {p}")

            with patch("bcimbalance.p4b_authorization.sha256_file", side_effect=fake_sha):
                grant = validate_authorization(
                    auth_path,
                    policy_path=POLICY,
                    expected_git_commit=AUTHORIZED_P4B_BASE,
                    expected_task_id=task_id,
                    expected_run_root=run_root,
                    realized_pip_freeze_path=freeze,
                )
            self.assertEqual(grant.task_id, task_id)
            self.assertEqual(grant.run_root, run_root)
            self.assertEqual(grant.p4b_policy_sha256, policy_digest)


if __name__ == "__main__":
    unittest.main()
