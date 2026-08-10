from __future__ import annotations

import ast
import unittest
from pathlib import Path


class P1ScopeGuardTests(unittest.TestCase):
    def test_foundation_contains_no_result_bearing_model_or_resampler_imports(self):
        prohibited_modules = {
            "imblearn",
            "xgboost",
            "lightgbm",
            "ctgan",
            "sklearn.ensemble",
        }
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

        roots = [Path("src/bcimbalance"), Path("scripts")]
        violations: list[str] = []
        for root in roots:
            for path in sorted(root.glob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if any(alias.name == module or alias.name.startswith(module + ".") for module in prohibited_modules):
                                violations.append(f"{path}: prohibited import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if any(module == banned or module.startswith(banned + ".") for banned in prohibited_modules):
                            violations.append(f"{path}: prohibited import-from {module}")
                        for alias in node.names:
                            if alias.name in prohibited_symbols:
                                violations.append(f"{path}: prohibited symbol import {alias.name}")
                    elif isinstance(node, ast.Call):
                        func = node.func
                        if isinstance(func, ast.Name) and func.id in prohibited_symbols:
                            violations.append(f"{path}: prohibited call {func.id}")
                        elif isinstance(func, ast.Attribute) and func.attr in prohibited_symbols:
                            violations.append(f"{path}: prohibited call .{func.attr}")

        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
