#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]


class RescoreSavedCasesTests(unittest.TestCase):
    def test_rebuild_preserves_dense_object_and_raw_schema_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases_path = root / "cases.json"
            summary_path = root / "summary.json"
            schema_path = root / "schema.json"
            parsed_dense = {
                "Number of Nodules": 1,
                "Nodules": [
                    {"Nodule ID": 1, "Calcification Patterns": "none"}
                ],
            }
            cases = [
                {
                    "ID": 214,
                    "Ground Truth": {
                        "Number of Nodules": 1,
                        "Nodules": [{"Nodule ID": 1}],
                    },
                    "Prediction": parsed_dense,
                    "Parsed Prediction Dense": parsed_dense,
                    "Raw Prediction": json.dumps(parsed_dense),
                    "Eval Metrics": {},
                }
            ]
            schema = {
                "type": "object",
                "properties": {
                    "Number of Nodules": {"type": "integer"},
                    "Nodules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Nodule ID": {"type": "integer"},
                                "Calcification Patterns": {
                                    "anyOf": [
                                        {"type": "string", "enum": ["central"]},
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            }
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            summary_path.write_text(json.dumps({}), encoding="utf-8")
            schema_path.write_text(json.dumps(schema), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(CODE_DIR / "rescore_eval_cases.py"),
                    "--eval_cases", str(cases_path),
                    "--summary_json", str(summary_path),
                    "--schema_file", str(schema_path),
                    "--arm_label", "gemma3_4B/base_sparse",
                    "--raw_validation_mode", "json_schema",
                    "--in_place",
                ],
                check=True,
                cwd=CODE_DIR,
                capture_output=True,
                text=True,
            )

            rebuilt = json.loads(cases_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(rebuilt["Parsed Prediction Dense"], parsed_dense)
            self.assertEqual(
                rebuilt["Prediction"],
                {"Number of Nodules": 1, "Nodules": [{"Nodule ID": 1}]},
            )
            self.assertFalse(rebuilt["Raw Schema Validation"]["valid"])
            self.assertEqual(
                rebuilt["Raw Schema Validation"]["errors"][0]["path"],
                "$.Nodules[0].Calcification Patterns",
            )
            self.assertTrue(
                Path(str(cases_path) + ".pre_null_alias_rebuild.bak").is_file()
            )
            self.assertTrue(
                Path(str(summary_path) + ".pre_null_alias_rebuild.bak").is_file()
            )

    def test_dynamic_arm_uses_authoritative_template_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases_path = root / "cases.json"
            summary_path = root / "summary.json"
            schema_path = root / "schema.json"
            parsed_dense = {"Number of Nodules": "0", "Overall Lung-RADS": "null"}
            cases_path.write_text(
                json.dumps(
                    [
                        {
                            "ID": 1,
                            "Ground Truth": {},
                            "Prediction": {},
                            "Parsed Prediction Dense": parsed_dense,
                            "Raw Prediction": json.dumps(parsed_dense),
                            "Dynamic Constraint Validation": {"valid": True, "errors": []},
                            "Eval Metrics": {},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            summary_path.write_text(json.dumps({}), encoding="utf-8")
            schema_path.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "properties": {"Number of Nodules": {"type": "integer"}},
                        "additionalProperties": False,
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(CODE_DIR / "rescore_eval_cases.py"),
                    "--eval_cases", str(cases_path),
                    "--summary_json", str(summary_path),
                    "--schema_file", str(schema_path),
                    "--arm_label", "llama3_1_70B/base_dc_dense",
                    "--raw_validation_mode", "dynamic_template",
                    "--in_place",
                ],
                check=True,
                cwd=CODE_DIR,
                capture_output=True,
                text=True,
            )
            rebuilt = json.loads(cases_path.read_text(encoding="utf-8"))[0]
            self.assertTrue(rebuilt["Raw Schema Validation"]["valid"])
            self.assertEqual(
                rebuilt["Raw Schema Validation"]["contract"],
                "authentic_dynamic_template",
            )


if __name__ == "__main__":
    unittest.main()
