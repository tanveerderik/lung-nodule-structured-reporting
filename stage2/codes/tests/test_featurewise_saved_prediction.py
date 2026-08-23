#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]


class FeaturewiseSavedPredictionTests(unittest.TestCase):
    def test_empty_saved_prediction_does_not_fall_back_to_raw_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases_path = root / "cases.json"
            schema_path = root / "template.json"
            output_json = root / "featurewise.json"
            output_csv = root / "featurewise.csv"
            cases_path.write_text(
                json.dumps(
                    [
                        {
                            "Ground Truth": {},
                            "Prediction": {},
                            "Raw Prediction": json.dumps(
                                {"Nodules": [{"Lobe": "right upper lobe"}]}
                            ),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            schema_path.write_text(
                json.dumps(
                    {
                        "Lungs Pleura": {
                            "Nodule Findings": {
                                "Nodules": [
                                    {
                                        "Lobe": {
                                            "data_type": "categorical",
                                            "values": ["right upper lobe", "null"],
                                        }
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(CODE_DIR / "featurewise_eval.py"),
                    "--eval_cases", str(cases_path),
                    "--schema_file", str(schema_path),
                    "--output_csv", str(output_csv),
                    "--output_json", str(output_json),
                ],
                check=True,
                cwd=CODE_DIR,
                capture_output=True,
                text=True,
            )
            rows = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["feature"], "Nodules.Lobe")
            self.assertEqual(rows[0]["FP"], 0)


if __name__ == "__main__":
    unittest.main()
