#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from prediction_normalization import normalize_prediction  # noqa: E402
from schema_validation import validate_raw_prediction  # noqa: E402


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "Calcification Patterns": {
            "anyOf": [
                {"type": "string", "enum": ["central", "diffuse"]},
                {"type": "null"},
            ]
        }
    },
    "additionalProperties": False,
}


class RawSchemaValidityTests(unittest.TestCase):
    def test_json_null_is_schema_valid_and_semantically_omitted(self):
        raw = {"Calcification Patterns": None}
        self.assertTrue(validate_raw_prediction(raw, SCHEMA)["valid"])
        self.assertEqual(normalize_prediction(raw), {})

    def test_none_alias_is_schema_invalid_but_semantically_omitted(self):
        raw = {"Calcification Patterns": "none"}
        validation = validate_raw_prediction(raw, SCHEMA)
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["errors"][0]["path"], "$.Calcification Patterns")
        self.assertEqual(normalize_prediction(raw), {})


if __name__ == "__main__":
    unittest.main()
