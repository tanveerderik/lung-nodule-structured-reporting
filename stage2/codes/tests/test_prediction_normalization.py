#!/usr/bin/env python3
import sys
import unittest
from copy import deepcopy
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from prediction_normalization import (  # noqa: E402
    normalize_prediction,
    remove_nulls,
    safe_json_loads,
)


class PredictionNormalizationTests(unittest.TestCase):
    def test_all_supported_scalar_aliases_are_omitted_case_insensitively(self):
        aliases = [
            None, "", "   ", "null", "NULL", " Null ",
            "none", "None", "NONE", " nOnE ",
            "N/A", "n/a", "N/a", " n/A ",
        ]
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertEqual(remove_nulls({"A": alias}), {})

    def test_numeric_zero_boolean_false_and_meaningful_terms_survive(self):
        value = {
            "integer_zero": 0,
            "float_zero": 0.0,
            "false": False,
            "negative": "negative",
            "absent": "absent",
            "no": "no",
            "normal": "normal",
            "unknown": "unknown",
            "not_seen": "not seen",
            "not_identified": "not identified",
        }
        self.assertEqual(remove_nulls(value), value)

    def test_equivalent_aliases_have_identical_normalized_results(self):
        results = [
            normalize_prediction(
                {"Number of Nodules": 1, "Nodules": [{"Lobe": alias}]}
            )
            for alias in (None, "null", "NULL", "none", "None", "NONE", "N/A")
        ]
        self.assertTrue(all(result == results[0] for result in results))

    def test_recursive_none_alias_inside_nodule_array(self):
        dense = {
            "Number of Nodules": 1,
            "Nodules": [
                {"Nodule ID": 1, "Calcification Patterns": "  none  "}
            ],
        }
        self.assertEqual(
            normalize_prediction(dense),
            {"Number of Nodules": 1, "Nodules": [{"Nodule ID": 1}]},
        )

    def test_normalization_does_not_mutate_dense_input(self):
        dense = {
            "Number of Nodules": "1",
            "Nodules": [{"Nodule ID": "1", "Margin": "none"}],
        }
        before = deepcopy(dense)
        normalize_prediction(dense)
        self.assertEqual(dense, before)

    def test_recursive_json_null_pruning(self):
        dense = {
            "Number of Nodules": "1",
            "Nodules": [
                {
                    "Nodule ID": "1",
                    "Lobe": "right upper lobe",
                    "Margin": None,
                    "Shape": "null",
                    "Segment": "N/A",
                    "Fissure": "NULL",
                    "Comparison Date": "",
                }
            ],
            "Overall Lung-RADS": None,
            "Follow-up Date": "null",
        }
        normalized = normalize_prediction(dense)
        self.assertEqual(normalized["Number of Nodules"], 1)
        self.assertEqual(
            normalized["Nodules"],
            [{"Nodule ID": 1, "Lobe": "right upper lobe"}],
        )
        self.assertNotIn("Overall Lung-RADS", normalized)
        self.assertNotIn("Follow-up Date", normalized)

    def test_count_zero_dense_becomes_sparse(self):
        dense = {
            "Number of Nodules": "0",
            "Overall Lung-RADS": None,
            "Recommend Imaging": "null",
            "Imaging Interval": None,
            "Follow-up Date": None,
        }
        self.assertEqual(normalize_prediction(dense), {})

    def test_real_report_level_value_survives_count_zero(self):
        dense = {
            "Number of Nodules": 0,
            "Overall Lung-RADS": "1",
            "Recommend Imaging": None,
        }
        self.assertEqual(normalize_prediction(dense), {"Overall Lung-RADS": "1"})

    def test_empty_nested_containers_are_removed(self):
        self.assertEqual(
            remove_nulls({"A": [], "B": {}, "C": [None, "null"], "D": 2}),
            {"D": 2},
        )

    def test_parser_preserves_dense_object(self):
        parsed = safe_json_loads('<json>{"Lobe": null}</json>')
        self.assertEqual(parsed, {"Lobe": None})
        self.assertEqual(normalize_prediction(parsed), {})


if __name__ == "__main__":
    unittest.main()
