import unittest

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from date_normalization import canonical_date_key, is_date_field  # noqa: E402
from nodule_scoring import score_case  # noqa: E402


class DateNormalizationTests(unittest.TestCase):
    def test_padding_is_not_semantic(self):
        self.assertEqual(
            canonical_date_key("3/2/2022"), canonical_date_key("03/02/2022")
        )

    def test_two_and_four_digit_year_equivalence(self):
        self.assertEqual(
            canonical_date_key("3/2/22"), canonical_date_key("03/02/2022")
        )

    def test_precision_is_preserved(self):
        self.assertNotEqual(
            canonical_date_key("3/2022"), canonical_date_key("3/1/2022")
        )
        self.assertNotEqual(
            canonical_date_key("2022"), canonical_date_key("1/1/2022")
        )

    def test_calendar_invalid_dates_are_rejected(self):
        self.assertIsNotNone(canonical_date_key("2/29/2024"))
        self.assertIsNone(canonical_date_key("2/29/2023"))
        self.assertIsNone(canonical_date_key("13/1/2022"))

    def test_date_field_paths_are_recognized(self):
        self.assertTrue(is_date_field("Nodules.Comparison Date"))
        self.assertTrue(is_date_field("Follow-up Date"))
        self.assertFalse(is_date_field("Imaging Interval"))

    def test_nodule_score_accepts_padding_only_difference(self):
        gold = {
            "Nodules": [
                {"Series ID": 1, "Image ID": 2, "Comparison Date": "3/2/2022"}
            ]
        }
        pred = {
            "Nodules": [
                {"Series ID": 1, "Image ID": 2, "Comparison Date": "03/02/2022"}
            ]
        }
        metrics = score_case(gold, pred)
        self.assertEqual(metrics["FP"], 0)
        self.assertEqual(metrics["FN"], 0)

    def test_nodule_score_does_not_invent_missing_precision(self):
        gold = {
            "Nodules": [
                {"Series ID": 1, "Image ID": 2, "Comparison Date": "3/2022"}
            ]
        }
        pred = {
            "Nodules": [
                {"Series ID": 1, "Image ID": 2, "Comparison Date": "3/1/2022"}
            ]
        }
        metrics = score_case(gold, pred)
        self.assertGreater(metrics["FP"], 0)
        self.assertGreater(metrics["FN"], 0)


if __name__ == "__main__":
    unittest.main()
