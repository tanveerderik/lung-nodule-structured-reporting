#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from prompt_policy import classify_instruction, summarize_instructions  # noqa: E402


SPARSE = (
    "Omit any key whose value is not explicitly stated. "
    "Do not output JSON null. Any direction means to omit that key from the final JSON."
)
DENSE = (
    "Use the complete dense template. output the JSON literal null for that key. "
    "Do not invent a non-null value merely to fill the dense template."
)


class PromptPolicyTests(unittest.TestCase):
    def test_sparse_policy(self):
        self.assertEqual(classify_instruction(SPARSE), "sparse_omit_null")

    def test_dense_policy(self):
        self.assertEqual(classify_instruction(DENSE), "dense_nullable")

    def test_conflicting_policy(self):
        self.assertEqual(
            classify_instruction(SPARSE + " " + DENSE),
            "conflicting_sparse_and_dense",
        )

    def test_instruction_hash_provenance(self):
        result = summarize_instructions([DENSE, DENSE])
        self.assertEqual(result["policy_counts"], {"dense_nullable": 2})
        self.assertEqual(result["unique_instruction_count"], 1)
        self.assertEqual(len(result["instruction_sha256"]), 1)


if __name__ == "__main__":
    unittest.main()
