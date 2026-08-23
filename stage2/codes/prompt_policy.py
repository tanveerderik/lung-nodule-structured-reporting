#!/usr/bin/env python3
"""Instruction-policy classification and provenance helpers."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable


SPARSE_MARKERS = (
    "Omit any key whose value is not explicitly stated",
    "Do not output JSON null",
    "means to omit that key from the final JSON",
)
DENSE_MARKERS = (
    "Use the complete dense template",
    "output the JSON literal null for that key",
    "Do not invent a non-null value merely to fill the dense template",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_instruction(text: str) -> str:
    sparse = all(marker in text for marker in SPARSE_MARKERS)
    dense = all(marker in text for marker in DENSE_MARKERS)
    if sparse and not dense:
        return "sparse_omit_null"
    if dense and not sparse:
        return "dense_nullable"
    if sparse and dense:
        return "conflicting_sparse_and_dense"
    return "unrecognized"


def summarize_instructions(instructions: Iterable[str]) -> dict[str, object]:
    policies: Counter[str] = Counter()
    hashes: set[str] = set()
    count = 0
    for instruction in instructions:
        if not isinstance(instruction, str):
            instruction = ""
        policies[classify_instruction(instruction)] += 1
        hashes.add(sha256_text(instruction))
        count += 1
    return {
        "records": count,
        "policy_counts": dict(policies),
        "unique_instruction_count": len(hashes),
        "instruction_sha256": sorted(hashes),
    }
