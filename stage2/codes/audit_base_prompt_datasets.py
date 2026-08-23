#!/usr/bin/env python3
"""Validate sparse and dense Base prompt datasets before inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prompt_policy import classify_instruction, sha256_text


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON array")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path}: every record must be a JSON object")
    return data


def without_instruction(record: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != field}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    sparse_path = args.sparse.expanduser().resolve()
    dense_path = args.dense.expanduser().resolve()
    sparse = load_records(sparse_path)
    dense = load_records(dense_path)

    if len(sparse) != len(dense):
        raise ValueError(
            f"Record-count mismatch: sparse={len(sparse)}, dense={len(dense)}"
        )

    sparse_policies: Counter[str] = Counter()
    dense_policies: Counter[str] = Counter()
    sparse_hashes: set[str] = set()
    dense_hashes: set[str] = set()
    altered_non_instruction_records: list[int] = []

    for index, (sparse_record, dense_record) in enumerate(zip(sparse, dense)):
        sparse_instruction = sparse_record.get(args.instruction_field)
        dense_instruction = dense_record.get(args.instruction_field)
        if not isinstance(sparse_instruction, str):
            raise ValueError(f"Sparse record {index} has no string instruction")
        if not isinstance(dense_instruction, str):
            raise ValueError(f"Dense record {index} has no string instruction")

        sparse_policies[classify_instruction(sparse_instruction)] += 1
        dense_policies[classify_instruction(dense_instruction)] += 1
        sparse_hashes.add(sha256_text(sparse_instruction))
        dense_hashes.add(sha256_text(dense_instruction))

        if without_instruction(sparse_record, args.instruction_field) != without_instruction(
            dense_record, args.instruction_field
        ):
            altered_non_instruction_records.append(index)

    valid = (
        set(sparse_policies) == {"sparse_omit_null"}
        and set(dense_policies) == {"dense_nullable"}
        and not altered_non_instruction_records
        and sparse_hashes.isdisjoint(dense_hashes)
    )
    report = {
        "valid": valid,
        "sparse_file": str(sparse_path),
        "dense_file": str(dense_path),
        "records": len(sparse),
        "sparse_instruction_policies": dict(sparse_policies),
        "dense_instruction_policies": dict(dense_policies),
        "sparse_unique_instruction_hashes": sorted(sparse_hashes),
        "dense_unique_instruction_hashes": sorted(dense_hashes),
        "altered_non_instruction_record_count": len(altered_non_instruction_records),
        "altered_non_instruction_record_indices": altered_non_instruction_records[:50],
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not valid:
        raise SystemExit("Sparse/dense prompt dataset audit failed")


if __name__ == "__main__":
    main()
