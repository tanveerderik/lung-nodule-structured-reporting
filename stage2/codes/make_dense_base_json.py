#!/usr/bin/env python3
"""Create dense-null-compatible copies of one or more Base JSON datasets.

This script changes only each record's ``instruction`` field. It preserves the
report input, reference output, IDs, and every other field exactly. The output
files are intended for the controlled unadapted Base versus Base+DC inference
experiment. They are not SFT training files unless their target outputs are also
converted to the dense representation.

Typical use from the project root when only the Base test set exists::

    python codes/make_dense_base_json.py \
        --test datasets/test_nodule_base.json

This creates ``test_nodule_base_dense.json`` beside the source file. If a
separate Base training dataset exists, ``--train`` may also be supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SPARSE_NULL_POLICY = (
    "Omit any key whose value is not explicitly stated or cannot be derived "
    "under the rules below. Do not output JSON null and do not output the "
    "string \"null\". Throughout the remaining instructions, any direction "
    "to replace or keep a value as null means to omit that key from the final "
    "JSON."
)

DENSE_NULL_POLICY = (
    "Use the complete dense template. Each generated nodule dictionary must "
    "include every nodule-level key shown above, and the output must include "
    "all four report-level keys: \"Overall Lung-RADS\", \"Recommend Imaging\", "
    "\"Imaging Interval\", and \"Follow-up Date\". When a value is not "
    "explicitly stated and cannot be derived under the rules below, output "
    "the JSON literal null for that key. Do not invent a non-null value merely "
    "to fill the dense template. Use JSON null, not the string \"null\"."
)

SPARSE_NODULE_DICTIONARY_POLICY = (
    "In the \"Nodules\" list, each dictionary represents an individual nodule "
    "and may contain only the above-defined keys."
)

DENSE_NODULE_DICTIONARY_POLICY = (
    "In the \"Nodules\" list, each dictionary represents an individual nodule "
    "and must contain every above-defined nodule-level key exactly once."
)

SPARSE_NO_NODULE_POLICY = (
    "If the report explicitly states that no lung nodule is present, omit "
    "\"Nodules\" and return {\"Number of Nodules\": 0}. Also include \"Overall "
    "Lung-RADS\", \"Recommend Imaging\", \"Imaging Interval\", or \"Follow-up "
    "Date\" only when explicitly reported. If the report contains no "
    "nodule-related information at all, return {}."
)

DENSE_NO_NODULE_POLICY = (
    "If the report explicitly states that no lung nodule is present, or if the "
    "report contains no nodule-related information, set \"Number of Nodules\" "
    "to 0 and omit the \"Nodules\" array. Still include \"Overall Lung-RADS\", "
    "\"Recommend Imaging\", \"Imaging Interval\", and \"Follow-up Date\"; use "
    "the JSON literal null for any of these report-level values that is not "
    "explicitly reported."
)

REPLACEMENTS = (
    ("sparse null policy", SPARSE_NULL_POLICY, DENSE_NULL_POLICY),
    (
        "optional nodule-dictionary policy",
        SPARSE_NODULE_DICTIONARY_POLICY,
        DENSE_NODULE_DICTIONARY_POLICY,
    ),
    ("sparse no-nodule policy", SPARSE_NO_NODULE_POLICY, DENSE_NO_NODULE_POLICY),
)

FORBIDDEN_AFTER_TRANSFORM = (
    "Do not output JSON null",
    "means to omit that key from the final JSON",
    "return {}",
    "may contain only the above-defined keys",
)

REQUIRED_AFTER_TRANSFORM = (
    "Use the complete dense template",
    "must contain every above-defined nodule-level key exactly once",
    "output the JSON literal null for that key",
    "Do not invent a non-null value merely to fill the dense template",
    "set \"Number of Nodules\" to 0 and omit the \"Nodules\" array",
)


def make_dense_instruction(instruction: str) -> str:
    """Replace the three sparse-only policies and validate the result."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction is missing or empty")

    # Permit safe re-execution on an already converted instruction.
    already_dense = all(text in instruction for text in REQUIRED_AFTER_TRANSFORM)
    if already_dense:
        transformed = instruction
    else:
        transformed = instruction
        for label, old, new in REPLACEMENTS:
            occurrences = transformed.count(old)
            if occurrences != 1:
                raise ValueError(
                    f"Expected exactly one {label}; found {occurrences}. "
                    "The source instruction differs from the audited Base prompt, "
                    "so no output should be trusted until it is reviewed."
                )
            transformed = transformed.replace(old, new, 1)

    forbidden = [text for text in FORBIDDEN_AFTER_TRANSFORM if text in transformed]
    missing = [text for text in REQUIRED_AFTER_TRANSFORM if text not in transformed]
    if forbidden or missing:
        raise ValueError(
            "Dense instruction validation failed. "
            f"Forbidden remnants={forbidden}; missing dense clauses={missing}"
        )
    return transformed


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty top-level JSON array")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError(f"{path}: every array element must be a JSON object")
    return data


def output_path_for(source: Path, output_dir: Path | None) -> Path:
    destination_dir = output_dir if output_dir is not None else source.parent
    if source.stem.endswith("_dense"):
        filename = source.name
    else:
        filename = f"{source.stem}_dense{source.suffix}"
    return destination_dir / filename


def atomic_json_dump(data: Any, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {destination}. Pass --overwrite intentionally."
        )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def convert_file(
    source: Path,
    destination: Path,
    *,
    instruction_field: str,
    overwrite: bool,
) -> tuple[int, int]:
    records = load_records(source)
    transformed_cache: dict[str, str] = {}

    # Snapshot every non-instruction field so conversion can be verified before
    # anything is written.
    untouched_snapshots = [
        {key: value for key, value in record.items() if key != instruction_field}
        for record in records
    ]

    for index, record in enumerate(records):
        instruction = record.get(instruction_field)
        if not isinstance(instruction, str):
            raise ValueError(
                f"{source}: record {index} has no string {instruction_field!r} field"
            )
        if instruction not in transformed_cache:
            transformed_cache[instruction] = make_dense_instruction(instruction)
        record[instruction_field] = transformed_cache[instruction]

    for index, (record, expected) in enumerate(zip(records, untouched_snapshots)):
        actual = {
            key: value for key, value in record.items() if key != instruction_field
        }
        if actual != expected:
            raise AssertionError(
                f"Internal safety check failed: non-instruction content changed "
                f"in {source}, record {index}"
            )

    atomic_json_dump(records, destination, overwrite=overwrite)

    # Verify the saved file, including preservation of all reference outputs.
    saved = load_records(destination)
    if len(saved) != len(records):
        raise AssertionError(f"{destination}: saved record count changed")
    for index, (record, expected) in enumerate(zip(saved, records)):
        if record != expected:
            raise AssertionError(
                f"{destination}: saved content mismatch at record {index}"
            )

    return len(records), len(transformed_cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create *_dense.json copies of Base datasets by replacing "
            "the sparse omit-null instruction with the dense JSON-null policy."
        )
    )
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional common output directory; default is beside each input file.",
    )
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace existing *_dense.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train is None and args.test is None:
        raise ValueError("Provide at least one input: --test and/or --train")

    sources = [
        path.expanduser().resolve()
        for path in (args.train, args.test)
        if path is not None
    ]
    if len(sources) != len(set(sources)):
        raise ValueError("--train and --test must identify different files")

    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_path_for(source, args.output_dir)
        destination = destination.expanduser().resolve()
        if destination == source:
            raise ValueError(f"Source and destination resolve to the same file: {source}")

        record_count, unique_instruction_count = convert_file(
            source,
            destination,
            instruction_field=args.instruction_field,
            overwrite=args.overwrite,
        )
        print(
            f"Wrote {destination} ({record_count} records; "
            f"{unique_instruction_count} unique instruction(s) converted)."
        )


if __name__ == "__main__":
    main()
