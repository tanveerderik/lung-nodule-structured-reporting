#!/usr/bin/env python3
"""Recompute saved case/summary metrics without running model inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from date_normalization import DATE_SCORING_POLICY
from nodule_scoring import SEMANTIC_NULL_ALIASES, score_case
from prediction_normalization import normalize_prediction, parse_json_object
from schema_validation import validate_raw_prediction


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def create_backup_once(path: Path) -> Path:
    backup = path.with_name(path.name + ".pre_null_alias_rebuild.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def aggregate(cases: list[dict[str, Any]]) -> dict[str, float]:
    tp = sum(float((case.get("Eval Metrics") or {}).get("TP", 0.0)) for case in cases)
    fp = sum(float((case.get("Eval Metrics") or {}).get("FP", 0.0)) for case in cases)
    fn = sum(float((case.get("Eval Metrics") or {}).get("FN", 0.0)) for case in cases)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_cases", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--schema_file", required=True)
    parser.add_argument("--arm_label", required=True)
    parser.add_argument(
        "--raw_validation_mode",
        choices=("json_schema", "dynamic_template"),
        required=True,
        help=(
            "Ordinary Base uses the typed JSON Schema. Authentic Base+DC uses "
            "its saved pre-normalization dynamic-template validation."
        ),
    )
    parser.add_argument(
        "--template_style",
        choices=("canonical", "legacy_snake_case"),
        default="canonical",
    )
    parser.add_argument("--min_similarity", type=float, default=0.20)
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Required acknowledgement that case metrics and summary will be updated.",
    )
    args = parser.parse_args()
    if not args.in_place:
        parser.error("--in_place is required")

    cases_path = Path(args.eval_cases)
    summary_path = Path(args.summary_json)
    schema_path = Path(args.schema_file)
    cases = load_json(cases_path)
    summary = load_json(summary_path)
    schema = load_json(schema_path)
    if not isinstance(cases, list) or not isinstance(summary, dict):
        raise SystemExit("ERROR: expected a case list and summary object")
    if not isinstance(schema, dict):
        raise SystemExit(f"ERROR: schema is not an object: {schema_path}")

    changed_predictions = 0
    changed_metrics = 0
    raw_schema_valid = 0
    raw_parse_invalid = 0
    for case in cases:
        case_id = case.get("ID")
        if "Parsed Prediction Dense" not in case:
            raise ValueError(
                f"{args.arm_label}: case ID {case_id!r} is missing "
                "Parsed Prediction Dense at $.Parsed Prediction Dense"
            )
        parsed_dense = case.get("Parsed Prediction Dense")
        if not isinstance(parsed_dense, dict):
            raise ValueError(
                f"{args.arm_label}: case ID {case_id!r} has a non-object "
                "Parsed Prediction Dense at $.Parsed Prediction Dense"
            )

        untouched_dense = deepcopy(parsed_dense)
        reparsed_raw = parse_json_object(case.get("Raw Prediction"))
        raw_parse_matches_saved = reparsed_raw is not None and reparsed_raw == parsed_dense
        if args.raw_validation_mode == "dynamic_template":
            native_validation = case.get("Dynamic Constraint Validation")
            if not isinstance(native_validation, dict) or not isinstance(
                native_validation.get("valid"), bool
            ):
                raise ValueError(
                    f"{args.arm_label}: case ID {case_id!r} is missing authoritative "
                    "Dynamic Constraint Validation at $.Dynamic Constraint Validation"
                )
            native_errors = list(native_validation.get("errors") or [])
            raw_validation = {
                "valid": bool(native_validation["valid"]),
                "error_count": len(native_errors),
                "errors": [
                    {"path": "$", "message": str(error), "validator": "dynamic_template"}
                    for error in native_errors
                ],
                "contract": "authentic_dynamic_template",
            }
        else:
            raw_validation = validate_raw_prediction(parsed_dense, schema)
            raw_validation["contract"] = "typed_json_schema"
        if not raw_parse_matches_saved:
            raw_parse_invalid += 1
            raw_validation = {
                "valid": False,
                "error_count": int(raw_validation["error_count"]) + 1,
                "errors": [
                    {
                        "path": "$",
                        "message": (
                            "Raw Prediction could not be parsed as the saved object"
                        ),
                    }
                ]
                + list(raw_validation["errors"]),
            }
        raw_schema_valid += int(raw_validation["valid"])
        case["Raw Schema Validation"] = raw_validation

        prediction = normalize_prediction(
            parsed_dense, template_style=args.template_style
        )
        if parsed_dense != untouched_dense:
            raise RuntimeError(
                f"{args.arm_label}: case ID {case_id!r}: normalization mutated "
                "Parsed Prediction Dense at $"
            )
        if prediction != case.get("Prediction"):
            changed_predictions += 1
        case["Prediction"] = prediction

        gold = normalize_prediction(case.get("Ground Truth"))
        metrics = score_case(gold, prediction, min_similarity=args.min_similarity)
        if metrics != case.get("Eval Metrics"):
            changed_metrics += 1
        case["Eval Metrics"] = metrics

    summary.update(aggregate(cases))
    summary["Date Scoring Normalization"] = DATE_SCORING_POLICY
    summary["Prediction Normalization"] = (
        "recursive semantic omission using case-insensitive aliases: empty string, "
        "null, none, N/A; canonical type coercion; count-zero sparse normalization"
    )
    summary["Semantic Null Aliases"] = sorted(SEMANTIC_NULL_ALIASES)
    summary["Predictions Rebuilt From Parsed Prediction Dense"] = True
    summary["Metrics Rescored From Rebuilt Predictions"] = True
    summary["Casewise Incorrect-Value Scoring Policy"] = (
        "preserved historical partial credit: incorrect non-null matched field "
        "adds TP=1, FP=0.5, FN=0.5; correct field adds TP=2"
    )
    summary["Raw Schema Validation"] = {
        "schema_file": str(schema_path.expanduser().resolve()),
        "valid_outputs": raw_schema_valid,
        "invalid_outputs": len(cases) - raw_schema_valid,
        "validity_rate": raw_schema_valid / len(cases) if cases else 0.0,
        "raw_parse_or_saved_object_mismatch_count": raw_parse_invalid,
        "evaluated_before_semantic_normalization": True,
        "contract": (
            "authentic_dynamic_template"
            if args.raw_validation_mode == "dynamic_template"
            else "typed_json_schema"
        ),
    }

    case_backup = create_backup_once(cases_path)
    summary_backup = create_backup_once(summary_path)
    atomic_json_dump(cases_path, cases)
    atomic_json_dump(summary_path, summary)
    print(
        f"{args.arm_label}: rebuilt {len(cases)} predictions "
        f"({changed_predictions} changed; {changed_metrics} metrics changed); "
        f"raw schema validity={raw_schema_valid}/{len(cases)}; "
        f"F1={summary['F1']:.12f}\n"
        f"Backups: {case_backup}, {summary_backup}"
    )


if __name__ == "__main__":
    main()
