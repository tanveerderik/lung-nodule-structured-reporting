#!/usr/bin/env python3
"""Audit a controlled ordinary-SFT versus sparse-XGrammar evaluation pair."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def case_key(case: dict[str, Any]) -> str:
    return json.dumps(case.get("ID"), sort_keys=True, ensure_ascii=False)


def index_cases(cases: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for case in cases:
        key = case_key(case)
        if key in indexed:
            raise ValueError(f"{label} contains duplicate ID {case.get('ID')!r}")
        indexed[key] = case
    return indexed


def parse_tagged_json(raw: Any, begin_tag: str, end_tag: str) -> dict[str, Any]:
    text = raw if isinstance(raw, str) else ""
    stripped = text.strip()
    prefix = begin_tag + "\n"
    suffix = "\n" + end_tag
    exact_tags = stripped.startswith(prefix) and stripped.endswith(suffix)

    body = stripped
    if exact_tags:
        body = stripped[len(prefix) : -len(suffix)]
    elif begin_tag in stripped and end_tag in stripped:
        body = stripped.split(begin_tag, 1)[1].rsplit(end_tag, 1)[0].strip()

    try:
        parsed = json.loads(body)
        json_valid = isinstance(parsed, dict)
    except Exception:
        parsed = None
        json_valid = False

    return {
        "exact_tags": exact_tags,
        "json_valid": json_valid,
        "object": parsed if json_valid else None,
    }


def count_nulls(value: Any) -> int:
    if value is None or (isinstance(value, str) and value.strip().lower() == "null"):
        return 1
    if isinstance(value, dict):
        return sum(count_nulls(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_nulls(item) for item in value)
    return 0


def count_list_status(value: Any) -> str:
    if not isinstance(value, dict):
        return "uncheckable"
    count = value.get("Number of Nodules")
    nodules = value.get("Nodules")
    if count is None and nodules is None:
        return "both_omitted"
    if isinstance(count, int) and not isinstance(count, bool) and isinstance(nodules, list):
        return "consistent" if count == len(nodules) else "inconsistent"
    return "uncheckable"


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate the JSON-Schema subset used by the lung-nodule schema."""
    if "anyOf" in schema:
        branch_errors = [validate_schema(value, branch, path) for branch in schema["anyOf"]]
        if any(not errors for errors in branch_errors):
            return []
        return [f"{path}: value does not satisfy anyOf"]

    expected = schema.get("type")
    type_valid = {
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }.get(expected, True)
    if not type_valid:
        return [f"{path}: expected {expected}, got {type(value).__name__}"]

    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value is above maximum")
    if isinstance(value, str) and "pattern" in schema:
        if re.fullmatch(schema["pattern"], value) is None:
            errors.append(f"{path}: string does not match pattern")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: array is shorter than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: array is longer than maxItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(validate_schema(item, item_schema, f"{path}[{index}]"))
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required property {required!r}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: additional property {key!r}")
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_schema(item, properties[key], f"{path}.{key}"))
    return errors


def first_token_difference(tokenizer: Any, left: str, right: str) -> dict[str, Any] | None:
    if tokenizer is None:
        limit = min(len(left), len(right))
        index = next((i for i in range(limit) if left[i] != right[i]), limit)
        if index == len(left) == len(right):
            return None
        return {
            "comparison_unit": "character (transformers unavailable)",
            "character_index": index,
            "ordinary": left[index] if index < len(left) else None,
            "xgrammar": right[index] if index < len(right) else None,
        }

    left_ids = tokenizer(left, add_special_tokens=False)["input_ids"]
    right_ids = tokenizer(right, add_special_tokens=False)["input_ids"]
    limit = min(len(left_ids), len(right_ids))
    index = next((i for i in range(limit) if left_ids[i] != right_ids[i]), limit)
    if index == len(left_ids) == len(right_ids):
        return None

    def describe(ids: list[int]) -> dict[str, Any] | None:
        if index >= len(ids):
            return None
        token_id = ids[index]
        return {
            "token_id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "decoded": tokenizer.decode([token_id]),
        }

    return {
        "comparison_unit": "token",
        "token_index": index,
        "ordinary": describe(left_ids),
        "xgrammar": describe(right_ids),
        "ordinary_token_count": len(left_ids),
        "xgrammar_token_count": len(right_ids),
    }


def summarize_arm(
    cases: list[dict[str, Any]],
    schema: dict[str, Any],
    begin_tag: str,
    end_tag: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    details: dict[str, dict[str, Any]] = {}
    exact_tags = json_valid = schema_valid = nulls = 0
    count_status = {
        "consistent": 0,
        "inconsistent": 0,
        "both_omitted": 0,
        "uncheckable": 0,
    }
    completion_tokens = 0

    for case in cases:
        key = case_key(case)
        parsed = parse_tagged_json(case.get("Raw Prediction"), begin_tag, end_tag)
        obj = parsed["object"]
        errors = [] if obj is None else validate_schema(obj, schema)
        is_schema_valid = obj is not None and not errors
        status = count_list_status(obj)
        token_value = (case.get("Token Usage") or {}).get("Completion Tokens")

        exact_tags += int(parsed["exact_tags"])
        json_valid += int(parsed["json_valid"])
        schema_valid += int(is_schema_valid)
        nulls += count_nulls(obj)
        count_status[status] += 1
        if isinstance(token_value, int):
            completion_tokens += token_value

        details[key] = {
            "parsed": parsed,
            "schema_valid": is_schema_valid,
            "schema_errors": errors[:5],
            "count_list_status": status,
        }

    total = len(cases)
    return (
        {
            "cases": total,
            "exact_tagged_outputs": exact_tags,
            "valid_json_objects": json_valid,
            "schema_valid_outputs": schema_valid,
            "null_selections": nulls,
            "count_list_status": count_status,
            "completion_tokens_total": completion_tokens,
        },
        details,
    )


def micro_f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def summarize_featurewise(
    ordinary_path: Path | None,
    xgrammar_path: Path | None,
) -> dict[str, Any] | None:
    if ordinary_path is None or xgrammar_path is None:
        return None
    ordinary = {row["feature"]: row for row in load_json(ordinary_path)}
    xgrammar = {row["feature"]: row for row in load_json(xgrammar_path)}
    if set(ordinary) != set(xgrammar):
        raise ValueError("Featurewise files do not contain identical feature sets")

    features = sorted(ordinary)
    ordinary_macro = sum(float(ordinary[name]["f1"]) for name in features) / len(features)
    xgrammar_macro = sum(float(xgrammar[name]["f1"]) for name in features) / len(features)

    ordinary_counts = {
        key: sum(float(ordinary[name][key]) for name in features)
        for key in ("TP", "FP", "FN")
    }
    xgrammar_counts = {
        key: sum(float(xgrammar[name][key]) for name in features)
        for key in ("TP", "FP", "FN")
    }
    deltas = {
        name: float(xgrammar[name]["f1"]) - float(ordinary[name]["f1"])
        for name in features
    }

    return {
        "feature_count": len(features),
        "ordinary_macro_f1": ordinary_macro,
        "xgrammar_macro_f1": xgrammar_macro,
        "macro_f1_delta": xgrammar_macro - ordinary_macro,
        "ordinary_micro_f1": micro_f1(**{k.lower(): v for k, v in ordinary_counts.items()}),
        "xgrammar_micro_f1": micro_f1(**{k.lower(): v for k, v in xgrammar_counts.items()}),
        "features_improved": sum(delta > 1e-12 for delta in deltas.values()),
        "features_worsened": sum(delta < -1e-12 for delta in deltas.values()),
        "features_unchanged": sum(abs(delta) <= 1e-12 for delta in deltas.values()),
        "largest_absolute_changes": [
            {"feature": name, "f1_delta": deltas[name]}
            for name in sorted(deltas, key=lambda item: abs(deltas[item]), reverse=True)[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary_cases", required=True, type=Path)
    parser.add_argument("--xgrammar_cases", required=True, type=Path)
    parser.add_argument("--ordinary_summary", required=True, type=Path)
    parser.add_argument("--xgrammar_summary", required=True, type=Path)
    parser.add_argument("--ordinary_repeat_cases", type=Path)
    parser.add_argument("--ordinary_repeat_summary", type=Path)
    parser.add_argument("--ordinary_featurewise", type=Path)
    parser.add_argument("--xgrammar_featurewise", type=Path)
    parser.add_argument("--schema_file", required=True, type=Path)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--json_begin_tag", default="<json>")
    parser.add_argument("--json_end_tag", default="</json>")
    parser.add_argument("--output_json", required=True, type=Path)
    args = parser.parse_args()

    ordinary_cases = load_json(args.ordinary_cases)
    xgrammar_cases = load_json(args.xgrammar_cases)
    ordinary_summary = load_json(args.ordinary_summary)
    xgrammar_summary = load_json(args.xgrammar_summary)
    schema = load_json(args.schema_file)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, trust_remote_code=True
        )
    except ImportError:
        tokenizer = None

    ordinary_by_id = index_cases(ordinary_cases, "ordinary cases")
    xgrammar_by_id = index_cases(xgrammar_cases, "XGrammar cases")
    if set(ordinary_by_id) != set(xgrammar_by_id):
        raise ValueError("Ordinary and XGrammar outputs contain different case IDs")

    ordinary_arm, ordinary_details = summarize_arm(
        ordinary_cases, schema, args.json_begin_tag, args.json_end_tag
    )
    xgrammar_arm, xgrammar_details = summarize_arm(
        xgrammar_cases, schema, args.json_begin_tag, args.json_end_tag
    )

    changed_raw = []
    changed_predictions = 0
    prompt_token_mismatches = []
    for key in ordinary_by_id:
        ordinary_case = ordinary_by_id[key]
        xgrammar_case = xgrammar_by_id[key]
        if ordinary_case.get("Prediction") != xgrammar_case.get("Prediction"):
            changed_predictions += 1
        ordinary_prompt_tokens = (ordinary_case.get("Token Usage") or {}).get("Prompt Tokens")
        xgrammar_prompt_tokens = (xgrammar_case.get("Token Usage") or {}).get("Prompt Tokens")
        if ordinary_prompt_tokens != xgrammar_prompt_tokens:
            prompt_token_mismatches.append(ordinary_case.get("ID"))
        ordinary_raw = ordinary_case.get("Raw Prediction") or ""
        xgrammar_raw = xgrammar_case.get("Raw Prediction") or ""
        if ordinary_raw != xgrammar_raw:
            changed_raw.append({
                "ID": ordinary_case.get("ID"),
                "ordinary_schema_valid": ordinary_details[key]["schema_valid"],
                "xgrammar_schema_valid": xgrammar_details[key]["schema_valid"],
                "first_token_difference": first_token_difference(
                    tokenizer, ordinary_raw, xgrammar_raw
                ),
            })

    controlled_fields = [
        "Eval Context Limit",
        "Max New Tokens",
        "Max Prompt Tokens",
        "Completion Add Special Tokens",
        "Request Sampling Configuration",
        "Date Scoring Normalization",
    ]
    config_mismatches = {
        field: {
            "ordinary": ordinary_summary.get(field),
            "xgrammar": xgrammar_summary.get(field),
        }
        for field in controlled_fields
        if ordinary_summary.get(field) != xgrammar_summary.get(field)
    }

    repeat_result = None
    repeat_valid = True
    if args.ordinary_repeat_cases is not None:
        repeat_cases = load_json(args.ordinary_repeat_cases)
        repeat_by_id = index_cases(repeat_cases, "ordinary repeat cases")
        if set(repeat_by_id) != set(ordinary_by_id):
            raise ValueError("Ordinary repeat contains different case IDs")
        repeat_changes = [
            ordinary_by_id[key].get("ID")
            for key in ordinary_by_id
            if ordinary_by_id[key].get("Raw Prediction")
            != repeat_by_id[key].get("Raw Prediction")
        ]
        repeat_summary_mismatches = {}
        if args.ordinary_repeat_summary is not None:
            repeat_summary = load_json(args.ordinary_repeat_summary)
            repeat_summary_mismatches = {
                field: {
                    "first": ordinary_summary.get(field),
                    "repeat": repeat_summary.get(field),
                }
                for field in controlled_fields
                if ordinary_summary.get(field) != repeat_summary.get(field)
            }
        repeat_valid = not repeat_changes and not repeat_summary_mismatches
        repeat_result = {
            "identical_raw_outputs": len(ordinary_cases) - len(repeat_changes),
            "changed_raw_outputs": len(repeat_changes),
            "changed_ids": repeat_changes[:50],
            "configuration_mismatches": repeat_summary_mismatches,
            "reproducible": repeat_valid,
        }

    xgrammar_guarantees_hold = (
        xgrammar_arm["exact_tagged_outputs"] == len(xgrammar_cases)
        and xgrammar_arm["valid_json_objects"] == len(xgrammar_cases)
        and xgrammar_arm["schema_valid_outputs"] == len(xgrammar_cases)
    )
    controlled_valid = (
        not config_mismatches
        and not prompt_token_mismatches
        and repeat_valid
        and xgrammar_guarantees_hold
    )

    report = {
        "comparison": "ordinary SFT vs sparse XGrammar",
        "controlled_comparison_valid": controlled_valid,
        "configuration_mismatches": config_mismatches,
        "prompt_token_mismatch_count": len(prompt_token_mismatches),
        "prompt_token_mismatch_ids": prompt_token_mismatches[:50],
        "ordinary_repeat": repeat_result,
        "ordinary": {
            **ordinary_arm,
            "summary_f1": ordinary_summary.get("F1"),
            "summary_precision": ordinary_summary.get("Precision"),
            "summary_recall": ordinary_summary.get("Recall"),
        },
        "xgrammar": {
            **xgrammar_arm,
            "summary_f1": xgrammar_summary.get("F1"),
            "summary_precision": xgrammar_summary.get("Precision"),
            "summary_recall": xgrammar_summary.get("Recall"),
        },
        "delta": {
            "summary_f1": (xgrammar_summary.get("F1") or 0.0)
            - (ordinary_summary.get("F1") or 0.0),
            "completion_tokens_total": xgrammar_arm["completion_tokens_total"]
            - ordinary_arm["completion_tokens_total"],
        },
        "identical_raw_outputs": len(ordinary_cases) - len(changed_raw),
        "changed_raw_outputs": len(changed_raw),
        "changed_normalized_predictions": changed_predictions,
        "changed_case_details": changed_raw,
        "featurewise": summarize_featurewise(
            args.ordinary_featurewise, args.xgrammar_featurewise
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not controlled_valid:
        raise SystemExit(
            "Controlled-comparison validation failed; do not use these results in the paper."
        )


if __name__ == "__main__":
    main()
