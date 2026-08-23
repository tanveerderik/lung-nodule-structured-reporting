#!/usr/bin/env python3
"""Audit and summarize sparse/dense Base prompt regimes.

Required valid arms
-------------------
1. ordinary Base with the sparse omit-null prompt (existing results)
2. ordinary Base with the dense nullable prompt (new dense results)
3. Base+DC with the same dense nullable prompt (new dense results)

If an old Base+DC sparse-prompt result is present, it is reported only as an
incompatible diagnostic and is excluded from the valid-effect calculations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from nodule_scoring import is_null
from prediction_normalization import parse_json_object
from schema_validation import validate_raw_prediction


DEFAULT_MODELS = (
    "llama3_2_1B",
    "gemma3_4B",
    "mistral_7B",
    "qwen2_5_7B",
    "llama3_1_8B",
    "llama3_1_70B",
)

ORDINARY_CASES = "base_eval_cases_controlled_ordinary.json"
ORDINARY_SUMMARY = "base_eval_summary_controlled_ordinary.json"
DC_CASES = "base_eval_cases_controlled_dynamic_template.json"
DC_SUMMARY = "base_eval_summary_controlled_dynamic_template.json"
PAIR_AUDIT = "base_eval_controlled_pair_audit.json"

def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def case_key(case: dict[str, Any]) -> str:
    return json.dumps(case.get("ID"), sort_keys=True, ensure_ascii=False)


def index_cases(cases: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{label}: cases are empty or malformed")
    indexed: dict[str, dict[str, Any]] = {}
    for case in cases:
        key = case_key(case)
        if key in indexed:
            raise ValueError(f"{label}: duplicate ID {case.get('ID')!r}")
        indexed[key] = case
    return indexed


def normalized_null_paths(value: Any, path: str = "$", *, root: bool = True) -> list[str]:
    violations: list[str] = []
    if is_null(value) and not (root and isinstance(value, (dict, list))):
        return [path]
    if isinstance(value, dict):
        if not root and not value:
            return [path]
        for key, item in value.items():
            violations.extend(
                normalized_null_paths(item, f"{path}.{key}", root=False)
            )
    elif isinstance(value, list):
        if not value:
            return [path]
        for index, item in enumerate(value):
            violations.extend(
                normalized_null_paths(item, f"{path}[{index}]", root=False)
            )
    return violations


def count_raw_leaves(value: Any) -> tuple[int, int]:
    """Return (null-like leaves, all scalar leaves) for a parsed raw object."""
    if isinstance(value, dict):
        nulls = leaves = 0
        for item in value.values():
            child_nulls, child_leaves = count_raw_leaves(item)
            nulls += child_nulls
            leaves += child_leaves
        return nulls, leaves
    if isinstance(value, list):
        nulls = leaves = 0
        for item in value:
            child_nulls, child_leaves = count_raw_leaves(item)
            nulls += child_nulls
            leaves += child_leaves
        return nulls, leaves
    return (1 if is_null(value) else 0), 1


def validate_normalization(
    cases: list[dict[str, Any]],
    label: str,
    schema: dict[str, Any],
    raw_validation_mode: str,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    raw_nulls = raw_leaves = 0
    raw_schema_invalid: list[dict[str, Any]] = []
    for case in cases:
        paths = normalized_null_paths(case.get("Prediction"), root=True)
        if paths:
            violations.append({"id": case.get("ID"), "paths": paths[:20]})
        parsed = case.get("Parsed Prediction Dense")
        reparsed = parse_json_object(case.get("Raw Prediction"))
        if raw_validation_mode == "dynamic_template":
            native = case.get("Dynamic Constraint Validation")
            if not isinstance(native, dict) or not isinstance(native.get("valid"), bool):
                raw_validation = {
                    "valid": False,
                    "error_count": 1,
                    "errors": [
                        {
                            "path": "$.Dynamic Constraint Validation",
                            "message": "missing authoritative dynamic-template validation",
                        }
                    ],
                }
            else:
                native_errors = list(native.get("errors") or [])
                raw_validation = {
                    "valid": bool(native["valid"]),
                    "error_count": len(native_errors),
                    "errors": [
                        {"path": "$", "message": str(error), "validator": "dynamic_template"}
                        for error in native_errors
                    ],
                }
        else:
            raw_validation = validate_raw_prediction(parsed, schema)
        if reparsed is None or reparsed != parsed:
            raw_validation = {
                "valid": False,
                "error_count": int(raw_validation["error_count"]) + 1,
                "errors": [
                    {
                        "path": "$",
                        "message": "Raw Prediction does not parse to the saved dense object",
                    }
                ]
                + list(raw_validation["errors"]),
            }
        if not raw_validation["valid"]:
            raw_schema_invalid.append(
                {
                    "id": case.get("ID"),
                    "errors": raw_validation["errors"][:20],
                }
            )
        child_nulls, child_leaves = count_raw_leaves(
            case.get("Parsed Prediction Dense", {})
        )
        raw_nulls += child_nulls
        raw_leaves += child_leaves
    return {
        "label": label,
        "valid": not violations,
        "normalized_null_violation_count": len(violations),
        "normalized_null_violations": violations[:20],
        "raw_null_like_leaves": raw_nulls,
        "raw_scalar_leaves": raw_leaves,
        "raw_null_like_leaf_rate": raw_nulls / raw_leaves if raw_leaves else 0.0,
        "raw_schema_valid_outputs": len(cases) - len(raw_schema_invalid),
        "raw_schema_invalid_outputs": len(raw_schema_invalid),
        "raw_schema_validity_rate": (
            (len(cases) - len(raw_schema_invalid)) / len(cases) if cases else 0.0
        ),
        "raw_schema_invalid_details": raw_schema_invalid[:50],
        "raw_validation_contract": (
            "authentic_dynamic_template"
            if raw_validation_mode == "dynamic_template"
            else "typed_json_schema"
        ),
    }


def same_examples(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    same_ids = list(reference) == list(candidate)
    mismatches: list[Any] = []
    if same_ids:
        for key in reference:
            left = reference[key]
            right = candidate[key]
            if (
                left.get("Input") != right.get("Input")
                or left.get("Ground Truth") != right.get("Ground Truth")
                or left.get("Raw Ground Truth") != right.get("Raw Ground Truth")
            ):
                mismatches.append(left.get("ID"))
    return {
        "label": label,
        "valid": same_ids and not mismatches,
        "same_ids_and_order": same_ids,
        "content_mismatch_count": len(mismatches),
        "content_mismatch_ids": mismatches[:50],
    }


def prompt_token_comparison(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mismatches: list[Any] = []
    missing: list[Any] = []
    if list(left) != list(right):
        return {
            "valid": False,
            "same_ids_and_order": False,
            "mismatch_count": None,
            "missing_count": None,
        }
    for key in left:
        left_tokens = (left[key].get("Token Usage") or {}).get("Prompt Tokens")
        right_tokens = (right[key].get("Token Usage") or {}).get("Prompt Tokens")
        if not isinstance(left_tokens, int) or not isinstance(right_tokens, int):
            missing.append(left[key].get("ID"))
        elif left_tokens != right_tokens:
            mismatches.append(left[key].get("ID"))
    return {
        "valid": not missing and not mismatches,
        "same_ids_and_order": True,
        "mismatch_count": len(mismatches),
        "mismatch_ids": mismatches[:50],
        "missing_count": len(missing),
        "missing_ids": missing[:50],
    }


def metric(summary: dict[str, Any], name: str) -> float:
    value = summary.get(name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Summary has no numeric {name}: {value!r}")
    return float(value)


def token_stat(summary: dict[str, Any], kind: str, stat: str) -> Any:
    return ((summary.get("Token Usage") or {}).get(kind) or {}).get(stat)


def truncation_or_limit_hits(
    cases: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    max_new_tokens = summary.get("Max New Tokens")
    count = 0
    exact_finish_reasons_available = True
    for case in cases:
        reason = case.get("Generation Finish Reason")
        if reason is None:
            exact_finish_reasons_available = False
            completion = (case.get("Token Usage") or {}).get("Completion Tokens")
            hit = (
                isinstance(completion, int)
                and isinstance(max_new_tokens, int)
                and completion >= max_new_tokens
            )
        else:
            hit = reason == "length"
        count += int(hit)
    return {
        "count": count,
        "measurement": (
            "finish_reason_length"
            if exact_finish_reasons_available
            else "completion_limit_hit_proxy_for_cases_without_finish_reason"
        ),
    }


def validate_dense_prompt_provenance(
    model: str,
    dense_base_summary: dict[str, Any],
    dense_dc_summary: dict[str, Any],
    dataset_audit: dict[str, Any],
) -> dict[str, Any]:
    base = dense_base_summary.get("Instruction Provenance") or {}
    dc = dense_dc_summary.get("Instruction Provenance") or {}
    expected_records = dataset_audit.get("records")
    expected_hashes = dataset_audit.get("dense_unique_instruction_hashes")
    expected_basename = Path(str(dataset_audit.get("dense_file", ""))).name

    checks = {
        "base_policy_is_dense_nullable": base.get("policy_counts")
        == {"dense_nullable": expected_records},
        "dc_policy_is_dense_nullable": dc.get("policy_counts")
        == {"dense_nullable": expected_records},
        "base_hash_matches_dataset_audit": base.get("instruction_sha256")
        == expected_hashes,
        "dc_hash_matches_dataset_audit": dc.get("instruction_sha256")
        == expected_hashes,
        "base_and_dc_hashes_identical": base.get("instruction_sha256")
        == dc.get("instruction_sha256"),
        "base_test_file_matches_dense_dataset": dense_base_summary.get(
            "Test File Basename"
        )
        == expected_basename,
        "dc_test_file_matches_dense_dataset": dense_dc_summary.get(
            "Test File Basename"
        )
        == expected_basename,
    }
    valid = all(checks.values())
    if not valid:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"{model}: dense prompt provenance failed: {failed}")
    return {"valid": valid, "checks": checks, "base": base, "dynamic": dc}


def validate_dense_pair_audit(
    model: str,
    audit: dict[str, Any],
    dense_base_summary: dict[str, Any],
    dense_dc_summary: dict[str, Any],
) -> dict[str, Any]:
    """Accept only the documented 70B safe-context runtime exception."""
    if audit.get("controlled_experiment_valid") is True:
        return {
            "valid": True,
            "intentional_runtime_difference": (
                audit.get("intentional_runtime_differences") or None
            ),
        }

    mismatches = audit.get("configuration_mismatches") or {}
    allowed_fields = {"Eval Context Limit", "Max Prompt Tokens"}
    expected_contexts = {
        "ordinary": 32768,
        "dynamic_template": 24576,
    }
    expected_max_prompt = {
        "ordinary": 16384,
        "dynamic_template": 8192,
    }
    exception_matches = (
        model == "llama3_1_70B"
        and set(mismatches) == allowed_fields
        and mismatches.get("Eval Context Limit") == expected_contexts
        and mismatches.get("Max Prompt Tokens") == expected_max_prompt
        and dense_base_summary.get("Eval Context Limit") == 32768
        and dense_dc_summary.get("Eval Context Limit") == 24576
        and dense_base_summary.get("Max New Tokens")
        == dense_dc_summary.get("Max New Tokens")
        == 16384
        and audit.get("same_case_ids_and_order") is True
        and audit.get("prompt_token_mismatch_count") == 0
        and audit.get("all_dynamic_outputs_valid") is True
        and (audit.get("ordinary") or {}).get("mode_valid") is True
        and (audit.get("ordinary") or {}).get("sampling_configuration_valid") is True
        and ((audit.get("ordinary") or {}).get("token_validation") or {}).get("valid")
        is True
        and (audit.get("dynamic_template") or {}).get("mode_valid") is True
        and (audit.get("dynamic_template") or {}).get("configuration_valid") is True
        and ((audit.get("dynamic_template") or {}).get("token_validation") or {}).get(
            "valid"
        )
        is True
    )
    if not exception_matches:
        raise ValueError(
            f"{model}: dense prompt-controlled pair audit failed; "
            f"configuration mismatches={mismatches!r}"
        )
    return {
        "valid": True,
        "intentional_runtime_difference": {
            "reason": "documented 70B memory-safe context limits",
            "ordinary_eval_context_limit": 32768,
            "dynamic_eval_context_limit": 24576,
            "scientific_controls_unchanged": True,
        },
    }


def regime_row(
    model: str,
    regime: str,
    interpretation: str,
    valid_for_main_analysis: bool,
    summary: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "regime": regime,
        "interpretation": interpretation,
        "valid_for_main_analysis": valid_for_main_analysis,
        "precision": metric(summary, "Precision"),
        "recall": metric(summary, "Recall"),
        "f1": metric(summary, "F1"),
        "tp": metric(summary, "TP"),
        "fp": metric(summary, "FP"),
        "fn": metric(summary, "FN"),
        "prompt_tokens_mean": token_stat(summary, "Prompt Tokens", "Mean"),
        "completion_tokens_mean": token_stat(summary, "Completion Tokens", "Mean"),
        "total_tokens_mean": token_stat(summary, "Total Tokens", "Mean"),
        "raw_null_like_leaf_rate": normalization["raw_null_like_leaf_rate"],
        "raw_schema_validity_rate": normalization["raw_schema_validity_rate"],
        "raw_schema_invalid_outputs": normalization["raw_schema_invalid_outputs"],
        "normalized_null_violation_count": normalization[
            "normalized_null_violation_count"
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-results", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--schema-file", type=Path, required=True)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model directory names, or 'all'.",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    sparse_root = args.sparse_results.expanduser().resolve()
    dense_root = args.dense_results.expanduser().resolve()
    schema_path = require(args.schema_file.expanduser().resolve())
    schema = load(schema_path)
    if not isinstance(schema, dict):
        raise ValueError(f"Schema is not an object: {schema_path}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dense_root / "base_prompt_regime_comparison"
    )
    models = DEFAULT_MODELS if args.models == "all" else tuple(
        item.strip() for item in args.models.split(",") if item.strip()
    )

    dataset_audit_path = require(dense_root / "base_prompt_dataset_audit.json")
    dataset_audit = load(dataset_audit_path)
    if dataset_audit.get("valid") is not True:
        raise ValueError("Dense prompt dataset audit is not valid")

    all_rows: list[dict[str, Any]] = []
    model_reports: dict[str, Any] = {}

    for model in models:
        sparse_dir = sparse_root / model
        dense_dir = dense_root / model

        sparse_base_cases = load(require(sparse_dir / ORDINARY_CASES))
        sparse_base_summary = load(require(sparse_dir / ORDINARY_SUMMARY))
        dense_base_cases = load(require(dense_dir / ORDINARY_CASES))
        dense_base_summary = load(require(dense_dir / ORDINARY_SUMMARY))
        dense_dc_cases = load(require(dense_dir / DC_CASES))
        dense_dc_summary = load(require(dense_dir / DC_SUMMARY))
        dense_pair_audit = load(require(dense_dir / PAIR_AUDIT))

        if sparse_base_summary.get("Decoding Mode") != "unconstrained":
            raise ValueError(f"{model}: sparse Base is not unconstrained")
        if dense_base_summary.get("Decoding Mode") != "unconstrained":
            raise ValueError(f"{model}: dense Base is not unconstrained")
        if dense_dc_summary.get("Decoding Mode") != "authentic_dynamic_template":
            raise ValueError(f"{model}: dense DC is not authentic dynamic-template")
        dense_pair_validation = validate_dense_pair_audit(
            model, dense_pair_audit, dense_base_summary, dense_dc_summary
        )

        dense_prompt_provenance = validate_dense_prompt_provenance(
            model, dense_base_summary, dense_dc_summary, dataset_audit
        )

        sparse_base_by_id = index_cases(sparse_base_cases, f"{model}/sparse_base")
        dense_base_by_id = index_cases(dense_base_cases, f"{model}/dense_base")
        dense_dc_by_id = index_cases(dense_dc_cases, f"{model}/dense_dc")

        example_checks = [
            same_examples(sparse_base_by_id, dense_base_by_id, "sparse_base_vs_dense_base"),
            same_examples(sparse_base_by_id, dense_dc_by_id, "sparse_base_vs_dense_dc"),
        ]
        if not all(item["valid"] for item in example_checks):
            raise ValueError(f"{model}: case/input/reference mismatch across regimes")

        dense_prompt_check = prompt_token_comparison(dense_base_by_id, dense_dc_by_id)
        if not dense_prompt_check["valid"]:
            raise ValueError(f"{model}: dense Base and dense DC prompts are not identical")

        normalizations = {
            "base_sparse": validate_normalization(
                sparse_base_cases, "base_sparse", schema, "json_schema"
            ),
            "base_dense": validate_normalization(
                dense_base_cases, "base_dense", schema, "json_schema"
            ),
            "base_dc_dense": validate_normalization(
                dense_dc_cases, "base_dc_dense", schema, "dynamic_template"
            ),
        }
        for arm, result in normalizations.items():
            if not result["valid"]:
                first = result["normalized_null_violations"][0]
                raise ValueError(
                    f"{model}/{arm}: case ID {first['id']!r}: normalized "
                    f"Prediction contains a missing-value alias at {first['paths'][0]}"
                )

        rows = [
            regime_row(
                model,
                "base_sparse",
                "ordinary Base; sparse omit-null instruction",
                True,
                sparse_base_summary,
                normalizations["base_sparse"],
            ),
            regime_row(
                model,
                "base_dense",
                "ordinary Base; dense nullable instruction",
                True,
                dense_base_summary,
                normalizations["base_dense"],
            ),
            regime_row(
                model,
                "base_dc_dense",
                "dynamic-template DC; dense nullable instruction",
                True,
                dense_dc_summary,
                normalizations["base_dc_dense"],
            ),
        ]

        sparse_dc_cases_path = sparse_dir / DC_CASES
        sparse_dc_summary_path = sparse_dir / DC_SUMMARY
        sparse_dc_diagnostic = None
        if sparse_dc_cases_path.is_file() and sparse_dc_summary_path.is_file():
            sparse_dc_cases = load(sparse_dc_cases_path)
            sparse_dc_summary = load(sparse_dc_summary_path)
            sparse_dc_normalization = validate_normalization(
                sparse_dc_cases,
                "base_dc_sparse_incompatible",
                schema,
                "dynamic_template",
            )
            normalizations["base_dc_sparse_incompatible"] = sparse_dc_normalization
            sparse_dc_diagnostic = regime_row(
                model,
                "base_dc_sparse_incompatible",
                "diagnostic only: dense forced decoder with sparse no-null instruction",
                False,
                sparse_dc_summary,
                sparse_dc_normalization,
            )
            rows.append(sparse_dc_diagnostic)

        all_rows.extend(rows)
        sparse_f1 = metric(sparse_base_summary, "F1")
        dense_base_f1 = metric(dense_base_summary, "F1")
        dense_dc_f1 = metric(dense_dc_summary, "F1")
        model_reports[model] = {
            "valid": True,
            "case_and_reference_checks": example_checks,
            "dense_prompt_provenance": dense_prompt_provenance,
            "dense_prompt_token_equality": dense_prompt_check,
            "dense_pair_validation": dense_pair_validation,
            "normalization_checks": normalizations,
            "f1": {
                "base_sparse": sparse_f1,
                "base_dense": dense_base_f1,
                "base_dc_dense": dense_dc_f1,
                "base_dc_sparse_incompatible": (
                    sparse_dc_diagnostic["f1"] if sparse_dc_diagnostic else None
                ),
            },
            "valid_effects": {
                "prompt_effect_on_ordinary_base": dense_base_f1 - sparse_f1,
                "dc_effect_under_identical_dense_prompt": dense_dc_f1 - dense_base_f1,
                "end_to_end_method_effect": dense_dc_f1 - sparse_f1,
            },
            "paper_metrics": {
                "base_sparse_raw_schema_validity_rate": normalizations[
                    "base_sparse"
                ]["raw_schema_validity_rate"],
                "base_dense_raw_schema_validity_rate": normalizations[
                    "base_dense"
                ]["raw_schema_validity_rate"],
                "base_dc_dense_raw_schema_validity_rate": normalizations[
                    "base_dc_dense"
                ]["raw_schema_validity_rate"],
                "base_sparse_completion_tokens_mean": token_stat(
                    sparse_base_summary, "Completion Tokens", "Mean"
                ),
                "base_dense_completion_tokens_mean": token_stat(
                    dense_base_summary, "Completion Tokens", "Mean"
                ),
                "base_dc_dense_completion_tokens_mean": token_stat(
                    dense_dc_summary, "Completion Tokens", "Mean"
                ),
                "base_sparse_truncations": truncation_or_limit_hits(
                    sparse_base_cases, sparse_base_summary
                )["count"],
                "base_dense_truncations": truncation_or_limit_hits(
                    dense_base_cases, dense_base_summary
                )["count"],
                "base_dc_dense_truncations": (
                    (dense_pair_audit.get("dynamic_template") or {}).get(
                        "length_truncated_outputs", 0
                    )
                ),
                "base_dc_dense_invalid_outputs": (
                    (dense_pair_audit.get("dynamic_template") or {}).get(
                        "constraint_invalid_outputs", 0
                    )
                ),
                "base_sparse_truncation_measurement": truncation_or_limit_hits(
                    sparse_base_cases, sparse_base_summary
                )["measurement"],
                "base_dense_truncation_measurement": truncation_or_limit_hits(
                    dense_base_cases, dense_base_summary
                )["measurement"],
                "base_dc_dense_truncation_measurement": truncation_or_limit_hits(
                    dense_dc_cases, dense_dc_summary
                )["measurement"],
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "base_prompt_regime_summary.csv", all_rows)

    paper_rows: list[dict[str, Any]] = []
    for model in models:
        model_report = model_reports[model]
        effects = model_report["valid_effects"]
        metrics = model_report["paper_metrics"]
        paper_rows.append(
            {
                "model": model,
                "base_sparse_f1": model_report["f1"]["base_sparse"],
                "base_dense_f1": model_report["f1"]["base_dense"],
                "base_dc_dense_f1": model_report["f1"]["base_dc_dense"],
                "prompt_delta": effects["prompt_effect_on_ordinary_base"],
                "dense_controlled_dc_delta": effects[
                    "dc_effect_under_identical_dense_prompt"
                ],
                "end_to_end_delta": effects["end_to_end_method_effect"],
                **metrics,
            }
        )
    write_csv(output_dir / "base_prompt_regime_paper_table.csv", paper_rows)

    report = {
        "valid": all(item["valid"] for item in model_reports.values()),
        "interpretation": {
            "primary_baseline": "base_sparse",
            "primary_dynamic_method": "base_dc_dense",
            "prompt_control": "base_dense",
            "excluded_diagnostic": "base_dc_sparse_incompatible",
        },
        "dataset_audit": dataset_audit,
        "models": model_reports,
    }
    (output_dir / "base_prompt_regime_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    row_lookup = {(row["model"], row["regime"]): row for row in all_rows}
    lines = [
        "| Model | Base sparse F1 | Base dense F1 | Base+DC dense F1 | Prompt Δ | DC Δ (dense-controlled) | End-to-end Δ | Raw contract validity: sparse / dense / DC | Completion tokens: sparse / dense / DC | Length finishes / limit hits: sparse / dense / DC | DC invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        effects = model_reports[model]["valid_effects"]
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    fmt(row_lookup[(model, "base_sparse")]["f1"]),
                    fmt(row_lookup[(model, "base_dense")]["f1"]),
                    fmt(row_lookup[(model, "base_dc_dense")]["f1"]),
                    fmt(effects["prompt_effect_on_ordinary_base"]),
                    fmt(effects["dc_effect_under_identical_dense_prompt"]),
                    fmt(effects["end_to_end_method_effect"]),
                    " / ".join(
                        fmt(model_reports[model]["paper_metrics"][key])
                        for key in (
                            "base_sparse_raw_schema_validity_rate",
                            "base_dense_raw_schema_validity_rate",
                            "base_dc_dense_raw_schema_validity_rate",
                        )
                    ),
                    " / ".join(
                        fmt(model_reports[model]["paper_metrics"][key])
                        for key in (
                            "base_sparse_completion_tokens_mean",
                            "base_dense_completion_tokens_mean",
                            "base_dc_dense_completion_tokens_mean",
                        )
                    ),
                    " / ".join(
                        str(model_reports[model]["paper_metrics"][key])
                        for key in (
                            "base_sparse_truncations",
                            "base_dense_truncations",
                            "base_dc_dense_truncations",
                        )
                    ),
                    str(
                        model_reports[model]["paper_metrics"][
                            "base_dc_dense_invalid_outputs"
                        ]
                    ),
                ]
            )
            + " |"
        )
    (output_dir / "base_prompt_regime_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"Validated {len(models)} model(s).")
    print(f"Wrote {output_dir / 'base_prompt_regime_summary.csv'}")
    print(f"Wrote {output_dir / 'base_prompt_regime_summary.md'}")
    print(f"Wrote {output_dir / 'base_prompt_regime_paper_table.csv'}")
    print(f"Wrote {output_dir / 'base_prompt_regime_audit.json'}")


if __name__ == "__main__":
    main()
