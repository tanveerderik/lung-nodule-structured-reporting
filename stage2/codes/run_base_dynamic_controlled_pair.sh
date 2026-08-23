#!/usr/bin/env bash

# Controlled paper-facing comparison of ordinary base-model decoding and the
# corrected dynamic-template constraint. Each model is loaded once
# across both GPUs; ordinary, repeat, and constrained requests run sequentially
# against that same vLLM process.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE_DIR="${CODE_DIR:-$ROOT/codes}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$CODE_DIR/sft_eval_vllm.py}"
FEATUREWISE_SCRIPT="${FEATUREWISE_SCRIPT:-$CODE_DIR/featurewise_eval.py}"
TOKENIZER_AUDIT_SCRIPT="${TOKENIZER_AUDIT_SCRIPT:-$CODE_DIR/audit_dynamic_template.py}"
RESCORE_SCRIPT="${RESCORE_SCRIPT:-$CODE_DIR/rescore_eval_cases.py}"

TEST_FILE="${TEST_FILE:-$ROOT/datasets/test_nodule_base.json}"
TEMPLATE_FILE="${TEMPLATE_FILE:-$ROOT/schemas/lungs_pleura_nodule_template.json}"
FEATURE_SCHEMA="${FEATURE_SCHEMA:-$ROOT/schemas/lungs_pleura_nodule_template.json}"
MODEL_SUBDIR="${MODEL_SUBDIR:-base}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
# The authentic dense decoder may expand to dozens of full nodule dictionaries.
# Keep the original implementation's 16,384-token completion allowance and a
# context window large enough for the rendered prompt plus that completion.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
EVAL_CONTEXT_LIMIT="${EVAL_CONTEXT_LIMIT:-$MAX_MODEL_LEN}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
WORKERS="${WORKERS:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
LIMIT="${LIMIT:-}"
RUN_ORDINARY="${RUN_ORDINARY:-1}"
RUN_ORDINARY_REPEAT="${RUN_ORDINARY_REPEAT:-1}"
RUN_FEATUREWISE="${RUN_FEATUREWISE:-1}"
RUN_TOKENIZER_AUDIT="${RUN_TOKENIZER_AUDIT:-1}"
# CPU-only recovery mode. It never starts vLLM or changes predictions/token
# usage. It updates saved metrics under the current scoring policy, regenerates
# featurewise files, then writes the controlled-pair audit.
AUDIT_ONLY="${AUDIT_ONLY:-0}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.20}"

TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MIN_P="${MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SEED="${SEED:-0}"

JSON_BEGIN_TAG="${JSON_BEGIN_TAG:-<json>}"
JSON_END_TAG="${JSON_END_TAG:-</json>}"
JSON_INDENT="${JSON_INDENT:-4}"
LOGITS_PROCESSOR_PATTERN="${LOGITS_PROCESSOR_PATTERN:-^dynamic_template_constraint\.DynamicTemplateLogitsProcessor$}"
DYNAMIC_TEMPLATE_STYLE="${DYNAMIC_TEMPLATE_STYLE:-canonical}"
DYNAMIC_INCLUDE_JSON_TAGS="${DYNAMIC_INCLUDE_JSON_TAGS:-1}"
DYNAMIC_LEGACY_COMPAT="${DYNAMIC_LEGACY_COMPAT:-1}"
DYNAMIC_TEMPLATE_ADD_SPECIAL_TOKENS="${DYNAMIC_TEMPLATE_ADD_SPECIAL_TOKENS:-0}"
DYNAMIC_MAX_NODULES="${DYNAMIC_MAX_NODULES:-49}"
DYNAMIC_LEGACY_TEMPERATURE="${DYNAMIC_LEGACY_TEMPERATURE:-1.0}"
DYNAMIC_LEGACY_TOP_P="${DYNAMIC_LEGACY_TOP_P:-0.9}"
DYNAMIC_LEGACY_TOP_K="${DYNAMIC_LEGACY_TOP_K:--1}"
DYNAMIC_LEGACY_MIN_P="${DYNAMIC_LEGACY_MIN_P:-0.0}"
DYNAMIC_TRACE="${DYNAMIC_TRACE:-0}"
RUNAWAY_REPEAT_THRESHOLD="${RUNAWAY_REPEAT_THRESHOLD:-64}"

ALL_MODELS=(
    llama3_2_1B
    gemma3_4B
    mistral_7B
    qwen2_5_7B
    llama3_1_8B
    llama3_1_70B
)

# Match the SFT controlled launcher: begin with one bounded diagnostic model.
# Use MODELS_CSV=all only after its audit passes.
MODELS=(llama3_1_8B)
if [[ "${MODELS_CSV:-}" == "all" ]]; then
    MODELS=("${ALL_MODELS[@]}")
elif [[ -n "${MODELS_CSV:-}" ]]; then
    IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
fi

SERVER_PID=""
FAILED_MODELS=()

cleanup_server() {
    if [[ -z "${SERVER_PID:-}" ]]; then
        return
    fi

    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping vLLM server PID $SERVER_PID..."
        kill "$SERVER_PID" 2>/dev/null || true

        for _ in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
    fi

    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
    local log_file="$1"

    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            return 0
        fi

        if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM exited before becoming ready."
            tail -n 100 "$log_file" || true
            return 1
        fi

        sleep 5
    done

    echo "Timed out waiting for vLLM."
    tail -n 100 "$log_file" || true
    return 1
}

require_nonempty() {
    local path="$1"
    if [[ ! -s "$path" ]]; then
        echo "ERROR: Missing or empty file: $path" >&2
        return 1
    fi
}

rescore_arm() {
    local cases_file="$1"
    local summary_file="$2"
    require_nonempty "$cases_file" || return 1
    require_nonempty "$summary_file" || return 1
    PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$RESCORE_SCRIPT" \
        --eval_cases "$cases_file" \
        --summary_json "$summary_file" \
        --min_similarity "$MIN_SIMILARITY" \
        --in_place
}

run_tokenizer_audit() {
    local model_dir="$1"
    local output_json="$2"
    local output_log="$3"
    local -a audit_args=(
        --tokenizer_dir "$model_dir"
        --template_file "$TEMPLATE_FILE"
        --output_json "$output_json"
        --template_style "$DYNAMIC_TEMPLATE_STYLE"
        --json_begin_tag "$JSON_BEGIN_TAG"
        --json_end_tag "$JSON_END_TAG"
        --json_indent "$JSON_INDENT"
        --max_dynamic_nodules "$DYNAMIC_MAX_NODULES"
    )

    [[ "$DYNAMIC_INCLUDE_JSON_TAGS" == "1" ]] \
        && audit_args+=(--include_json_tags) \
        || audit_args+=(--no-include_json_tags)
    [[ "$DYNAMIC_LEGACY_COMPAT" == "1" ]] \
        && audit_args+=(--legacy_compat) \
        || audit_args+=(--no-legacy_compat)
    [[ "$DYNAMIC_TEMPLATE_ADD_SPECIAL_TOKENS" == "1" ]] \
        && audit_args+=(--template_add_special_tokens) \
        || audit_args+=(--no-template_add_special_tokens)

    if ! PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$TOKENIZER_AUDIT_SCRIPT" "${audit_args[@]}" \
        >"$output_log" 2>&1; then
        echo "ERROR: Dynamic-template tokenizer audit failed: $output_json" >&2
        tail -n 100 "$output_log" || true
        return 1
    fi

    require_nonempty "$output_json"
}

run_eval_arm() {
    local mode="$1"
    local model_dir="$2"
    local served_name="$3"
    local cases_file="$4"
    local summary_file="$5"
    local eval_log="$6"
    local -a limit_args=()
    local -a mode_args=(--decoding_mode "$mode")

    if [[ -n "$LIMIT" ]]; then
        limit_args=(--limit "$LIMIT")
    fi

    if [[ "$mode" == "authentic_dynamic_template" ]]; then
        mode_args+=(
            --template_file "$TEMPLATE_FILE"
            --json_begin_tag "$JSON_BEGIN_TAG"
            --json_end_tag "$JSON_END_TAG"
            --json_indent "$JSON_INDENT"
            --dynamic_template_style "$DYNAMIC_TEMPLATE_STYLE"
            --dynamic_max_nodules "$DYNAMIC_MAX_NODULES"
            --dynamic_legacy_temperature "$DYNAMIC_LEGACY_TEMPERATURE"
            --dynamic_legacy_top_p "$DYNAMIC_LEGACY_TOP_P"
            --dynamic_legacy_top_k "$DYNAMIC_LEGACY_TOP_K"
            --dynamic_legacy_min_p "$DYNAMIC_LEGACY_MIN_P"
        )
        [[ "$DYNAMIC_INCLUDE_JSON_TAGS" == "1" ]] \
            && mode_args+=(--dynamic_include_json_tags) \
            || mode_args+=(--no-dynamic_include_json_tags)
        [[ "$DYNAMIC_LEGACY_COMPAT" == "1" ]] \
            && mode_args+=(--dynamic_legacy_compat) \
            || mode_args+=(--no-dynamic_legacy_compat)
        [[ "$DYNAMIC_TEMPLATE_ADD_SPECIAL_TOKENS" == "1" ]] \
            && mode_args+=(--dynamic_template_add_special_tokens) \
            || mode_args+=(--no-dynamic_template_add_special_tokens)
        [[ "$DYNAMIC_TRACE" == "1" ]] && mode_args+=(--dynamic_trace)
    elif [[ "$mode" != "unconstrained" ]]; then
        echo "ERROR: Unsupported base controlled-pair mode: $mode" >&2
        return 1
    fi

    echo
    echo "Running $mode -> $cases_file"
    rm -f "$cases_file" "$summary_file"

    if ! PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$EVAL_SCRIPT" \
        --model_name "$served_name" \
        --tokenizer_dir "$model_dir" \
        --test_file "$TEST_FILE" \
        --base_url "http://127.0.0.1:${PORT}/v1" \
        --output_json "$cases_file" \
        --summary_json "$summary_file" \
        --eval_context_limit "$EVAL_CONTEXT_LIMIT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --workers "$WORKERS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --top_k "$TOP_K" \
        --min_p "$MIN_P" \
        --presence_penalty "$PRESENCE_PENALTY" \
        --frequency_penalty "$FREQUENCY_PENALTY" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --seed "$SEED" \
        --no-completion_add_special_tokens \
        "${mode_args[@]}" \
        "${limit_args[@]}" \
        2>&1 | tee "$eval_log"; then
        echo "ERROR: $mode evaluator exited nonzero." >&2
        return 1
    fi

    require_nonempty "$cases_file"
    require_nonempty "$summary_file"
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vLLM server died during $mode." >&2
        return 1
    fi
}

write_pair_audit() {
    local model_name="$1"
    local ordinary_cases="$2"
    local ordinary_summary="$3"
    local repeat_cases="$4"
    local repeat_summary="$5"
    local dynamic_cases="$6"
    local dynamic_summary="$7"
    local ordinary_featurewise="$8"
    local dynamic_featurewise="$9"
    local output_json="${10}"
    local output_log="${11}"

    rm -f "$output_json"

    if ! "$PYTHON_BIN" - \
        "$model_name" \
        "$GPUS" \
        "$TENSOR_PARALLEL_SIZE" \
        "$WORKERS" \
        "$MAX_NUM_SEQS" \
        "$DYNAMIC_LEGACY_COMPAT" \
        "$RUNAWAY_REPEAT_THRESHOLD" \
        "$TEMPLATE_FILE" \
        "$ordinary_cases" \
        "$ordinary_summary" \
        "$repeat_cases" \
        "$repeat_summary" \
        "$dynamic_cases" \
        "$dynamic_summary" \
        "$ordinary_featurewise" \
        "$dynamic_featurewise" \
        "$output_json" \
        <<'PY' 2>&1 | tee "$output_log"
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

model = sys.argv[1]
gpus = sys.argv[2]
tensor_parallel_size = int(sys.argv[3])
workers = int(sys.argv[4])
max_num_seqs = int(sys.argv[5])
expected_legacy_compat = sys.argv[6] == "1"
runaway_repeat_threshold = int(sys.argv[7])
expected_template = Path(sys.argv[8]).name
ordinary_cases_path = Path(sys.argv[9])
ordinary_summary_path = Path(sys.argv[10])
repeat_cases_path = Path(sys.argv[11]) if sys.argv[11] else None
repeat_summary_path = Path(sys.argv[12]) if sys.argv[12] else None
dynamic_cases_path = Path(sys.argv[13])
dynamic_summary_path = Path(sys.argv[14])
ordinary_featurewise_path = Path(sys.argv[15]) if sys.argv[15] else None
dynamic_featurewise_path = Path(sys.argv[16]) if sys.argv[16] else None
output_path = Path(sys.argv[17])


def load(path: Path) -> Any:
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


def validate_token_totals(
    cases: list[dict[str, Any]],
    summary: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    keys = {
        "Prompt Tokens": "prompt",
        "Completion Tokens": "completion",
        "Total Tokens": "total",
    }
    computed: dict[str, int] = {}
    reported: dict[str, Any] = {}
    matches: dict[str, bool] = {}
    missing = 0

    for summary_key, short_key in keys.items():
        values = []
        for case in cases:
            value = (case.get("Token Usage") or {}).get(summary_key)
            if not isinstance(value, int):
                missing += 1
            else:
                values.append(value)
        computed[short_key] = sum(values)
        reported[short_key] = (
            (summary.get("Token Usage") or {})
            .get(summary_key, {})
            .get("Total")
        )
        matches[short_key] = reported[short_key] == computed[short_key]

    valid = missing == 0 and all(matches.values())
    return {
        "label": label,
        "valid": valid,
        "missing_usage_values": missing,
        "computed": computed,
        "reported": reported,
        "matches": matches,
    }


def micro_f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2.0 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


def summarize_featurewise(
    ordinary_path: Path | None,
    dynamic_path: Path | None,
) -> dict[str, Any] | None:
    if ordinary_path is None or dynamic_path is None:
        return None

    ordinary = {row["feature"]: row for row in load(ordinary_path)}
    dynamic = {row["feature"]: row for row in load(dynamic_path)}
    if set(ordinary) != set(dynamic):
        raise ValueError("Featurewise files do not contain identical feature sets")

    schema_features = sorted(ordinary)

    # The patched evaluator emits every schema row so both arms are aligned.
    # Exclude only rows that are completely unobserved in the reference and in
    # both prediction arms. Retain prediction-only features: their FP counts are
    # real evidence and must not disappear from the paired comparison.
    features = [
        name
        for name in schema_features
        if any(
            float(row.get(metric) or 0.0) != 0.0
            for row in (ordinary[name], dynamic[name])
            for metric in ("gold_appearances", "TP", "FP", "FN")
        )
    ]
    ordinary_macro = (
        sum(float(ordinary[name]["f1"]) for name in features) / len(features)
        if features else 0.0
    )
    dynamic_macro = (
        sum(float(dynamic[name]["f1"]) for name in features) / len(features)
        if features else 0.0
    )
    ordinary_counts = {
        key: sum(float(ordinary[name][key]) for name in features)
        for key in ("TP", "FP", "FN")
    }
    dynamic_counts = {
        key: sum(float(dynamic[name][key]) for name in features)
        for key in ("TP", "FP", "FN")
    }
    deltas = {
        name: float(dynamic[name]["f1"]) - float(ordinary[name]["f1"])
        for name in features
    }

    return {
        "schema_feature_count": len(schema_features),
        "feature_count": len(features),
        "ordinary_macro_f1": ordinary_macro,
        "dynamic_template_macro_f1": dynamic_macro,
        "macro_f1_delta": dynamic_macro - ordinary_macro,
        "ordinary_micro_f1": micro_f1(
            ordinary_counts["TP"], ordinary_counts["FP"], ordinary_counts["FN"]
        ),
        "dynamic_template_micro_f1": micro_f1(
            dynamic_counts["TP"], dynamic_counts["FP"], dynamic_counts["FN"]
        ),
        "features_improved": sum(delta > 1e-12 for delta in deltas.values()),
        "features_worsened": sum(delta < -1e-12 for delta in deltas.values()),
        "features_unchanged": sum(abs(delta) <= 1e-12 for delta in deltas.values()),
        "largest_absolute_changes": [
            {"feature": name, "f1_delta": deltas[name]}
            for name in sorted(deltas, key=lambda item: abs(deltas[item]), reverse=True)[:20]
        ],
    }


ordinary_cases = load(ordinary_cases_path)
ordinary_summary = load(ordinary_summary_path)
dynamic_cases = load(dynamic_cases_path)
dynamic_summary = load(dynamic_summary_path)

if not isinstance(ordinary_cases, list) or not ordinary_cases:
    raise ValueError("Ordinary cases are empty or malformed")
if not isinstance(dynamic_cases, list) or not dynamic_cases:
    raise ValueError("Dynamic-template cases are empty or malformed")

ordinary_by_id = index_cases(ordinary_cases, "ordinary cases")
dynamic_by_id = index_cases(dynamic_cases, "dynamic-template cases")
same_case_ids = list(ordinary_by_id) == list(dynamic_by_id)

controlled_fields = [
    "Original Dataset Cases",
    "Total Evaluated Cases",
    "Skipped Too Long",
    "Eval Context Limit",
    "Max New Tokens",
    "Max Prompt Tokens",
    "Completion Add Special Tokens",
    "Date Scoring Normalization",
]
configuration_mismatches = {
    field: {
        "ordinary": ordinary_summary.get(field),
        "dynamic_template": dynamic_summary.get(field),
    }
    for field in controlled_fields
    if ordinary_summary.get(field) != dynamic_summary.get(field)
}

# The completed 70B evaluation intentionally used a smaller safe runtime
# context for the dynamic arm.  This is acceptable only for the exact audited
# values below; prompts, cases, references, max completion length, and all
# sampling settings remain controlled.
intentional_runtime_differences = {}
if model == "llama3_1_70B":
    expected_70b = {
        "Eval Context Limit": {"ordinary": 32768, "dynamic_template": 24576},
        "Max Prompt Tokens": {"ordinary": 16384, "dynamic_template": 8192},
    }
    if configuration_mismatches == expected_70b:
        intentional_runtime_differences = configuration_mismatches
        configuration_mismatches = {}

ordinary_mode_valid = ordinary_summary.get("Decoding Mode") == "unconstrained"
dynamic_mode_valid = (
    dynamic_summary.get("Decoding Mode") == "authentic_dynamic_template"
    and dynamic_summary.get("Authentic Dynamic Constraint") is True
    and dynamic_summary.get("Schema File") in (None, "")
    and Path(str(dynamic_summary.get("Template File", ""))).name == expected_template
)
dynamic_config = dynamic_summary.get("Dynamic Template Configuration")
expected_dynamic_config = {
    "template_style": "canonical",
    "include_json_tags": True,
    "json_begin_tag": "<json>",
    "json_end_tag": "</json>",
    "json_indent": 4,
    "legacy_compat": expected_legacy_compat,
    "template_add_special_tokens": False,
    "max_dynamic_nodules": 49,
    "legacy_temperature": 1.0,
    "legacy_top_p": 0.9,
    "legacy_top_k": -1,
    "legacy_min_p": 0.0,
    "legacy_presence_penalty": 0.0,
    "legacy_frequency_penalty": 0.0,
    "legacy_repetition_penalty": 1.0,
}
dynamic_config_valid = (
    isinstance(dynamic_config, dict)
    and all(
        dynamic_config.get(key) == value
        for key, value in expected_dynamic_config.items()
    )
)

expected_ordinary_sampling = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 0,
}
ordinary_sampling_valid = (
    ordinary_summary.get("Request Sampling Configuration")
    == expected_ordinary_sampling
)

dynamic_invalid_cases = [
    case
    for case in dynamic_cases
    if not (case.get("Dynamic Constraint Validation") or {}).get("valid")
]
dynamic_invalid_ids = [case.get("ID") for case in dynamic_invalid_cases]
dynamic_valid_count = len(dynamic_cases) - len(dynamic_invalid_cases)
dynamic_all_outputs_valid = not dynamic_invalid_cases

dynamic_missing_parsed_field_ids = [
    case.get("ID")
    for case in dynamic_cases
    if "Parsed Prediction Dense" not in case
]
dynamic_nonempty_parsed_ids = [
    case.get("ID")
    for case in dynamic_cases
    if isinstance(case.get("Parsed Prediction Dense"), dict)
    and bool(case.get("Parsed Prediction Dense"))
]
dynamic_length_truncated_ids = [
    case.get("ID")
    for case in dynamic_invalid_cases
    if case.get("Generation Finish Reason") == "length"
]


def has_suspected_runaway_suffix(case: dict[str, Any]) -> bool:
    raw = case.get("Raw Prediction")
    if not isinstance(raw, str) or not raw:
        return False
    # Detect both one-character loops ("333...") and short periodic loops
    # (for example a repeatedly emitted date fragment). The old detector only
    # recognized the first form and missed three Gemma failures.
    suffix = raw[-max(4096, runaway_repeat_threshold * 64):]
    for period in range(1, 33):
        repeats = max(8, (runaway_repeat_threshold + period - 1) // period)
        repeated_length = period * repeats
        if len(suffix) < repeated_length:
            continue
        block = suffix[-period:]
        if block.strip() and suffix.endswith(block * repeats):
            return True
    return False


dynamic_suspected_runaway_ids = [
    case.get("ID")
    for case in dynamic_invalid_cases
    if has_suspected_runaway_suffix(case)
]
dynamic_finish_reason_counts = Counter(
    str(case.get("Generation Finish Reason")) for case in dynamic_cases
)
dynamic_invalid_finish_reason_counts = Counter(
    str(case.get("Generation Finish Reason")) for case in dynamic_invalid_cases
)
dynamic_validation_error_counts = Counter(
    str(error)
    for case in dynamic_invalid_cases
    for error in (case.get("Dynamic Constraint Validation") or {}).get("errors", [])
)

prompt_token_mismatch_ids = []
changed_normalized_predictions = 0
changed_raw_outputs = 0
if same_case_ids:
    for key in ordinary_by_id:
        ordinary_case = ordinary_by_id[key]
        dynamic_case = dynamic_by_id[key]
        ordinary_prompt = (ordinary_case.get("Token Usage") or {}).get("Prompt Tokens")
        dynamic_prompt = (dynamic_case.get("Token Usage") or {}).get("Prompt Tokens")
        if ordinary_prompt != dynamic_prompt:
            prompt_token_mismatch_ids.append(ordinary_case.get("ID"))
        if ordinary_case.get("Prediction") != dynamic_case.get("Prediction"):
            changed_normalized_predictions += 1
        if ordinary_case.get("Raw Prediction") != dynamic_case.get("Raw Prediction"):
            changed_raw_outputs += 1

repeat_result = None
repeat_valid = True
if repeat_cases_path is not None:
    repeat_cases = load(repeat_cases_path)
    repeat_summary = load(repeat_summary_path) if repeat_summary_path is not None else {}
    repeat_by_id = index_cases(repeat_cases, "ordinary repeat cases")
    repeat_same_ids = list(repeat_by_id) == list(ordinary_by_id)
    repeat_changed_ids = []
    repeat_prompt_mismatch_ids = []
    if repeat_same_ids:
        for key in ordinary_by_id:
            ordinary_case = ordinary_by_id[key]
            repeat_case = repeat_by_id[key]
            if ordinary_case.get("Raw Prediction") != repeat_case.get("Raw Prediction"):
                repeat_changed_ids.append(ordinary_case.get("ID"))
            ordinary_prompt = (ordinary_case.get("Token Usage") or {}).get("Prompt Tokens")
            repeat_prompt = (repeat_case.get("Token Usage") or {}).get("Prompt Tokens")
            if ordinary_prompt != repeat_prompt:
                repeat_prompt_mismatch_ids.append(ordinary_case.get("ID"))

    repeat_fields = controlled_fields + [
        "Decoding Mode",
        "Request Sampling Configuration",
    ]
    repeat_configuration_mismatches = {
        field: {
            "ordinary": ordinary_summary.get(field),
            "repeat": repeat_summary.get(field),
        }
        for field in repeat_fields
        if ordinary_summary.get(field) != repeat_summary.get(field)
    }
    repeat_token_validation = validate_token_totals(
        repeat_cases, repeat_summary, "ordinary_repeat"
    )
    repeat_valid = (
        repeat_same_ids
        and not repeat_changed_ids
        and not repeat_prompt_mismatch_ids
        and not repeat_configuration_mismatches
        and repeat_token_validation["valid"]
    )
    repeat_result = {
        "reproducible": repeat_valid,
        "same_case_ids_and_order": repeat_same_ids,
        "changed_raw_outputs": len(repeat_changed_ids),
        "changed_ids": repeat_changed_ids[:50],
        "prompt_token_mismatch_count": len(repeat_prompt_mismatch_ids),
        "configuration_mismatches": repeat_configuration_mismatches,
        "token_validation": repeat_token_validation,
    }

ordinary_token_validation = validate_token_totals(
    ordinary_cases, ordinary_summary, "ordinary"
)
dynamic_token_validation = validate_token_totals(
    dynamic_cases, dynamic_summary, "dynamic_template"
)

# Experimental integrity is distinct from decoder success, but the corrected
# pipeline must not silently approve invalid constrained outputs.
controlled_experiment_valid = (
    same_case_ids
    and not configuration_mismatches
    and ordinary_mode_valid
    and ordinary_sampling_valid
    and dynamic_mode_valid
    and dynamic_config_valid
    and not prompt_token_mismatch_ids
    and repeat_valid
    and ordinary_token_validation["valid"]
    and dynamic_token_validation["valid"]
)

ordinary_completion = ordinary_token_validation["computed"]["completion"]
dynamic_completion = dynamic_token_validation["computed"]["completion"]
report = {
    "comparison": (
        "ordinary base vs corrected typed dynamic-template DC"
        if expected_legacy_compat
        else "ordinary base vs strict trie dynamic-template DC"
    ),
    "model": model,
    "controlled_comparison_valid": controlled_experiment_valid,
    "controlled_experiment_valid": controlled_experiment_valid,
    "all_dynamic_outputs_valid": dynamic_all_outputs_valid,
    "server_configuration": {
        "gpus": gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "workers": workers,
        "max_num_seqs": max_num_seqs,
        "same_server_process_for_both_arms": True,
        "conditions_run_sequentially": True,
    },
    "same_case_ids_and_order": same_case_ids,
    "configuration_mismatches": configuration_mismatches,
    "intentional_runtime_differences": intentional_runtime_differences,
    "prompt_token_mismatch_count": len(prompt_token_mismatch_ids),
    "prompt_token_mismatch_ids": prompt_token_mismatch_ids[:50],
    "ordinary_repeat": repeat_result,
    "ordinary": {
        "cases": len(ordinary_cases),
        "mode_valid": ordinary_mode_valid,
        "sampling_configuration_valid": ordinary_sampling_valid,
        "summary_precision": ordinary_summary.get("Precision"),
        "summary_recall": ordinary_summary.get("Recall"),
        "summary_f1": ordinary_summary.get("F1"),
        "completion_tokens_total": ordinary_completion,
        "token_validation": ordinary_token_validation,
    },
    "dynamic_template": {
        "cases": len(dynamic_cases),
        "mode_valid": dynamic_mode_valid,
        "configuration_valid": dynamic_config_valid,
        "legacy_compat": expected_legacy_compat,
        "constraint_valid_outputs": dynamic_valid_count,
        "constraint_invalid_outputs": len(dynamic_invalid_cases),
        "constraint_validity_rate": (
            dynamic_valid_count / len(dynamic_cases) if dynamic_cases else 0.0
        ),
        "invalid_constraint_ids": dynamic_invalid_ids[:50],
        "invalid_constraint_ids_truncated": len(dynamic_invalid_ids) > 50,
        "parsed_dense_objects": len(dynamic_nonempty_parsed_ids),
        "missing_parsed_field_count": len(dynamic_missing_parsed_field_ids),
        "missing_parsed_field_ids": dynamic_missing_parsed_field_ids[:50],
        "length_truncated_outputs": len(dynamic_length_truncated_ids),
        "length_truncated_ids": dynamic_length_truncated_ids[:50],
        "suspected_runaway_slot_outputs": len(dynamic_suspected_runaway_ids),
        "suspected_runaway_slot_ids": dynamic_suspected_runaway_ids[:50],
        "runaway_repeat_threshold": runaway_repeat_threshold,
        "finish_reason_counts": dict(dynamic_finish_reason_counts),
        "invalid_finish_reason_counts": dict(dynamic_invalid_finish_reason_counts),
        "validation_error_counts": dict(dynamic_validation_error_counts),
        "summary_precision": dynamic_summary.get("Precision"),
        "summary_recall": dynamic_summary.get("Recall"),
        "summary_f1": dynamic_summary.get("F1"),
        "completion_tokens_total": dynamic_completion,
        "token_validation": dynamic_token_validation,
    },
    "delta": {
        "summary_f1": (dynamic_summary.get("F1") or 0.0)
        - (ordinary_summary.get("F1") or 0.0),
        "completion_tokens_total": dynamic_completion - ordinary_completion,
    },
    "identical_raw_outputs": len(ordinary_cases) - changed_raw_outputs,
    "changed_raw_outputs": changed_raw_outputs,
    "changed_normalized_predictions": changed_normalized_predictions,
    "featurewise": summarize_featurewise(
        ordinary_featurewise_path, dynamic_featurewise_path
    ),
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
if not controlled_experiment_valid:
    raise SystemExit(
        "Controlled base experiment failed integrity validation; do not use these results."
    )
if not dynamic_all_outputs_valid:
    raise SystemExit(
        "Corrected dynamic decoder produced "
        f"{len(dynamic_invalid_cases)}/{len(dynamic_cases)} invalid outputs. "
        "Do not use these results; inspect the saved audit and generation logs."
    )
PY
    then
        echo "ERROR: Controlled base-pair audit failed." >&2
        return 1
    fi

    require_nonempty "$output_json"
}

run_model_pair() {
    local model_name="${1//[[:space:]]/}"
    local model_dir="$ROOT/models/$model_name/$MODEL_SUBDIR"
    local result_dir="$RESULTS_DIR/$model_name"
    local served_name="${model_name}_base_controlled_pair"
    local server_log="$result_dir/vllm_base_controlled_pair.log"

    local ordinary_cases="$result_dir/base_eval_cases_controlled_ordinary.json"
    local ordinary_summary="$result_dir/base_eval_summary_controlled_ordinary.json"
    local ordinary_log="$result_dir/base_eval_controlled_ordinary.log"
    local repeat_cases="$result_dir/base_eval_cases_controlled_ordinary_repeat.json"
    local repeat_summary="$result_dir/base_eval_summary_controlled_ordinary_repeat.json"
    local repeat_log="$result_dir/base_eval_controlled_ordinary_repeat.log"
    local dynamic_cases="$result_dir/base_eval_cases_controlled_dynamic_template.json"
    local dynamic_summary="$result_dir/base_eval_summary_controlled_dynamic_template.json"
    local dynamic_log="$result_dir/base_eval_controlled_dynamic_template.log"
    local audit_json="$result_dir/base_eval_controlled_pair_audit.json"
    local audit_log="$result_dir/base_eval_controlled_pair_audit.log"
    local tokenizer_audit_json="$result_dir/base_dynamic_template_controlled_tokenizer_audit.json"
    local tokenizer_audit_log="$result_dir/base_dynamic_template_controlled_tokenizer_audit.log"
    local ordinary_featurewise="$result_dir/base_featurewise_f1_controlled_ordinary"
    local dynamic_featurewise="$result_dir/base_featurewise_f1_controlled_dynamic_template"

    echo
    echo "======================================================================"
    if [[ "$RUN_ORDINARY" == "1" ]]; then
        echo "CONTROLLED BASE PAIR: $model_name"
    else
        echo "DYNAMIC-ONLY RERUN:   $model_name"
    fi
    echo "MODEL PATH:           $model_dir"
    echo "GPUs / TP:            $GPUS / $TENSOR_PARALLEL_SIZE"
    echo "Workers / seqs:       $WORKERS / $MAX_NUM_SEQS"
    echo "======================================================================"

    if [[ "$AUDIT_ONLY" == "1" ]]; then
        echo "CPU-ONLY RECOVERY: predictions and token usage will not be modified."

        for existing_result in \
            "$ordinary_cases" \
            "$ordinary_summary" \
            "$repeat_cases" \
            "$repeat_summary" \
            "$dynamic_cases" \
            "$dynamic_summary"; do
            require_nonempty "$existing_result" || return 1
        done

        if [[ "$RUN_ORDINARY_REPEAT" != "1" ]]; then
            echo "ERROR: AUDIT_ONLY requires the saved ordinary repeat files." >&2
            return 1
        fi

        echo "Rescoring saved arms with precision-preserving date normalization..."
        rescore_arm "$ordinary_cases" "$ordinary_summary" || return 1
        rescore_arm "$repeat_cases" "$repeat_summary" || return 1
        rescore_arm "$dynamic_cases" "$dynamic_summary" || return 1

        if [[ "$RUN_FEATUREWISE" == "1" ]]; then
            echo "Regenerating ordinary featurewise metrics from saved cases..."
            "$PYTHON_BIN" "$FEATUREWISE_SCRIPT" \
                --eval_cases "$ordinary_cases" \
                --schema_file "$FEATURE_SCHEMA" \
                --output_csv "${ordinary_featurewise}.csv" \
                --output_json "${ordinary_featurewise}.json" \
                --min_similarity "$MIN_SIMILARITY" || return 1

            echo "Regenerating dynamic-template featurewise metrics from saved cases..."
            "$PYTHON_BIN" "$FEATUREWISE_SCRIPT" \
                --eval_cases "$dynamic_cases" \
                --schema_file "$FEATURE_SCHEMA" \
                --output_csv "${dynamic_featurewise}.csv" \
                --output_json "${dynamic_featurewise}.json" \
                --min_similarity "$MIN_SIMILARITY" || return 1
        else
            require_nonempty "${ordinary_featurewise}.json" || return 1
            require_nonempty "${dynamic_featurewise}.json" || return 1
        fi

        write_pair_audit \
            "$model_name" \
            "$ordinary_cases" \
            "$ordinary_summary" \
            "$repeat_cases" \
            "$repeat_summary" \
            "$dynamic_cases" \
            "$dynamic_summary" \
            "${ordinary_featurewise}.json" \
            "${dynamic_featurewise}.json" \
            "$audit_json" \
            "$audit_log" || return 1

        echo "CPU-only controlled base-pair audit passed: $audit_json"
        return 0
    fi

    if [[ ! -f "$model_dir/config.json" ]]; then
        echo "ERROR: Missing model: $model_dir/config.json" >&2
        return 1
    fi

    mkdir -p "$result_dir"
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "ERROR: Port $PORT already has an OpenAI-compatible server." >&2
        return 1
    fi

    if [[ "$RUN_TOKENIZER_AUDIT" == "1" ]]; then
        run_tokenizer_audit \
            "$model_dir" "$tokenizer_audit_json" "$tokenizer_audit_log" || return 1
    fi

    VLLM_USE_V1=0 \
    PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$VLLM_BIN" serve "$model_dir" \
        --served-model-name "$served_name" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --disable-custom-all-reduce \
        --generation-config vllm \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --logits-processor-pattern "$LOGITS_PROCESSOR_PATTERN" \
        >"$server_log" 2>&1 &
    SERVER_PID=$!

    wait_for_server "$server_log" || return 1

    if [[ "$RUN_ORDINARY" == "1" ]]; then
        run_eval_arm unconstrained "$model_dir" "$served_name" \
            "$ordinary_cases" "$ordinary_summary" "$ordinary_log" || return 1
    else
        echo "Reusing saved ordinary arm; no ordinary inference will run."
        rescore_arm "$ordinary_cases" "$ordinary_summary" || return 1
    fi

    local audit_repeat_cases=""
    local audit_repeat_summary=""
    if [[ "$RUN_ORDINARY_REPEAT" == "1" ]]; then
        if [[ "$RUN_ORDINARY" == "1" ]]; then
            run_eval_arm unconstrained "$model_dir" "$served_name" \
                "$repeat_cases" "$repeat_summary" "$repeat_log" || return 1
        else
            echo "Reusing saved ordinary repeat arm."
            rescore_arm "$repeat_cases" "$repeat_summary" || return 1
        fi
        audit_repeat_cases="$repeat_cases"
        audit_repeat_summary="$repeat_summary"
    elif [[ "$RUN_ORDINARY" == "1" ]]; then
        rm -f "$repeat_cases" "$repeat_summary" "$repeat_log"
    fi

    run_eval_arm authentic_dynamic_template "$model_dir" "$served_name" \
        "$dynamic_cases" "$dynamic_summary" "$dynamic_log" || return 1

    local audit_ordinary_featurewise=""
    local audit_dynamic_featurewise=""
    if [[ "$RUN_FEATUREWISE" == "1" ]]; then
        "$PYTHON_BIN" "$FEATUREWISE_SCRIPT" \
            --eval_cases "$ordinary_cases" \
            --schema_file "$FEATURE_SCHEMA" \
            --output_csv "${ordinary_featurewise}.csv" \
            --output_json "${ordinary_featurewise}.json" \
            --min_similarity "$MIN_SIMILARITY" || return 1
        "$PYTHON_BIN" "$FEATUREWISE_SCRIPT" \
            --eval_cases "$dynamic_cases" \
            --schema_file "$FEATURE_SCHEMA" \
            --output_csv "${dynamic_featurewise}.csv" \
            --output_json "${dynamic_featurewise}.json" \
            --min_similarity "$MIN_SIMILARITY" || return 1
        audit_ordinary_featurewise="${ordinary_featurewise}.json"
        audit_dynamic_featurewise="${dynamic_featurewise}.json"
    else
        if [[ "$RUN_ORDINARY" == "1" ]]; then
            rm -f \
                "${ordinary_featurewise}.csv" \
                "${ordinary_featurewise}.json"
        fi
        rm -f \
            "${dynamic_featurewise}.csv" \
            "${dynamic_featurewise}.json"
    fi

    write_pair_audit \
        "$model_name" \
        "$ordinary_cases" \
        "$ordinary_summary" \
        "$audit_repeat_cases" \
        "$audit_repeat_summary" \
        "$dynamic_cases" \
        "$dynamic_summary" \
        "$audit_ordinary_featurewise" \
        "$audit_dynamic_featurewise" \
        "$audit_json" \
        "$audit_log" || return 1

    cleanup_server
    echo "Controlled base pair passed for $model_name: $audit_json"
}

for required in \
    "$EVAL_SCRIPT" \
    "$FEATUREWISE_SCRIPT" \
    "$TOKENIZER_AUDIT_SCRIPT" \
    "$RESCORE_SCRIPT" \
    "$CODE_DIR/date_normalization.py" \
    "$CODE_DIR/nodule_scoring.py" \
    "$CODE_DIR/dynamic_template_constraint.py" \
    "$TEST_FILE" \
    "$TEMPLATE_FILE" \
    "$FEATURE_SCHEMA"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: Required file not found: $required" >&2
        exit 1
    fi
done

if [[ "$DYNAMIC_TRACE" == "1" && -z "$LIMIT" ]]; then
    echo "ERROR: DYNAMIC_TRACE=1 requires a bounded LIMIT (for example LIMIT=3)." >&2
    exit 1
fi

if (( WORKERS < 1 || MAX_NUM_SEQS < 1 )); then
    echo "ERROR: WORKERS and MAX_NUM_SEQS must be positive integers." >&2
    exit 1
fi
if (( WORKERS > MAX_NUM_SEQS )); then
    echo "NOTE: WORKERS exceeds MAX_NUM_SEQS; extra client requests will queue."
fi

if [[ "$GPUS" != "0,1" || "$TENSOR_PARALLEL_SIZE" != "2" ]]; then
    echo "WARNING: This project expects GPUS=0,1 and TENSOR_PARALLEL_SIZE=2." >&2
fi

for model_name in "${MODELS[@]}"; do
    if ! run_model_pair "$model_name"; then
        FAILED_MODELS+=("$model_name")
        cleanup_server
        sleep 5
    fi
done

if [[ "${#FAILED_MODELS[@]}" -gt 0 ]]; then
    echo "Failed models: ${FAILED_MODELS[*]}" >&2
    exit 1
fi

if [[ "$RUN_ORDINARY" == "1" ]]; then
    echo "All controlled base ordinary/dynamic-template pairs completed successfully."
else
    echo "All requested base dynamic-template-only reruns completed successfully."
fi
