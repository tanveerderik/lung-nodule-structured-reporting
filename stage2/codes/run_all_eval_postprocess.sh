#!/usr/bin/env bash

# Generate per-model and combined post-processing outputs for the final four
# paper-facing regimes:
#   1. base, controlled ordinary greedy decoding
#   2. base + controlled corrected typed dynamic-template constraints
#   3. SFT, controlled ordinary greedy decoding
#   4. SFT + controlled sparse XGrammar constraints
#
# This script is CPU-only. Run it after both controlled launchers have finished
# for all six models. Only the new controlled-pair result names are accepted.

set -euo pipefail

ROOT="${ROOT:-$HOME/llm_medical/lungs_pleura_nodule_focused}"
CODE_DIR="${CODE_DIR:-$ROOT/codes}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
FEATURE_SCHEMA="${FEATURE_SCHEMA:-$ROOT/schemas/lungs_pleura_nodule_template.json}"
XGRAMMAR_SCHEMA="${XGRAMMAR_SCHEMA:-$ROOT/schemas/lung_nodule_xgrammar_schema.json}"
EXPECTED_BASE_DC_MODE="${EXPECTED_BASE_DC_MODE:-authentic_dynamic_template}"
EXPECTED_SFT_DC_MODE="${EXPECTED_SFT_DC_MODE:-v0_tagged_grammar}"
EXPECTED_TEMPLATE_BASENAME="${EXPECTED_TEMPLATE_BASENAME:-lungs_pleura_nodule_template.json}"
EXPECTED_XGRAMMAR_SCHEMA_BASENAME="${EXPECTED_XGRAMMAR_SCHEMA_BASENAME:-lung_nodule_xgrammar_schema.json}"
EXPECTED_CASES="${EXPECTED_CASES:-250}"
EXPECTED_BASE_MAX_NEW_TOKENS="${EXPECTED_BASE_MAX_NEW_TOKENS:-16384}"
BASE_CONTEXT_AUDIT_MAX_NEW_TOKENS="${BASE_CONTEXT_AUDIT_MAX_NEW_TOKENS:-16384}"
SFT_CONTEXT_AUDIT_MAX_NEW_TOKENS="${SFT_CONTEXT_AUDIT_MAX_NEW_TOKENS:-1536}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.20}"
RUN_CONTEXT_AUDIT="${RUN_CONTEXT_AUDIT:-0}"

MODELS=(
    llama3_2_1B
    gemma3_4B
    mistral_7B
    qwen2_5_7B
    llama3_1_8B
    llama3_1_70B
)

REGIMES=(base base_dc sft sft_dc)

declare -A CASES_FILENAME=(
    [base]="base_eval_cases_controlled_ordinary.json"
    [base_dc]="base_eval_cases_controlled_dynamic_template.json"
    [sft]="sft_eval_cases_controlled_ordinary.json"
    [sft_dc]="sft_eval_cases_controlled_xgrammar.json"
)

declare -A SUMMARY_FILENAME=(
    [base]="base_eval_summary_controlled_ordinary.json"
    [base_dc]="base_eval_summary_controlled_dynamic_template.json"
    [sft]="sft_eval_summary_controlled_ordinary.json"
    [sft_dc]="sft_eval_summary_controlled_xgrammar.json"
)

declare -A FEATUREWISE_STEM=(
    [base]="base_featurewise_f1_controlled_ordinary"
    [base_dc]="base_featurewise_f1_controlled_dynamic_template"
    [sft]="sft_featurewise_f1_controlled_ordinary"
    [sft_dc]="sft_featurewise_f1_controlled_xgrammar"
)

declare -A FEATUREWISE_COMBINED_PREFIX=(
    [base]="base_featurewise_f1_side_by_side"
    [base_dc]="base_dc_featurewise_f1_side_by_side"
    [sft]="sft_featurewise_f1_side_by_side"
    [sft_dc]="sft_dc_featurewise_f1_side_by_side"
)

declare -A SUMMARY_COMBINED_PREFIX=(
    [base]="base_eval_summary_side_by_side"
    [base_dc]="base_dc_eval_summary_side_by_side"
    [sft]="sft_eval_summary_side_by_side"
    [sft_dc]="sft_dc_eval_summary_side_by_side"
)

require_file() {
    local path="$1"
    if [[ ! -s "$path" ]]; then
        echo "ERROR: Missing or empty file: $path" >&2
        exit 1
    fi
}

require_file "$CODE_DIR/featurewise_eval.py"
require_file "$CODE_DIR/date_normalization.py"
require_file "$CODE_DIR/nodule_scoring.py"
require_file "$CODE_DIR/compile_featurewise_summaries.py"
require_file "$CODE_DIR/compile_eval_summaries.py"
require_file "$FEATURE_SCHEMA"
require_file "$XGRAMMAR_SCHEMA"
mkdir -p "$RESULTS_DIR"

"$PYTHON_BIN" - <<'PY'
import importlib.util

required = ["pandas", "openpyxl", "tabulate"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "ERROR: Missing postprocessing Python packages: "
        + ", ".join(missing)
        + ". Install them in the active llm_medical environment before rerunning."
    )
PY

# Validate provenance before generating or overwriting any derived result. This
# prevents diagnostic LIMIT runs, standalone SFT runs, sparse XGrammar, and the
# dense legacy decoder from being silently mixed under the same paper labels.
validate_model_outputs() {
    local model="$1"
    local result_dir="$RESULTS_DIR/$model"

    local base_cases="$result_dir/${CASES_FILENAME[base]}"
    local base_summary="$result_dir/${SUMMARY_FILENAME[base]}"
    local base_repeat_cases="$result_dir/base_eval_cases_controlled_ordinary_repeat.json"
    local base_repeat_summary="$result_dir/base_eval_summary_controlled_ordinary_repeat.json"
    local base_dc_cases="$result_dir/${CASES_FILENAME[base_dc]}"
    local base_dc_summary="$result_dir/${SUMMARY_FILENAME[base_dc]}"
    local sft_cases="$result_dir/${CASES_FILENAME[sft]}"
    local sft_summary="$result_dir/${SUMMARY_FILENAME[sft]}"
    local sft_repeat_cases="$result_dir/sft_eval_cases_controlled_ordinary_repeat.json"
    local sft_repeat_summary="$result_dir/sft_eval_summary_controlled_ordinary_repeat.json"
    local sft_dc_cases="$result_dir/${CASES_FILENAME[sft_dc]}"
    local sft_dc_summary="$result_dir/${SUMMARY_FILENAME[sft_dc]}"
    local base_pair_audit="$result_dir/base_eval_controlled_pair_audit.json"
    local sft_pair_audit="$result_dir/sft_eval_controlled_pair_audit.json"

    for path in \
        "$base_cases" "$base_summary" \
        "$base_repeat_cases" "$base_repeat_summary" \
        "$base_dc_cases" "$base_dc_summary" \
        "$sft_cases" "$sft_summary" \
        "$sft_repeat_cases" "$sft_repeat_summary" \
        "$sft_dc_cases" "$sft_dc_summary" \
        "$base_pair_audit" "$sft_pair_audit"; do
        require_file "$path"
    done

    "$PYTHON_BIN" - \
        "$model" \
        "$EXPECTED_CASES" \
        "$EXPECTED_BASE_DC_MODE" \
        "$EXPECTED_SFT_DC_MODE" \
        "$EXPECTED_TEMPLATE_BASENAME" \
        "$EXPECTED_XGRAMMAR_SCHEMA_BASENAME" \
        "$EXPECTED_BASE_MAX_NEW_TOKENS" \
        "$base_cases" "$base_summary" \
        "$base_repeat_cases" "$base_repeat_summary" \
        "$base_dc_cases" "$base_dc_summary" \
        "$sft_cases" "$sft_summary" \
        "$sft_repeat_cases" "$sft_repeat_summary" \
        "$sft_dc_cases" "$sft_dc_summary" \
        "$base_pair_audit" "$sft_pair_audit" <<'PY'
import json
import math
import sys
from pathlib import Path

model = sys.argv[1]
expected_cases = int(sys.argv[2]) if sys.argv[2] else None
expected_base_dc_mode = sys.argv[3]
expected_sft_dc_mode = sys.argv[4]
expected_template = sys.argv[5]
expected_schema = sys.argv[6]
expected_base_max_new_tokens = int(sys.argv[7])

paths = {
    "base_cases": Path(sys.argv[8]),
    "base_summary": Path(sys.argv[9]),
    "base_repeat_cases": Path(sys.argv[10]),
    "base_repeat_summary": Path(sys.argv[11]),
    "base_dc_cases": Path(sys.argv[12]),
    "base_dc_summary": Path(sys.argv[13]),
    "sft_cases": Path(sys.argv[14]),
    "sft_summary": Path(sys.argv[15]),
    "sft_repeat_cases": Path(sys.argv[16]),
    "sft_repeat_summary": Path(sys.argv[17]),
    "sft_dc_cases": Path(sys.argv[18]),
    "sft_dc_summary": Path(sys.argv[19]),
    "base_pair_audit": Path(sys.argv[20]),
    "sft_pair_audit": Path(sys.argv[21]),
}


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


base_cases = load(paths["base_cases"])
base_summary = load(paths["base_summary"])
base_repeat_cases = load(paths["base_repeat_cases"])
base_repeat_summary = load(paths["base_repeat_summary"])
base_dc_cases = load(paths["base_dc_cases"])
base_dc_summary = load(paths["base_dc_summary"])
sft_cases = load(paths["sft_cases"])
sft_summary = load(paths["sft_summary"])
sft_repeat_cases = load(paths["sft_repeat_cases"])
sft_repeat_summary = load(paths["sft_repeat_summary"])
sft_dc_cases = load(paths["sft_dc_cases"])
sft_dc_summary = load(paths["sft_dc_summary"])
base_audit = load(paths["base_pair_audit"])
sft_audit = load(paths["sft_pair_audit"])

arms = {
    "base": (paths["base_cases"], base_cases, paths["base_summary"], base_summary),
    "base_dc": (
        paths["base_dc_cases"], base_dc_cases,
        paths["base_dc_summary"], base_dc_summary,
    ),
    "sft": (paths["sft_cases"], sft_cases, paths["sft_summary"], sft_summary),
    "sft_dc": (
        paths["sft_dc_cases"], sft_dc_cases,
        paths["sft_dc_summary"], sft_dc_summary,
    ),
}

expected_sampling = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 0,
}

for arm, (case_path, cases, summary_path, summary) in arms.items():
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"ERROR [{model}/{arm}]: {case_path} is not a non-empty list.")
    if expected_cases is not None and len(cases) != expected_cases:
        raise SystemExit(
            f"ERROR [{model}/{arm}]: found {len(cases)} cases; expected "
            f"{expected_cases}. A LIMIT diagnostic may be occupying final filenames."
        )
    if summary.get("Total Evaluated Cases") != len(cases):
        raise SystemExit(
            f"ERROR [{model}/{arm}]: summary/case count mismatch in {summary_path}."
        )
    if expected_cases is not None and summary.get("Original Dataset Cases") != expected_cases:
        raise SystemExit(
            f"ERROR [{model}/{arm}]: Original Dataset Cases is "
            f"{summary.get('Original Dataset Cases')!r}; expected {expected_cases}."
        )
    if summary.get("Skipped Too Long") != 0:
        raise SystemExit(
            f"ERROR [{model}/{arm}]: {summary.get('Skipped Too Long')} cases were skipped."
        )
    if summary.get("Completion Add Special Tokens") is not False:
        raise SystemExit(
            f"ERROR [{model}/{arm}]: add_special_tokens was not explicitly false."
        )
    if "Prediction Normalization" not in summary:
        raise SystemExit(
            f"ERROR [{model}/{arm}]: summary was not produced by the patched evaluator."
        )
    if not summary.get("Date Scoring Normalization"):
        raise SystemExit(
            f"ERROR [{model}/{arm}]: date-aware metrics are missing. Run the "
            "CPU-only saved-result rescore before postprocessing."
        )
    required_case_fields = {"Prediction", "Raw Prediction", "Token Usage", "Eval Metrics"}
    missing = sorted(required_case_fields - set(cases[0]))
    if missing:
        raise SystemExit(
            f"ERROR [{model}/{arm}]: first case is missing patched fields: {missing}"
        )

date_policies = {
    summary.get("Date Scoring Normalization") for _, _, _, summary in arms.values()
}
if len(date_policies) != 1:
    raise SystemExit(
        f"ERROR [{model}]: evaluation arms use different date-scoring policies."
    )

for arm, summary in (("base", base_summary), ("sft", sft_summary), ("sft_dc", sft_dc_summary)):
    if summary.get("Request Sampling Configuration") != expected_sampling:
        raise SystemExit(
            f"ERROR [{model}/{arm}]: sampling configuration is not the controlled greedy setup."
        )

if base_summary.get("Decoding Mode") != "unconstrained":
    raise SystemExit(f"ERROR [{model}/base]: expected unconstrained decoding.")
if sft_summary.get("Decoding Mode") != "unconstrained":
    raise SystemExit(f"ERROR [{model}/sft]: expected controlled ordinary decoding.")
for arm, summary in (("base", base_summary), ("base_dc", base_dc_summary)):
    if summary.get("Max New Tokens") != expected_base_max_new_tokens:
        raise SystemExit(
            f"ERROR [{model}/{arm}]: Max New Tokens is "
            f"{summary.get('Max New Tokens')!r}; expected "
            f"{expected_base_max_new_tokens}. Old 1,536-token dense results "
            "must not be mixed with the corrected base run."
        )

def validate_repeat(label, cases, summary, repeat_cases, repeat_summary):
    """Validate a controlled ordinary repeat directly, not only via its audit."""
    if not isinstance(repeat_cases, list) or len(repeat_cases) != len(cases):
        raise SystemExit(f"ERROR [{model}/{label}]: repeat case count does not match.")
    if expected_cases is not None and len(repeat_cases) != expected_cases:
        raise SystemExit(
            f"ERROR [{model}/{label}]: found {len(repeat_cases)} cases; "
            f"expected {expected_cases}."
        )
    if repeat_summary.get("Total Evaluated Cases") != len(repeat_cases):
        raise SystemExit(f"ERROR [{model}/{label}]: repeat summary/case count mismatch.")

    repeat_controlled_fields = [
        "Original Dataset Cases",
        "Total Evaluated Cases",
        "Skipped Too Long",
        "Eval Context Limit",
        "Max New Tokens",
        "Max Prompt Tokens",
        "Decoding Mode",
        "Completion Add Special Tokens",
        "Request Sampling Configuration",
        "Date Scoring Normalization",
        "Precision",
        "Recall",
        "F1",
        "IoU",
        "TP",
        "FP",
        "FN",
        "Token Usage",
    ]
    repeat_mismatches = [
        field
        for field in repeat_controlled_fields
        if summary.get(field) != repeat_summary.get(field)
    ]
    if repeat_mismatches:
        raise SystemExit(
            f"ERROR [{model}/{label}]: summary fields differ: {repeat_mismatches}"
        )

    changed = [
        first.get("ID")
        for first, second in zip(cases, repeat_cases)
        if first.get("ID") != second.get("ID")
        or first.get("Raw Prediction") != second.get("Raw Prediction")
        or first.get("Prediction") != second.get("Prediction")
        or first.get("Token Usage") != second.get("Token Usage")
    ]
    if changed:
        raise SystemExit(
            f"ERROR [{model}/{label}]: ordinary decoding changed for "
            f"{len(changed)} cases; first IDs: {changed[:10]}"
        )


validate_repeat(
    "base_repeat", base_cases, base_summary, base_repeat_cases, base_repeat_summary
)
validate_repeat(
    "sft_repeat", sft_cases, sft_summary, sft_repeat_cases, sft_repeat_summary
)
if base_dc_summary.get("Decoding Mode") != expected_base_dc_mode:
    raise SystemExit(
        f"ERROR [{model}/base_dc]: expected {expected_base_dc_mode!r}, got "
        f"{base_dc_summary.get('Decoding Mode')!r}."
    )
if sft_dc_summary.get("Decoding Mode") != expected_sft_dc_mode:
    raise SystemExit(
        f"ERROR [{model}/sft_dc]: expected {expected_sft_dc_mode!r}, got "
        f"{sft_dc_summary.get('Decoding Mode')!r}."
    )

# Authentic legacy dynamic-template validation applies only to base + DC.
actual_template = Path(str(base_dc_summary.get("Template File", ""))).name
if actual_template != expected_template:
    raise SystemExit(
        f"ERROR [{model}/base_dc]: template {actual_template!r}; expected {expected_template!r}."
    )
if base_dc_summary.get("Schema File") not in (None, ""):
    raise SystemExit(f"ERROR [{model}/base_dc]: an XGrammar schema was active.")
if base_dc_summary.get("Authentic Dynamic Constraint") is not True:
    raise SystemExit(f"ERROR [{model}/base_dc]: authentic DC flag is not true.")
config = base_dc_summary.get("Dynamic Template Configuration")
if not isinstance(config, dict):
    raise SystemExit(f"ERROR [{model}/base_dc]: dynamic-template config is missing.")
expected_dynamic_config = {
    "template_style": "canonical",
    "include_json_tags": True,
    "json_begin_tag": "<json>",
    "json_end_tag": "</json>",
    "json_indent": 4,
    "legacy_compat": True,
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
bad_dynamic_config = {
    key: {"expected": expected, "actual": config.get(key)}
    for key, expected in expected_dynamic_config.items()
    if config.get(key) != expected
}
if bad_dynamic_config:
    raise SystemExit(
        f"ERROR [{model}/base_dc]: dynamic-template configuration mismatch: "
        f"{bad_dynamic_config}"
    )
invalid_dense = [
    case.get("ID")
    for case in base_dc_cases
    if not (case.get("Dynamic Constraint Validation") or {}).get("valid")
]
valid_dense_count = len(base_dc_cases) - len(invalid_dense)
nonempty_parsed_dense_count = sum(
    isinstance(case.get("Parsed Prediction Dense"), dict)
    and bool(case.get("Parsed Prediction Dense"))
    for case in base_dc_cases
)
length_truncated_count = sum(
    case.get("Generation Finish Reason") == "length"
    for case in base_dc_cases
    if not (case.get("Dynamic Constraint Validation") or {}).get("valid")
)

# Sparse XGrammar validation applies only to controlled SFT + DC.
actual_schema = Path(str(sft_dc_summary.get("Schema File", ""))).name
if actual_schema != expected_schema:
    raise SystemExit(
        f"ERROR [{model}/sft_dc]: schema {actual_schema!r}; expected {expected_schema!r}."
    )
if sft_dc_summary.get("Template File") not in (None, ""):
    raise SystemExit(f"ERROR [{model}/sft_dc]: dense template was unexpectedly active.")
if sft_dc_summary.get("Authentic Dynamic Constraint") is not False:
    raise SystemExit(f"ERROR [{model}/sft_dc]: dynamic-template DC flag must be false.")
if sft_dc_summary.get("Guided Decoding Backend") != "xgrammar:no-fallback":
    raise SystemExit(
        f"ERROR [{model}/sft_dc]: expected xgrammar:no-fallback, got "
        f"{sft_dc_summary.get('Guided Decoding Backend')!r}."
    )

def require_matching_f1(label, audit_value, summary_value):
    try:
        matches = math.isclose(
            float(audit_value), float(summary_value), rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise SystemExit(f"ERROR [{model}]: {label} F1 does not match its summary.")


def validate_common_audit(label, audit, expected_comparison):
    if audit.get("comparison") != expected_comparison:
        raise SystemExit(f"ERROR [{model}/{label}]: unrecognized controlled-pair audit.")
    if audit.get("controlled_comparison_valid") is not True:
        raise SystemExit(f"ERROR [{model}/{label}]: controlled-pair audit did not pass.")
    if audit.get("configuration_mismatches"):
        raise SystemExit(f"ERROR [{model}/{label}]: audit reports configuration mismatches.")
    if audit.get("prompt_token_mismatch_count") != 0:
        raise SystemExit(f"ERROR [{model}/{label}]: audit reports prompt-token mismatches.")
    repeat = audit.get("ordinary_repeat")
    if not isinstance(repeat, dict) or repeat.get("reproducible") is not True:
        raise SystemExit(
            f"ERROR [{model}/{label}]: controlled ordinary repeat was not reproducible."
        )


# The new base-pair audit proves both arms were run sequentially on the same
# two-GPU vLLM process, with identical inputs/configuration and a stable repeat.
validate_common_audit(
    "base_pair",
    base_audit,
    "ordinary base vs corrected typed dynamic-template DC",
)
if base_audit.get("controlled_experiment_valid") is not True:
    raise SystemExit(
        f"ERROR [{model}/base_pair]: audit is stale or experimental integrity failed."
    )
base_server = base_audit.get("server_configuration") or {}
expected_base_server = {
    "gpus": "0,1",
    "tensor_parallel_size": 2,
    "same_server_process_for_both_arms": True,
    "conditions_run_sequentially": True,
}
server_mismatches = {
    key: {"expected": expected, "actual": base_server.get(key)}
    for key, expected in expected_base_server.items()
    if base_server.get(key) != expected
}
if server_mismatches:
    raise SystemExit(
        f"ERROR [{model}/base_pair]: server configuration mismatch: {server_mismatches}"
    )
for concurrency_field in ("workers", "max_num_seqs"):
    value = base_server.get(concurrency_field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(
            f"ERROR [{model}/base_pair]: {concurrency_field} must be a positive "
            f"integer; found {value!r}."
        )
if base_audit.get("same_case_ids_and_order") is not True:
    raise SystemExit(f"ERROR [{model}/base_pair]: audit reports different case IDs/order.")
base_audit_ordinary = base_audit.get("ordinary") or {}
base_audit_dynamic = base_audit.get("dynamic_template") or {}
if base_audit_ordinary.get("cases") != len(base_cases):
    raise SystemExit(f"ERROR [{model}/base_pair]: ordinary case count is stale.")
if base_audit_dynamic.get("cases") != len(base_dc_cases):
    raise SystemExit(f"ERROR [{model}/base_pair]: dynamic case count is stale.")
if base_audit_ordinary.get("mode_valid") is not True:
    raise SystemExit(f"ERROR [{model}/base_pair]: ordinary mode validation failed.")
if base_audit_ordinary.get("sampling_configuration_valid") is not True:
    raise SystemExit(f"ERROR [{model}/base_pair]: ordinary sampling validation failed.")
if base_audit_dynamic.get("mode_valid") is not True:
    raise SystemExit(f"ERROR [{model}/base_pair]: dynamic mode validation failed.")
if base_audit_dynamic.get("configuration_valid") is not True:
    raise SystemExit(f"ERROR [{model}/base_pair]: dynamic configuration validation failed.")
if base_audit_dynamic.get("constraint_valid_outputs") != valid_dense_count:
    raise SystemExit(
        f"ERROR [{model}/base_pair]: audited valid-output count does not match cases."
    )
if base_audit_dynamic.get("constraint_invalid_outputs") != len(invalid_dense):
    raise SystemExit(
        f"ERROR [{model}/base_pair]: audited invalid-output count does not match cases."
    )
if base_audit_dynamic.get("parsed_dense_objects") != nonempty_parsed_dense_count:
    raise SystemExit(
        f"ERROR [{model}/base_pair]: audited parsed-object count does not match cases."
    )
if base_audit_dynamic.get("length_truncated_outputs") != length_truncated_count:
    raise SystemExit(
        f"ERROR [{model}/base_pair]: audited truncation count does not match cases."
    )
if base_audit.get("all_dynamic_outputs_valid") != (len(invalid_dense) == 0):
    raise SystemExit(
        f"ERROR [{model}/base_pair]: all-output-valid flag does not match cases."
    )
for arm_name, arm in (
    ("ordinary", base_audit_ordinary),
    ("dynamic_template", base_audit_dynamic),
):
    if (arm.get("token_validation") or {}).get("valid") is not True:
        raise SystemExit(
            f"ERROR [{model}/base_pair]: {arm_name} audit token validation failed."
        )
if ((base_audit.get("ordinary_repeat") or {}).get("token_validation") or {}).get(
    "valid"
) is not True:
    raise SystemExit(f"ERROR [{model}/base_pair]: repeat token validation failed.")
require_matching_f1("base ordinary audit", base_audit_ordinary.get("summary_f1"), base_summary.get("F1"))
require_matching_f1(
    "base dynamic audit", base_audit_dynamic.get("summary_f1"), base_dc_summary.get("F1")
)

base_ids = [json.dumps(case.get("ID"), sort_keys=True) for case in base_cases]
base_dynamic_ids = [json.dumps(case.get("ID"), sort_keys=True) for case in base_dc_cases]
if base_ids != base_dynamic_ids:
    raise SystemExit(f"ERROR [{model}/base_pair]: controlled arms have different IDs/order.")

# The SFT audit proves prompt/configuration equality, ordinary-repeat
# reproducibility, and complete XGrammar tag/JSON/schema validity.
validate_common_audit("sft_pair", sft_audit, "ordinary SFT vs sparse XGrammar")
sft_audit_ordinary = sft_audit.get("ordinary") or {}
sft_audit_xgrammar = sft_audit.get("xgrammar") or {}
if sft_audit_ordinary.get("cases") != len(sft_cases):
    raise SystemExit(f"ERROR [{model}/sft_pair]: ordinary case count is stale.")
if sft_audit_xgrammar.get("cases") != len(sft_dc_cases):
    raise SystemExit(f"ERROR [{model}/sft_pair]: XGrammar case count is stale.")
for field in ("exact_tagged_outputs", "valid_json_objects", "schema_valid_outputs"):
    if sft_audit_xgrammar.get(field) != len(sft_dc_cases):
        raise SystemExit(
            f"ERROR [{model}/sft_pair]: XGrammar audit field {field!r} is incomplete."
        )
require_matching_f1(
    "SFT ordinary audit", sft_audit_ordinary.get("summary_f1"), sft_summary.get("F1")
)
require_matching_f1(
    "SFT XGrammar audit", sft_audit_xgrammar.get("summary_f1"), sft_dc_summary.get("F1")
)

sft_ids = [json.dumps(case.get("ID"), sort_keys=True) for case in sft_cases]
xgrammar_ids = [json.dumps(case.get("ID"), sort_keys=True) for case in sft_dc_cases]
if sft_ids != xgrammar_ids:
    raise SystemExit(f"ERROR [{model}/sft_pair]: controlled arms have different IDs/order.")

print(f"Validated {model}: both controlled pairs and all four regimes passed.")
PY
}

echo "======================================================================"
echo "VALIDATING BOTH CONTROLLED PAIRS (FOUR EVALUATION ARMS)"
echo "======================================================================"

for model in "${MODELS[@]}"; do
    validate_model_outputs "$model"
done

echo "Validation passed."

echo
echo "======================================================================"
echo "GENERATING PER-MODEL FEATUREWISE RESULTS"
echo "======================================================================"

for regime in "${REGIMES[@]}"; do
    echo
    echo "--- Regime: $regime ---"

    for model in "${MODELS[@]}"; do
        result_dir="$RESULTS_DIR/$model"
        cases_file="$result_dir/${CASES_FILENAME[$regime]}"
        output_stem="$result_dir/${FEATUREWISE_STEM[$regime]}"

        echo "[$regime] $model"

        "$PYTHON_BIN" "$CODE_DIR/featurewise_eval.py" \
            --eval_cases "$cases_file" \
            --schema_file "$FEATURE_SCHEMA" \
            --output_csv "${output_stem}.csv" \
            --output_json "${output_stem}.json" \
            --min_similarity "$MIN_SIMILARITY"
    done

done

echo
echo "======================================================================"
echo "COMPILING PER-REGIME FEATUREWISE TABLES AND EVALUATION SUMMARIES"
echo "======================================================================"

for regime in "${REGIMES[@]}"; do
    "$PYTHON_BIN" "$CODE_DIR/compile_featurewise_summaries.py" \
        --results_dir "$RESULTS_DIR" \
        --featurewise_filename "${FEATUREWISE_STEM[$regime]}.json" \
        --output_prefix "${FEATUREWISE_COMBINED_PREFIX[$regime]}"

    "$PYTHON_BIN" "$CODE_DIR/compile_eval_summaries.py" \
        --results_dir "$RESULTS_DIR" \
        --summary_filename "${SUMMARY_FILENAME[$regime]}" \
        --output_prefix "${SUMMARY_COMBINED_PREFIX[$regime]}"
done

echo
echo "======================================================================"
echo "BUILDING CROSS-REGIME F1, TOKEN-USAGE, AND VALIDATION TABLES"
echo "======================================================================"

RESULTS_DIR="$RESULTS_DIR" MODELS_CSV="$(IFS=,; echo "${MODELS[*]}")" \
"$PYTHON_BIN" <<'PY'
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

results_dir = Path(os.environ["RESULTS_DIR"]).expanduser().resolve()
models = [x for x in os.environ["MODELS_CSV"].split(",") if x]

regimes = {
    "base": {
        "label": "Baseline (controlled ordinary greedy)",
        "cases": "base_eval_cases_controlled_ordinary.json",
        "summary": "base_eval_summary_controlled_ordinary.json",
        "featurewise": "base_featurewise_f1_controlled_ordinary.json",
    },
    "base_dc": {
        "label": "Baseline + authentic dynamic-template DC (controlled)",
        "cases": "base_eval_cases_controlled_dynamic_template.json",
        "summary": "base_eval_summary_controlled_dynamic_template.json",
        "featurewise": "base_featurewise_f1_controlled_dynamic_template.json",
    },
    "sft": {
        "label": "SFT (controlled ordinary greedy)",
        "cases": "sft_eval_cases_controlled_ordinary.json",
        "summary": "sft_eval_summary_controlled_ordinary.json",
        "featurewise": "sft_featurewise_f1_controlled_ordinary.json",
    },
    "sft_dc": {
        "label": "SFT + sparse XGrammar (controlled)",
        "cases": "sft_eval_cases_controlled_xgrammar.json",
        "summary": "sft_eval_summary_controlled_xgrammar.json",
        "featurewise": "sft_featurewise_f1_controlled_xgrammar.json",
    },
}


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, full_key))
        else:
            out[full_key] = value
    return out


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def f1_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    return precision, recall, f1


summary_rows: list[dict[str, Any]] = []
token_validation_rows: list[dict[str, Any]] = []
feature_rows: list[dict[str, Any]] = []
feature_aggregate_rows: list[dict[str, Any]] = []
base_controlled_pair_rows: list[dict[str, Any]] = []
sft_controlled_pair_rows: list[dict[str, Any]] = []

for model in models:
    base_audit_path = results_dir / model / "base_eval_controlled_pair_audit.json"
    with base_audit_path.open("r", encoding="utf-8") as f:
        base_audit = json.load(f)

    base_ordinary = base_audit.get("ordinary") or {}
    dynamic = base_audit.get("dynamic_template") or {}
    base_delta = base_audit.get("delta") or {}
    base_repeat = base_audit.get("ordinary_repeat") or {}
    base_featurewise = base_audit.get("featurewise") or {}
    base_server = base_audit.get("server_configuration") or {}

    base_controlled_pair_rows.append(
        {
            "model": model,
            "controlled_comparison_valid": base_audit.get("controlled_comparison_valid"),
            "controlled_experiment_valid": base_audit.get("controlled_experiment_valid"),
            "all_dynamic_outputs_valid": base_audit.get("all_dynamic_outputs_valid"),
            "ordinary_repeat_reproducible": base_repeat.get("reproducible"),
            "same_case_ids_and_order": base_audit.get("same_case_ids_and_order"),
            "prompt_token_mismatch_count": base_audit.get("prompt_token_mismatch_count"),
            "configuration_mismatch_count": len(base_audit.get("configuration_mismatches") or {}),
            "case_count": base_ordinary.get("cases"),
            "ordinary_summary_f1": base_ordinary.get("summary_f1"),
            "dynamic_template_summary_f1": dynamic.get("summary_f1"),
            "summary_f1_delta": base_delta.get("summary_f1"),
            "dynamic_constraint_valid_outputs": dynamic.get("constraint_valid_outputs"),
            "dynamic_constraint_invalid_outputs": dynamic.get("constraint_invalid_outputs"),
            "dynamic_constraint_validity_rate": dynamic.get("constraint_validity_rate"),
            "dynamic_parsed_dense_objects": dynamic.get("parsed_dense_objects"),
            "dynamic_length_truncated_outputs": dynamic.get("length_truncated_outputs"),
            "dynamic_suspected_runaway_slot_outputs": dynamic.get("suspected_runaway_slot_outputs"),
            "dynamic_finish_reason_counts": json.dumps(
                dynamic.get("finish_reason_counts") or {}, sort_keys=True
            ),
            "dynamic_invalid_finish_reason_counts": json.dumps(
                dynamic.get("invalid_finish_reason_counts") or {}, sort_keys=True
            ),
            "dynamic_validation_error_counts": json.dumps(
                dynamic.get("validation_error_counts") or {}, sort_keys=True
            ),
            "identical_raw_outputs": base_audit.get("identical_raw_outputs"),
            "changed_raw_outputs": base_audit.get("changed_raw_outputs"),
            "changed_normalized_predictions": base_audit.get("changed_normalized_predictions"),
            "ordinary_completion_tokens_total": base_ordinary.get("completion_tokens_total"),
            "dynamic_completion_tokens_total": dynamic.get("completion_tokens_total"),
            "completion_tokens_total_delta": base_delta.get("completion_tokens_total"),
            "ordinary_featurewise_macro_f1": base_featurewise.get("ordinary_macro_f1"),
            "dynamic_featurewise_macro_f1": base_featurewise.get("dynamic_template_macro_f1"),
            "featurewise_macro_f1_delta": base_featurewise.get("macro_f1_delta"),
            "ordinary_featurewise_micro_f1": base_featurewise.get("ordinary_micro_f1"),
            "dynamic_featurewise_micro_f1": base_featurewise.get("dynamic_template_micro_f1"),
            "features_improved": base_featurewise.get("features_improved"),
            "features_worsened": base_featurewise.get("features_worsened"),
            "features_unchanged": base_featurewise.get("features_unchanged"),
            "gpus": base_server.get("gpus"),
            "tensor_parallel_size": base_server.get("tensor_parallel_size"),
            "workers": base_server.get("workers"),
            "max_num_seqs": base_server.get("max_num_seqs"),
            "same_server_process": base_server.get("same_server_process_for_both_arms"),
            "conditions_run_sequentially": base_server.get("conditions_run_sequentially"),
            "audit_file": str(base_audit_path),
        }
    )

    sft_audit_path = results_dir / model / "sft_eval_controlled_pair_audit.json"
    with sft_audit_path.open("r", encoding="utf-8") as f:
        sft_audit = json.load(f)

    ordinary = sft_audit.get("ordinary") or {}
    xgrammar = sft_audit.get("xgrammar") or {}
    delta = sft_audit.get("delta") or {}
    repeat = sft_audit.get("ordinary_repeat") or {}
    featurewise = sft_audit.get("featurewise") or {}

    sft_controlled_pair_rows.append(
        {
            "model": model,
            "controlled_comparison_valid": sft_audit.get("controlled_comparison_valid"),
            "ordinary_repeat_reproducible": repeat.get("reproducible"),
            "prompt_token_mismatch_count": sft_audit.get("prompt_token_mismatch_count"),
            "configuration_mismatch_count": len(sft_audit.get("configuration_mismatches") or {}),
            "case_count": ordinary.get("cases"),
            "ordinary_summary_f1": ordinary.get("summary_f1"),
            "xgrammar_summary_f1": xgrammar.get("summary_f1"),
            "summary_f1_delta": delta.get("summary_f1"),
            "ordinary_schema_valid_outputs": ordinary.get("schema_valid_outputs"),
            "xgrammar_schema_valid_outputs": xgrammar.get("schema_valid_outputs"),
            "ordinary_exact_tagged_outputs": ordinary.get("exact_tagged_outputs"),
            "xgrammar_exact_tagged_outputs": xgrammar.get("exact_tagged_outputs"),
            "ordinary_null_selections": ordinary.get("null_selections"),
            "xgrammar_null_selections": xgrammar.get("null_selections"),
            "identical_raw_outputs": sft_audit.get("identical_raw_outputs"),
            "changed_raw_outputs": sft_audit.get("changed_raw_outputs"),
            "changed_normalized_predictions": sft_audit.get("changed_normalized_predictions"),
            "ordinary_completion_tokens_total": ordinary.get("completion_tokens_total"),
            "xgrammar_completion_tokens_total": xgrammar.get("completion_tokens_total"),
            "completion_tokens_total_delta": delta.get("completion_tokens_total"),
            "ordinary_featurewise_macro_f1": featurewise.get("ordinary_macro_f1"),
            "xgrammar_featurewise_macro_f1": featurewise.get("xgrammar_macro_f1"),
            "featurewise_macro_f1_delta": featurewise.get("macro_f1_delta"),
            "ordinary_featurewise_micro_f1": featurewise.get("ordinary_micro_f1"),
            "xgrammar_featurewise_micro_f1": featurewise.get("xgrammar_micro_f1"),
            "features_improved": featurewise.get("features_improved"),
            "features_worsened": featurewise.get("features_worsened"),
            "features_unchanged": featurewise.get("features_unchanged"),
            "audit_file": str(sft_audit_path),
        }
    )

for regime_key, spec in regimes.items():
    for model in models:
        model_dir = results_dir / model
        summary_path = model_dir / spec["summary"]
        cases_path = model_dir / spec["cases"]
        feature_path = model_dir / spec["featurewise"]

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        summary_row = {
            "regime": regime_key,
            "regime_label": spec["label"],
            "model": model,
            "summary_file": str(summary_path),
        }
        summary_row.update(flatten(summary))
        summary_rows.append(summary_row)

        with cases_path.open("r", encoding="utf-8") as f:
            cases = json.load(f)

        case_prompt = []
        case_completion = []
        case_total = []
        missing_token_usage = 0

        for case in cases:
            usage = case.get("Token Usage") or {}
            p = usage.get("Prompt Tokens")
            c = usage.get("Completion Tokens")
            t = usage.get("Total Tokens")

            if p is None or c is None or t is None:
                missing_token_usage += 1
                continue

            case_prompt.append(int(p))
            case_completion.append(int(c))
            case_total.append(int(t))

        reported_prompt = (
            summary.get("Token Usage", {})
            .get("Prompt Tokens", {})
            .get("Total")
        )
        reported_completion = (
            summary.get("Token Usage", {})
            .get("Completion Tokens", {})
            .get("Total")
        )
        reported_total = (
            summary.get("Token Usage", {})
            .get("Total Tokens", {})
            .get("Total")
        )

        computed_prompt = sum(case_prompt)
        computed_completion = sum(case_completion)
        computed_total = sum(case_total)

        token_validation_rows.append(
            {
                "regime": regime_key,
                "regime_label": spec["label"],
                "model": model,
                "case_count": len(cases),
                "cases_with_complete_token_usage": len(case_total),
                "cases_missing_token_usage": missing_token_usage,
                "computed_prompt_tokens": computed_prompt,
                "reported_prompt_tokens": reported_prompt,
                "prompt_total_matches": reported_prompt == computed_prompt,
                "computed_completion_tokens": computed_completion,
                "reported_completion_tokens": reported_completion,
                "completion_total_matches": reported_completion == computed_completion,
                "computed_total_tokens": computed_total,
                "reported_total_tokens": reported_total,
                "total_matches": reported_total == computed_total,
            }
        )

        with feature_path.open("r", encoding="utf-8") as f:
            feature_data = json.load(f)

        supported_tp = 0.0
        supported_fp = 0.0
        supported_fn = 0.0
        supported_features = 0
        zero_support_features = 0
        zero_support_fp = 0.0

        for item in feature_data:
            row = {
                "regime": regime_key,
                "regime_label": spec["label"],
                "model": model,
                "feature": item.get("feature"),
                "source_file": str(feature_path),
            }
            for metric in [
                "f1",
                "precision",
                "recall",
                "TP",
                "FP",
                "FN",
                "gold_appearances",
            ]:
                row[metric] = item.get(metric)
            feature_rows.append(row)

            gold = float(item.get("gold_appearances") or 0.0)
            tp = float(item.get("TP") or 0.0)
            fp = float(item.get("FP") or 0.0)
            fn = float(item.get("FN") or 0.0)

            if gold > 0:
                supported_features += 1
                supported_tp += tp
                supported_fp += fp
                supported_fn += fn
            else:
                zero_support_features += 1
                zero_support_fp += fp

        precision, recall, f1 = f1_from_counts(
            supported_tp,
            supported_fp,
            supported_fn,
        )

        feature_aggregate_rows.append(
            {
                "regime": regime_key,
                "regime_label": spec["label"],
                "model": model,
                "supported_feature_count": supported_features,
                "TP_supported_features": supported_tp,
                "FP_supported_features": supported_fp,
                "FN_supported_features": supported_fn,
                "precision_supported_features": precision,
                "recall_supported_features": recall,
                "f1_supported_features": f1,
                "zero_support_feature_count": zero_support_features,
                "zero_support_FP": zero_support_fp,
            }
        )

summary_df = pd.DataFrame(summary_rows)
feature_df = pd.DataFrame(feature_rows)
feature_aggregate_df = pd.DataFrame(feature_aggregate_rows)
token_validation_df = pd.DataFrame(token_validation_rows)
base_controlled_pair_df = (
    pd.DataFrame(base_controlled_pair_rows).sort_values("model").reset_index(drop=True)
)
sft_controlled_pair_df = (
    pd.DataFrame(sft_controlled_pair_rows).sort_values("model").reset_index(drop=True)
)

regime_order = ["base", "base_dc", "sft", "sft_dc"]
summary_df["regime"] = pd.Categorical(summary_df["regime"], regime_order, ordered=True)
feature_df["regime"] = pd.Categorical(feature_df["regime"], regime_order, ordered=True)
feature_aggregate_df["regime"] = pd.Categorical(
    feature_aggregate_df["regime"], regime_order, ordered=True
)
token_validation_df["regime"] = pd.Categorical(
    token_validation_df["regime"], regime_order, ordered=True
)

summary_df = summary_df.sort_values(["regime", "model"]).reset_index(drop=True)
feature_df = feature_df.sort_values(["regime", "feature", "model"]).reset_index(drop=True)
feature_aggregate_df = feature_aggregate_df.sort_values(["regime", "model"]).reset_index(drop=True)
token_validation_df = token_validation_df.sort_values(["regime", "model"]).reset_index(drop=True)

# Main all-regime summary, including the evaluator's aggregate metrics and token statistics.
preferred = [
    "regime",
    "regime_label",
    "model",
    "Precision",
    "Recall",
    "F1",
    "IoU",
    "TP",
    "FP",
    "FN",
    "Original Dataset Cases",
    "Total Evaluated Cases",
    "Skipped Too Long",
    "Token Usage.Prompt Tokens.Total",
    "Token Usage.Prompt Tokens.Mean",
    "Token Usage.Prompt Tokens.Median",
    "Token Usage.Prompt Tokens.Min",
    "Token Usage.Prompt Tokens.Max",
    "Token Usage.Prompt Tokens.P95",
    "Token Usage.Completion Tokens.Total",
    "Token Usage.Completion Tokens.Mean",
    "Token Usage.Completion Tokens.Median",
    "Token Usage.Completion Tokens.Min",
    "Token Usage.Completion Tokens.Max",
    "Token Usage.Completion Tokens.P95",
    "Token Usage.Total Tokens.Total",
    "Token Usage.Total Tokens.Mean",
    "Token Usage.Total Tokens.Median",
    "Token Usage.Total Tokens.Min",
    "Token Usage.Total Tokens.Max",
    "Token Usage.Total Tokens.P95",
    "Decoding Mode",
    "Completion Add Special Tokens",
    "Request Sampling Configuration.temperature",
    "Request Sampling Configuration.top_p",
    "Request Sampling Configuration.top_k",
    "Request Sampling Configuration.min_p",
    "Request Sampling Configuration.presence_penalty",
    "Request Sampling Configuration.frequency_penalty",
    "Request Sampling Configuration.repetition_penalty",
    "Request Sampling Configuration.seed",
    "Schema File",
    "Template File",
    "Guided Decoding Backend",
    "Dynamic Template Processor",
    "Authentic Dynamic Constraint",
    "Legacy Fidelity Note",
    "Prediction Normalization",
    "Date Scoring Normalization",
]
summary_columns = [c for c in preferred if c in summary_df.columns]
summary_out = summary_df[summary_columns].copy()

summary_out.to_csv(results_dir / "all_eval_summary_side_by_side.csv", index=False)
summary_out.to_markdown(results_dir / "all_eval_summary_side_by_side.md", index=False)
summary_out.to_excel(results_dir / "all_eval_summary_side_by_side.xlsx", index=False)

# Dedicated token table for efficiency reporting.
token_columns = [
    "regime",
    "regime_label",
    "model",
    "F1",
    "Token Usage.Prompt Tokens.Total",
    "Token Usage.Prompt Tokens.Mean",
    "Token Usage.Completion Tokens.Total",
    "Token Usage.Completion Tokens.Mean",
    "Token Usage.Completion Tokens.Median",
    "Token Usage.Completion Tokens.P95",
    "Token Usage.Completion Tokens.Max",
    "Token Usage.Total Tokens.Total",
    "Token Usage.Total Tokens.Mean",
    "Token Usage.Total Tokens.Median",
    "Token Usage.Total Tokens.P95",
    "Token Usage.Total Tokens.Max",
]
token_columns = [c for c in token_columns if c in summary_df.columns]
token_df = summary_df[token_columns].copy()
token_df.to_csv(results_dir / "all_eval_token_usage_side_by_side.csv", index=False)
token_df.to_markdown(results_dir / "all_eval_token_usage_side_by_side.md", index=False)

# Full featurewise results across all four regimes.
feature_df.to_csv(results_dir / "all_featurewise_f1_all_regimes_long.csv", index=False)

with pd.ExcelWriter(
    results_dir / "all_featurewise_f1_all_regimes.xlsx",
    engine="openpyxl",
) as writer:
    feature_df.to_excel(writer, sheet_name="long", index=False)

    feature_df = feature_df.copy()
    feature_df["regime_model"] = (
        feature_df["regime_label"].astype(str)
        + " | "
        + feature_df["model"].astype(str)
    )

    for metric in ["f1", "precision", "recall", "TP", "FP", "FN", "gold_appearances"]:
        wide = feature_df.pivot_table(
            index="feature",
            columns="regime_model",
            values=metric,
            aggfunc="first",
            observed=False,
        )
        wide.reset_index().to_excel(writer, sheet_name=metric[:31], index=False)

# Exact micro aggregate restricted to features represented in the reference set.
feature_aggregate_df.to_csv(
    results_dir / "all_featurewise_supported_micro_summary.csv",
    index=False,
)
feature_aggregate_df.to_markdown(
    results_dir / "all_featurewise_supported_micro_summary.md",
    index=False,
)

# Verify that token totals in summaries match the underlying case files.
token_validation_df.to_csv(
    results_dir / "all_eval_token_usage_validation.csv",
    index=False,
)
token_validation_df.to_markdown(
    results_dir / "all_eval_token_usage_validation.md",
    index=False,
)

# Compact, paper-facing audits of both clean controlled comparisons.
base_controlled_pair_df.to_csv(
    results_dir / "base_controlled_pair_audit_side_by_side.csv",
    index=False,
)
base_controlled_pair_df.to_markdown(
    results_dir / "base_controlled_pair_audit_side_by_side.md",
    index=False,
)
base_controlled_pair_df.to_excel(
    results_dir / "base_controlled_pair_audit_side_by_side.xlsx",
    index=False,
)

sft_controlled_pair_df.to_csv(
    results_dir / "sft_controlled_pair_audit_side_by_side.csv",
    index=False,
)
sft_controlled_pair_df.to_markdown(
    results_dir / "sft_controlled_pair_audit_side_by_side.md",
    index=False,
)
sft_controlled_pair_df.to_excel(
    results_dir / "sft_controlled_pair_audit_side_by_side.xlsx",
    index=False,
)

validation_failures = token_validation_df[
    (token_validation_df["cases_missing_token_usage"] != 0)
    | (~token_validation_df["prompt_total_matches"])
    | (~token_validation_df["completion_total_matches"])
    | (~token_validation_df["total_matches"])
]
base_pair_validation_passed = bool(
    base_controlled_pair_df["controlled_comparison_valid"].eq(True).all()
    and base_controlled_pair_df["controlled_experiment_valid"].eq(True).all()
    and base_controlled_pair_df["ordinary_repeat_reproducible"].eq(True).all()
    and base_controlled_pair_df["same_case_ids_and_order"].eq(True).all()
)
base_all_dynamic_outputs_valid = bool(
    base_controlled_pair_df["all_dynamic_outputs_valid"].eq(True).all()
)
base_dynamic_invalid_output_count = int(
    base_controlled_pair_df["dynamic_constraint_invalid_outputs"].fillna(0).sum()
)
base_dynamic_length_truncated_output_count = int(
    base_controlled_pair_df["dynamic_length_truncated_outputs"].fillna(0).sum()
)
base_dynamic_suspected_runaway_output_count = int(
    base_controlled_pair_df["dynamic_suspected_runaway_slot_outputs"].fillna(0).sum()
)
sft_pair_validation_passed = bool(
    sft_controlled_pair_df["controlled_comparison_valid"].eq(True).all()
    and sft_controlled_pair_df["ordinary_repeat_reproducible"].eq(True).all()
)

manifest = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "results_dir": str(results_dir),
    "models": models,
    "regimes": list(regimes),
    "base_controlled_pair_validation_passed": base_pair_validation_passed,
    "base_all_dynamic_outputs_valid": base_all_dynamic_outputs_valid,
    "base_dynamic_invalid_output_count": base_dynamic_invalid_output_count,
    "base_dynamic_length_truncated_output_count": base_dynamic_length_truncated_output_count,
    "base_dynamic_suspected_runaway_output_count": base_dynamic_suspected_runaway_output_count,
    "sft_controlled_pair_validation_passed": sft_pair_validation_passed,
    "token_validation_passed": validation_failures.empty,
    "token_validation_failure_count": int(len(validation_failures)),
    "generated_files": [
        "all_eval_summary_side_by_side.csv",
        "all_eval_summary_side_by_side.md",
        "all_eval_summary_side_by_side.xlsx",
        "all_eval_token_usage_side_by_side.csv",
        "all_eval_token_usage_side_by_side.md",
        "all_featurewise_f1_all_regimes_long.csv",
        "all_featurewise_f1_all_regimes.xlsx",
        "all_featurewise_supported_micro_summary.csv",
        "all_featurewise_supported_micro_summary.md",
        "all_eval_token_usage_validation.csv",
        "all_eval_token_usage_validation.md",
        "base_controlled_pair_audit_side_by_side.csv",
        "base_controlled_pair_audit_side_by_side.md",
        "base_controlled_pair_audit_side_by_side.xlsx",
        "sft_controlled_pair_audit_side_by_side.csv",
        "sft_controlled_pair_audit_side_by_side.md",
        "sft_controlled_pair_audit_side_by_side.xlsx",
    ],
}

with (results_dir / "all_eval_postprocess_manifest.json").open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nGenerated cross-regime outputs:")
for filename in manifest["generated_files"]:
    print(" -", results_dir / filename)
print(" -", results_dir / "all_eval_postprocess_manifest.json")

if not validation_failures.empty:
    print("\nERROR: Token validation failed for these rows:")
    print(validation_failures.to_string(index=False))
    raise SystemExit(
        "Token totals do not match the underlying case files; do not use the combined outputs."
    )
else:
    print("\nToken totals in all summaries match the underlying case files.")

if not base_all_dynamic_outputs_valid:
    raise SystemExit(
        "The corrected typed decoder produced "
        f"{base_dynamic_invalid_output_count} invalid outputs across all models "
        f"({base_dynamic_length_truncated_output_count} length-truncated; "
        f"{base_dynamic_suspected_runaway_output_count} suspected runaway slots). "
        "Do not publish the combined outputs; inspect the base-pair audits."
    )
PY

# Optional dataset/tokenizer context audit. This is different from generated-token
# usage: it measures tokenizer-specific prompt/gold lengths and context limits.
if [[ "$RUN_CONTEXT_AUDIT" == "1" ]]; then
    require_file "$CODE_DIR/audit_context_length.py"
    require_file "$ROOT/datasets/test_nodule_base.json"
    require_file "$ROOT/datasets/test_nodule.json"

    echo
    echo "======================================================================"
    echo "RUNNING OPTIONAL TOKENIZER CONTEXT AUDITS"
    echo "======================================================================"

    "$PYTHON_BIN" "$CODE_DIR/audit_context_length.py" \
        --root "$ROOT" \
        --model-subdir base \
        --dataset "$ROOT/datasets/test_nodule_base.json" \
        --models "${MODELS[@]}" \
        --max-new-tokens "$BASE_CONTEXT_AUDIT_MAX_NEW_TOKENS" \
        --output-json "$RESULTS_DIR/base_context_audit.json"

    "$PYTHON_BIN" "$CODE_DIR/audit_context_length.py" \
        --root "$ROOT" \
        --model-subdir sft/run_002_merged \
        --dataset "$ROOT/datasets/test_nodule.json" \
        --models "${MODELS[@]}" \
        --max-new-tokens "$SFT_CONTEXT_AUDIT_MAX_NEW_TOKENS" \
        --output-json "$RESULTS_DIR/sft_context_audit.json"
fi

echo
echo "======================================================================"
echo "ALL FOUR EVALUATION REGIMES POST-PROCESSED SUCCESSFULLY"
echo "======================================================================"
echo "Results directory: $RESULTS_DIR"
