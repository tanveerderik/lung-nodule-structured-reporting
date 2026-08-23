#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE_DIR="${CODE_DIR:-$ROOT/codes}"
SPARSE_RESULTS_DIR="${SPARSE_RESULTS_DIR:-$ROOT/results_corrected}"
DENSE_RESULTS_DIR="${DENSE_RESULTS_DIR:-$ROOT/results_base_dense}"
OUTPUT_DIR="${OUTPUT_DIR:-$DENSE_RESULTS_DIR/base_prompt_regime_comparison}"
MODELS_CSV="${MODELS_CSV:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_SCHEMA="${FEATURE_SCHEMA:-$ROOT/schemas/lungs_pleura_nodule_template.json}"
RAW_SCHEMA="${RAW_SCHEMA:-$ROOT/schemas/lung_nodule_xgrammar_schema.json}"
REBUILD_PREDICTIONS="${REBUILD_PREDICTIONS:-1}"
RUN_FEATUREWISE="${RUN_FEATUREWISE:-1}"

ALL_MODELS=(
    llama3_2_1B gemma3_4B mistral_7B qwen2_5_7B llama3_1_8B llama3_1_70B
)
if [[ "$MODELS_CSV" == "all" ]]; then
    MODELS=("${ALL_MODELS[@]}")
else
    IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
fi

require_nonempty() {
    [[ -s "$1" ]] || { echo "ERROR: Missing or empty file: $1" >&2; exit 1; }
}

require_nonempty "$FEATURE_SCHEMA"
require_nonempty "$RAW_SCHEMA"
require_nonempty "$CODE_DIR/rescore_eval_cases.py"
require_nonempty "$CODE_DIR/featurewise_eval.py"

process_arm() {
    local model="$1"
    local arm="$2"
    local cases_file="$3"
    local summary_file="$4"
    local feature_stem="$5"
    local raw_validation_mode="$6"
    require_nonempty "$cases_file"
    require_nonempty "$summary_file"

    if [[ "$REBUILD_PREDICTIONS" == "1" ]]; then
        PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" "$CODE_DIR/rescore_eval_cases.py" \
            --eval_cases "$cases_file" \
            --summary_json "$summary_file" \
            --schema_file "$RAW_SCHEMA" \
            --arm_label "$model/$arm" \
            --raw_validation_mode "$raw_validation_mode" \
            --in_place
    fi

    if [[ "$RUN_FEATUREWISE" == "1" ]]; then
        PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" "$CODE_DIR/featurewise_eval.py" \
            --eval_cases "$cases_file" \
            --schema_file "$FEATURE_SCHEMA" \
            --output_csv "${feature_stem}.csv" \
            --output_json "${feature_stem}.json"
    fi
}

for model in "${MODELS[@]}"; do
    process_arm \
        "$model" "base_sparse" \
        "$SPARSE_RESULTS_DIR/$model/base_eval_cases_controlled_ordinary.json" \
        "$SPARSE_RESULTS_DIR/$model/base_eval_summary_controlled_ordinary.json" \
        "$SPARSE_RESULTS_DIR/$model/base_featurewise_f1_controlled_ordinary" \
        "json_schema"
    process_arm \
        "$model" "base_dense" \
        "$DENSE_RESULTS_DIR/$model/base_eval_cases_controlled_ordinary.json" \
        "$DENSE_RESULTS_DIR/$model/base_eval_summary_controlled_ordinary.json" \
        "$DENSE_RESULTS_DIR/$model/base_featurewise_f1_controlled_ordinary" \
        "json_schema"
    process_arm \
        "$model" "base_dc_dense" \
        "$DENSE_RESULTS_DIR/$model/base_eval_cases_controlled_dynamic_template.json" \
        "$DENSE_RESULTS_DIR/$model/base_eval_summary_controlled_dynamic_template.json" \
        "$DENSE_RESULTS_DIR/$model/base_featurewise_f1_controlled_dynamic_template" \
        "dynamic_template"
done

"$PYTHON_BIN" "$CODE_DIR/compare_base_prompt_regimes.py" \
    --sparse-results "$SPARSE_RESULTS_DIR" \
    --dense-results "$DENSE_RESULTS_DIR" \
    --schema-file "$RAW_SCHEMA" \
    --models "$MODELS_CSV" \
    --output-dir "$OUTPUT_DIR"
