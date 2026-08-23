#!/usr/bin/env bash

# Controlled paper-facing comparison of ordinary SFT and sparse XGrammar.
# Each model is loaded once; ordinary, repeat, and XGrammar requests all use
# that same vLLM process and identical prompt/sampling settings.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE_DIR="${CODE_DIR:-$ROOT/codes}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$CODE_DIR/sft_eval_vllm.py}"
PAIR_AUDIT_SCRIPT="${PAIR_AUDIT_SCRIPT:-$CODE_DIR/compare_sft_decoding_pair.py}"
FEATUREWISE_SCRIPT="${FEATUREWISE_SCRIPT:-$CODE_DIR/featurewise_eval.py}"
RESCORE_SCRIPT="${RESCORE_SCRIPT:-$CODE_DIR/rescore_eval_cases.py}"

TEST_FILE="${TEST_FILE:-$ROOT/datasets/test_nodule.json}"
SCHEMA_FILE="${SCHEMA_FILE:-$ROOT/schemas/lung_nodule_xgrammar_schema.json}"
FEATURE_SCHEMA="${FEATURE_SCHEMA:-$ROOT/schemas/lungs_pleura_nodule_template.json}"
MODEL_SUBDIR="${MODEL_SUBDIR:-sft/run_002_merged}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EVAL_CONTEXT_LIMIT="${EVAL_CONTEXT_LIMIT:-$MAX_MODEL_LEN}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
WORKERS="${WORKERS:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
LIMIT="${LIMIT:-}"
RUN_ORDINARY_REPEAT="${RUN_ORDINARY_REPEAT:-1}"
RUN_FEATUREWISE="${RUN_FEATUREWISE:-1}"
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
GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-xgrammar:no-fallback}"
JSON_BEGIN_TAG="${JSON_BEGIN_TAG:-<json>}"
JSON_END_TAG="${JSON_END_TAG:-</json>}"
JSON_INDENT="${JSON_INDENT:-4}"

ALL_MODELS=(
    llama3_2_1B
    gemma3_4B
    mistral_7B
    qwen2_5_7B
    llama3_1_8B
    llama3_1_70B
)

# Deliberately default to the one-model diagnostic. Use MODELS_CSV=all only
# after this comparison passes and its audit says controlled_comparison_valid.
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
        echo "ERROR: Missing or empty output: $path" >&2
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
    if [[ "$mode" == "v0_tagged_grammar" ]]; then
        mode_args+=(
            --schema_file "$SCHEMA_FILE"
            --json_begin_tag "$JSON_BEGIN_TAG"
            --json_end_tag "$JSON_END_TAG"
            --guided_decoding_backend "$GUIDED_DECODING_BACKEND"
            --json_indent "$JSON_INDENT"
        )
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

run_model_pair() {
    local model_name="${1//[[:space:]]/}"
    local model_dir="$ROOT/models/$model_name/$MODEL_SUBDIR"
    local result_dir="$RESULTS_DIR/$model_name"
    local served_name="${model_name}_run_002_controlled_pair"
    local server_log="$result_dir/vllm_sft_controlled_pair.log"

    local ordinary_cases="$result_dir/sft_eval_cases_controlled_ordinary.json"
    local ordinary_summary="$result_dir/sft_eval_summary_controlled_ordinary.json"
    local ordinary_log="$result_dir/sft_eval_controlled_ordinary.log"
    local repeat_cases="$result_dir/sft_eval_cases_controlled_ordinary_repeat.json"
    local repeat_summary="$result_dir/sft_eval_summary_controlled_ordinary_repeat.json"
    local repeat_log="$result_dir/sft_eval_controlled_ordinary_repeat.log"
    local xgrammar_cases="$result_dir/sft_eval_cases_controlled_xgrammar.json"
    local xgrammar_summary="$result_dir/sft_eval_summary_controlled_xgrammar.json"
    local xgrammar_log="$result_dir/sft_eval_controlled_xgrammar.log"
    local audit_json="$result_dir/sft_eval_controlled_pair_audit.json"
    local audit_log="$result_dir/sft_eval_controlled_pair_audit.log"
    local ordinary_featurewise="$result_dir/sft_featurewise_f1_controlled_ordinary"
    local xgrammar_featurewise="$result_dir/sft_featurewise_f1_controlled_xgrammar"

    echo
    echo "======================================================================"
    echo "CONTROLLED PAIR: $model_name"
    echo "MODEL PATH:      $model_dir"
    echo "GPUs / TP:       $GPUS / $TENSOR_PARALLEL_SIZE"
    echo "Workers / seqs:  $WORKERS / $MAX_NUM_SEQS"
    echo "======================================================================"

    if [[ ! -f "$model_dir/config.json" ]]; then
        echo "ERROR: Missing model: $model_dir/config.json" >&2
        return 1
    fi
    mkdir -p "$result_dir"

    local -a repeat_audit_args=()
    if [[ "$RUN_ORDINARY_REPEAT" == "1" ]]; then
        repeat_audit_args=(
            --ordinary_repeat_cases "$repeat_cases"
            --ordinary_repeat_summary "$repeat_summary"
        )
    fi

    if [[ "$AUDIT_ONLY" == "1" ]]; then
        echo "CPU-ONLY RESCORE/AUDIT: no SFT inference will run."
        rescore_arm "$ordinary_cases" "$ordinary_summary" || return 1
        if [[ "$RUN_ORDINARY_REPEAT" == "1" ]]; then
            rescore_arm "$repeat_cases" "$repeat_summary" || return 1
        fi
        rescore_arm "$xgrammar_cases" "$xgrammar_summary" || return 1

        local -a audit_featurewise_args=()
        if [[ "$RUN_FEATUREWISE" == "1" ]]; then
            "$PYTHON_BIN" "$FEATUREWISE_SCRIPT" \
                --eval_cases "$ordinary_cases" \
                --schema_file "$FEATURE_SCHEMA" \
                --output_csv "${ordinary_featurewise}.csv" \
                --output_json "${ordinary_featurewise}.json" \
                --min_similarity "$MIN_SIMILARITY" || return 1
            "$PYTHON_BIN" "$FEATUREWISE_SCRIPT" \
                --eval_cases "$xgrammar_cases" \
                --schema_file "$FEATURE_SCHEMA" \
                --output_csv "${xgrammar_featurewise}.csv" \
                --output_json "${xgrammar_featurewise}.json" \
                --min_similarity "$MIN_SIMILARITY" || return 1
            audit_featurewise_args=(
                --ordinary_featurewise "${ordinary_featurewise}.json"
                --xgrammar_featurewise "${xgrammar_featurewise}.json"
            )
        fi

        "$PYTHON_BIN" "$PAIR_AUDIT_SCRIPT" \
            --ordinary_cases "$ordinary_cases" \
            --xgrammar_cases "$xgrammar_cases" \
            --ordinary_summary "$ordinary_summary" \
            --xgrammar_summary "$xgrammar_summary" \
            --schema_file "$SCHEMA_FILE" \
            --tokenizer_dir "$model_dir" \
            --json_begin_tag "$JSON_BEGIN_TAG" \
            --json_end_tag "$JSON_END_TAG" \
            --output_json "$audit_json" \
            "${repeat_audit_args[@]}" \
            "${audit_featurewise_args[@]}" \
            2>&1 | tee "$audit_log" || return 1

        echo "CPU-only SFT/XGrammar rescore and audit passed: $audit_json"
        return 0
    fi

    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "ERROR: Port $PORT already has an OpenAI-compatible server." >&2
        return 1
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
        >"$server_log" 2>&1 &
    SERVER_PID=$!

    wait_for_server "$server_log" || return 1
    run_eval_arm unconstrained "$model_dir" "$served_name" \
        "$ordinary_cases" "$ordinary_summary" "$ordinary_log" || return 1

    if [[ "$RUN_ORDINARY_REPEAT" == "1" ]]; then
        run_eval_arm unconstrained "$model_dir" "$served_name" \
            "$repeat_cases" "$repeat_summary" "$repeat_log" || return 1
        repeat_audit_args=(
            --ordinary_repeat_cases "$repeat_cases"
            --ordinary_repeat_summary "$repeat_summary"
        )
    fi

    run_eval_arm v0_tagged_grammar "$model_dir" "$served_name" \
        "$xgrammar_cases" "$xgrammar_summary" "$xgrammar_log" || return 1

    local -a featurewise_audit_args=()
    if [[ "$RUN_FEATUREWISE" == "1" ]]; then
        "$PYTHON_BIN" "$FEATUREWISE_SCRIPT" \
            --eval_cases "$ordinary_cases" \
            --schema_file "$FEATURE_SCHEMA" \
            --output_csv "${ordinary_featurewise}.csv" \
            --output_json "${ordinary_featurewise}.json" \
            --min_similarity "$MIN_SIMILARITY" || return 1
        "$PYTHON_BIN" "$FEATUREWISE_SCRIPT" \
            --eval_cases "$xgrammar_cases" \
            --schema_file "$FEATURE_SCHEMA" \
            --output_csv "${xgrammar_featurewise}.csv" \
            --output_json "${xgrammar_featurewise}.json" \
            --min_similarity "$MIN_SIMILARITY" || return 1
        featurewise_audit_args=(
            --ordinary_featurewise "${ordinary_featurewise}.json"
            --xgrammar_featurewise "${xgrammar_featurewise}.json"
        )
    fi

    if ! "$PYTHON_BIN" "$PAIR_AUDIT_SCRIPT" \
        --ordinary_cases "$ordinary_cases" \
        --xgrammar_cases "$xgrammar_cases" \
        --ordinary_summary "$ordinary_summary" \
        --xgrammar_summary "$xgrammar_summary" \
        --schema_file "$SCHEMA_FILE" \
        --tokenizer_dir "$model_dir" \
        --json_begin_tag "$JSON_BEGIN_TAG" \
        --json_end_tag "$JSON_END_TAG" \
        --output_json "$audit_json" \
        "${repeat_audit_args[@]}" \
        "${featurewise_audit_args[@]}" \
        2>&1 | tee "$audit_log"; then
        echo "ERROR: Controlled-pair audit failed." >&2
        return 1
    fi

    cleanup_server
    echo "Controlled pair passed for $model_name: $audit_json"
}

for required in \
    "$EVAL_SCRIPT" \
    "$PAIR_AUDIT_SCRIPT" \
    "$FEATUREWISE_SCRIPT" \
    "$RESCORE_SCRIPT" \
    "$CODE_DIR/date_normalization.py" \
    "$CODE_DIR/nodule_scoring.py" \
    "$TEST_FILE" \
    "$SCHEMA_FILE" \
    "$FEATURE_SCHEMA"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: Required file not found: $required" >&2
        exit 1
    fi
done

if [[ "$WORKERS" != "1" || "$MAX_NUM_SEQS" != "1" ]]; then
    echo "WARNING: Controlled diagnostic is intended for WORKERS=1 MAX_NUM_SEQS=1." >&2
fi

for model_name in "${MODELS[@]}"; do
    if ! run_model_pair "$model_name"; then
        FAILED_MODELS+=("$model_name")
        cleanup_server
    fi
done

if [[ "${#FAILED_MODELS[@]}" -gt 0 ]]; then
    echo "Failed models: ${FAILED_MODELS[*]}" >&2
    exit 1
fi

echo "All controlled SFT/XGrammar pairs completed successfully."
