#!/usr/bin/env bash
set -euo pipefail

# CPU-only Stage 2 rescore/bootstrap from archived predictions.
# No model inference is performed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-$REPO_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT}"
SCHEMA="${SCHEMA:-$REPO_ROOT/stage2/schemas/lungs_pleura_nodule_template.json}"
SCORER_DIR="${SCORER_DIR:-$SCRIPT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results_weighted_bootstrap}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-$SCRIPT_DIR/bootstrap_stage2_ci.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ITERATIONS="${ITERATIONS:-10000}"
SEED="${SEED:-20260822}"

for path in "$BOOTSTRAP_SCRIPT" "$SCHEMA" "$SCORER_DIR/featurewise_eval.py"; do
    if [[ ! -s "$path" ]]; then
        echo "ERROR: Missing or empty file: $path" >&2
        exit 1
    fi
done

# The bootstrap script expects both of these archived result trees under
# RESULTS_ROOT so that all five deployment configurations are case-locked.
for dir in "$RESULTS_ROOT/results" "$RESULTS_ROOT/results_dense_prompt"; do
    if [[ ! -d "$dir" ]]; then
        echo "ERROR: Missing result directory: $dir" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

PYTHONPATH="$SCORER_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$BOOTSTRAP_SCRIPT" \
    --results-root "$RESULTS_ROOT" \
    --schema "$SCHEMA" \
    --scorer-dir "$SCORER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --seed "$SEED"

echo
echo "Stage 2 bootstrap complete."
echo "Micro-F1:                 $OUTPUT_DIR/bootstrap_overall_f1.csv"
echo "Gold-support weighted F1: $OUTPUT_DIR/bootstrap_gold_support_weighted_f1.csv"
echo "Weighted paired deltas:   $OUTPUT_DIR/bootstrap_gold_support_weighted_deltas.csv"
echo "Manifest:                 $OUTPUT_DIR/bootstrap_manifest.json"
