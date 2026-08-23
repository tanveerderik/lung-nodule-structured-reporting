#!/usr/bin/env bash

# Run the prompt-controlled dense ablation:
#   ordinary Base + dense nullable instruction
#   Base+DC       + the identical dense nullable instruction
#
# Existing sparse-prompt Base results remain in SPARSE_RESULTS_DIR and are not
# touched.  This script writes to a separate DENSE_RESULTS_DIR so the two
# ordinary Base conditions cannot overwrite one another.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE_DIR="${CODE_DIR:-$ROOT/codes}"
SPARSE_TEST_FILE="${SPARSE_TEST_FILE:-$ROOT/datasets/test_nodule_base.json}"
DENSE_TEST_FILE="${DENSE_TEST_FILE:-$ROOT/datasets/test_nodule_base_dense.json}"
SPARSE_RESULTS_DIR="${SPARSE_RESULTS_DIR:-$ROOT/results_corrected}"
DENSE_RESULTS_DIR="${DENSE_RESULTS_DIR:-$ROOT/results_base_dense}"
BASE_PAIR_LAUNCHER="${BASE_PAIR_LAUNCHER:-$CODE_DIR/run_base_dynamic_controlled_pair.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for required in \
    "$SPARSE_TEST_FILE" \
    "$DENSE_TEST_FILE" \
    "$CODE_DIR/audit_base_prompt_datasets.py" \
    "$BASE_PAIR_LAUNCHER"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: Required file not found: $required" >&2
        exit 1
    fi
done

if [[ "$DENSE_RESULTS_DIR" == "$SPARSE_RESULTS_DIR" ]]; then
    echo "ERROR: DENSE_RESULTS_DIR must differ from SPARSE_RESULTS_DIR." >&2
    echo "Using one directory would overwrite the sparse ordinary Base outputs." >&2
    exit 1
fi

mkdir -p "$DENSE_RESULTS_DIR"
"$PYTHON_BIN" "$CODE_DIR/audit_base_prompt_datasets.py" \
    --sparse "$SPARSE_TEST_FILE" \
    --dense "$DENSE_TEST_FILE" \
    --output-json "$DENSE_RESULTS_DIR/base_prompt_dataset_audit.json"

echo
echo "======================================================================"
echo "DENSE PROMPT-CONTROLLED BASE PAIR"
echo "Sparse results retained at: $SPARSE_RESULTS_DIR"
echo "Dense results written to:   $DENSE_RESULTS_DIR"
echo "Dense test dataset:         $DENSE_TEST_FILE"
echo "======================================================================"

# A repeat is useful for a fresh reproducibility audit, but it doubles the
# ordinary dense arm. Set RUN_ORDINARY_REPEAT=1 explicitly if desired.
export RUN_ORDINARY=1
export RUN_ORDINARY_REPEAT="${RUN_ORDINARY_REPEAT:-0}"
export TEST_FILE="$DENSE_TEST_FILE"
export RESULTS_DIR="$DENSE_RESULTS_DIR"
export ROOT CODE_DIR PYTHON_BIN

bash "$BASE_PAIR_LAUNCHER"
