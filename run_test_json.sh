#!/usr/bin/env bash
# run_test_json.sh — run the DSL GP on every individual task in test.json.
#
# Usage:
#   ./run_test_json.sh [population_size [max_epochs]]
#
# Examples:
#   ./run_test_json.sh                   # pop=300, epochs=1500
#   ./run_test_json.sh 100 200           # quick smoke-test
#
# Logs are written to logs/<task_id>_<timestamp>.log
# Results summary is written to results_test_json.txt

set -euo pipefail

POP_SIZE="${1:-1000}"
MAX_EPOCHS="${2:-10000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/antonio/miniconda3/bin/python3.10
FEE_LIB="$SCRIPT_DIR/../fast-evolutionary-evaluation/build/lib.linux-x86_64-cpython-310"
DATA_DIR="$SCRIPT_DIR/data"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS="$SCRIPT_DIR/results_test_json.txt"

ARC_JSON_PATH="${ARC_JSON_PATH:-$SCRIPT_DIR/../learnable-dsl/data/1d-arc-merged.json}"

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------ #
# 1. Convert test.json → FEE format (one dir per task, idempotent)   #
# ------------------------------------------------------------------ #
echo "=== Preparing data ==="
$PYTHON "$SCRIPT_DIR/prepare_test_data.py" \
    --json "$ARC_JSON_PATH" \
    --data-dir "$DATA_DIR"

# ------------------------------------------------------------------ #
# 2. List individual task ids                                         #
# ------------------------------------------------------------------ #
TASK_IDS=$($PYTHON "$SCRIPT_DIR/prepare_test_data.py" \
    --json "$ARC_JSON_PATH" \
    --data-dir "$DATA_DIR" \
    --list-tasks)

echo ""
echo "=== Running GP for each task (pop=$POP_SIZE, epochs=$MAX_EPOCHS) ==="
echo "" > "$RESULTS"

for TASK_ID in $TASK_IDS; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="$LOG_DIR/${TASK_ID}_${TS}.log"

    echo "--- $TASK_ID ---"

    PYTHONPATH="$FEE_LIB" \
    RUNNING_TASK="$TASK_ID" \
    DATA_FOLDER="$DATA_DIR" \
    ARC_TASK_ID="" \
    ARC_JSON_PATH="$ARC_JSON_PATH" \
    POPULATION_SIZE="$POP_SIZE" \
    MAX_EPOCHS="$MAX_EPOCHS" \
    MUTATION_PROBABILITY=0.50 \
    CROSSOVER_PROBABILITY=0.50 \
    LLM_POPULATION_GENERATION=False \
        $PYTHON "$SCRIPT_DIR/gp.py" 2>&1 | tee "$LOG"

    BEST=$(grep "Epoch|.*|Top|" "$LOG" | tail -1 || echo "no result")
    echo "$TASK_ID: $BEST" >> "$RESULTS"
    echo "  Best: $BEST"
    echo ""
done

echo ""
echo "=== Results summary written to $RESULTS ==="
cat "$RESULTS"
