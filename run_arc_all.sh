#!/usr/bin/env bash
# run_arc_all.sh — run the DSL GP on every task in the real ARC tasks.json.
#
# Usage:
#   ./run_arc_all.sh [population_size [max_epochs]]
#
# Examples:
#   ./run_arc_all.sh                   # pop=300, epochs=1500
#   ./run_arc_all.sh 100 200           # quick smoke-test
#
# Override the ARC JSON source:
#   ARC_JSON_PATH=/path/to/tasks.json ./run_arc_all.sh
#
# Logs are written to logs/arc_<task_id>_<timestamp>.log
# Results summary is written to results_arc.txt
#
# The real ARC-AGI dataset can be obtained from:
#   https://github.com/fchollet/ARC-AGI  (training + evaluation sets)
#   https://github.com/fchollet/ARC-AGI-2 (2025 edition)

set -euo pipefail

POP_SIZE="${1:-300}"
MAX_EPOCHS="${2:-1500}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/antonio/miniconda3/bin/python3.10
FEE_LIB="$SCRIPT_DIR/../fast-evolutionary-evaluation/build/lib.linux-x86_64-cpython-310"
DATA_DIR="$SCRIPT_DIR/data"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS="$SCRIPT_DIR/results_arc.txt"

ARC_JSON_PATH="${ARC_JSON_PATH:-$SCRIPT_DIR/../learnable-dsl/data/tasks.json}"

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------ #
# 1. Convert tasks.json → per-task FEE format (idempotent)           #
# ------------------------------------------------------------------ #
echo "=== Preparing data from $ARC_JSON_PATH ==="
$PYTHON "$SCRIPT_DIR/prepare_test_data.py" \
    --json "$ARC_JSON_PATH" \
    --data-dir "$DATA_DIR"

# ------------------------------------------------------------------ #
# 2. List task ids                                                     #
# ------------------------------------------------------------------ #
TASK_IDS=$($PYTHON "$SCRIPT_DIR/prepare_test_data.py" \
    --json "$ARC_JSON_PATH" \
    --data-dir "$DATA_DIR" \
    --list-tasks)

echo ""
echo "=== Running GP for each ARC task (pop=$POP_SIZE, epochs=$MAX_EPOCHS) ==="
echo "" > "$RESULTS"

for TASK_ID in $TASK_IDS; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="$LOG_DIR/arc_${TASK_ID}_${TS}.log"

    echo "--- $TASK_ID ---"

    PYTHONPATH="$FEE_LIB" \
    RUNNING_TASK="$TASK_ID" \
    DATA_FOLDER="$DATA_DIR" \
    ARC_TASK_ID="$TASK_ID" \
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
