#!/usr/bin/env bash
# run_partition_focus.sh — re-run the 4 unsolved partition task types with
# larger population, more epochs, and the new dsl_reflect_about_marker primitive.

set -euo pipefail

POP_SIZE="${1:-2000}"
MAX_EPOCHS="${2:-200000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/antonio/miniconda3/bin/python3.10
FEE_LIB="$SCRIPT_DIR/../fast-evolutionary-evaluation/build/lib.linux-x86_64-cpython-310"
PARTITION_DIR="$SCRIPT_DIR/data/partitions"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS="$SCRIPT_DIR/results_partition_focus.txt"

mkdir -p "$LOG_DIR"
echo "" > "$RESULTS"

TASK_TYPES=(
    1d_mirror
    1d_recolor_cmp
    1d_recolor_cnt
    1d_recolor_oe
)

echo "=== Focused partition GP: pop=$POP_SIZE epochs=$MAX_EPOCHS ==="
echo "New primitives: dsl_reflect_about_marker, dsl_get_marker_distance,"
echo "                dsl_filter_odd_size, dsl_filter_even_size, dsl_filter_size_n"
echo ""

for TASK_TYPE in "${TASK_TYPES[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="$LOG_DIR/partition_focus_${TASK_TYPE}_${TS}.log"

    echo "--- $TASK_TYPE ---"

    PYTHONPATH="$FEE_LIB" \
    RUNNING_TASK="$TASK_TYPE" \
    DATA_FOLDER="$PARTITION_DIR" \
    ARC_TASK_ID="" \
    ARC_JSON_PATH="" \
    POPULATION_SIZE="$POP_SIZE" \
    MAX_EPOCHS="$MAX_EPOCHS" \
    MUTATION_PROBABILITY=0.50 \
    CROSSOVER_PROBABILITY=0.50 \
    LLM_POPULATION_GENERATION=False \
    LLM_ELITE_MUTATION=False \
        $PYTHON "$SCRIPT_DIR/gp.py" 2>&1 | tee "$LOG"

    BEST=$(grep "Epoch|.*|Top|" "$LOG" | tail -1 || echo "no result")
    echo "$TASK_TYPE: $BEST" >> "$RESULTS"
    echo "  Best: $BEST"
    echo ""
done

echo "=== Results written to $RESULTS ==="
cat "$RESULTS"
