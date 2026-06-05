#!/usr/bin/env bash
# run_partition.sh — run the DSL GP on task TYPES (not instances).
#
# Each GP run evaluates programs against all 50 instances of a task type
# simultaneously. FEE averages the score across all instance files, so
# spurious solutions that only work on a subset of instances are naturally
# filtered out — no test set needed.
#
# Usage:
#   ./run_partition.sh [population_size [max_epochs]]
#
# Examples:
#   ./run_partition.sh                   # pop=1000, epochs=10000
#   ./run_partition.sh 300 1500          # quick smoke-test
#
# Results written to results_partition.txt (does NOT touch results_test_json.txt)

set -euo pipefail

POP_SIZE="${1:-1000}"
MAX_EPOCHS="${2:-10000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/antonio/miniconda3/bin/python3.10
FEE_LIB="$SCRIPT_DIR/../fast-evolutionary-evaluation/build/lib.linux-x86_64-cpython-310"
PARTITION_DIR="$SCRIPT_DIR/data/partitions"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS="$SCRIPT_DIR/results_partition.txt"

mkdir -p "$LOG_DIR"
echo "" > "$RESULTS"

TASK_TYPES=(
    1d_denoising_1c
    1d_denoising_mc
    1d_fill
    1d_flip
    1d_hollow
    1d_mirror
    1d_move_1p
    1d_move_2p
    1d_move_2p_dp
    1d_move_3p
    1d_move_dp
    1d_padded_fill
    1d_pcopy_1c
    1d_pcopy_mc
    1d_recolor_cmp
    1d_recolor_cnt
    1d_recolor_oe
    1d_scale_dp
)

echo "=== Partition GP: pop=$POP_SIZE epochs=$MAX_EPOCHS ==="
echo "Each run evaluates programs across all 50 instances of a task type."
echo ""

for TASK_TYPE in "${TASK_TYPES[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="$LOG_DIR/partition_${TASK_TYPE}_${TS}.log"

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

echo ""
echo "=== Results written to $RESULTS ==="
cat "$RESULTS"
