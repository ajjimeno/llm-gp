#!/usr/bin/env bash
# run_arc.sh — prepare data and launch the GP for a single ARC task.
#
# Usage:
#   ./run_arc.sh <task_id> [population_size]
#
# Example:
#   ./run_arc.sh 00576224 300
#
# Environment variables (can also be set in .env):
#   ARC_JSON_PATH  path to the ARC tasks JSON (default: ../learnable-dsl/tasks.json)
#   DATA_FOLDER    where converted task files are written  (default: ./data)
#   LLM_NAME       LLM to use for guided mutation (default: empty = no LLM)

set -euo pipefail

TASK_ID="${1:?Usage: $0 <task_id> [population_size]}"
POP_SIZE="${2:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARC_JSON_PATH="${ARC_JSON_PATH:-$SCRIPT_DIR/../learnable-dsl/tasks.json}"
DATA_FOLDER="${DATA_FOLDER:-$SCRIPT_DIR/data}"

PYTHON=/home/antonio/miniconda3/bin/python3.10
FEE_LIB="$SCRIPT_DIR/../fast-evolutionary-evaluation/build/lib.linux-x86_64-cpython-310"

# ------------------------------------------------------------------ #
# 1. Convert the ARC task to FEE .txt format                          #
# ------------------------------------------------------------------ #
TRAINING_DIR="$DATA_FOLDER/arc/training"
TESTING_DIR="$DATA_FOLDER/arc/testing"
mkdir -p "$TRAINING_DIR" "$TESTING_DIR"

$PYTHON "$SCRIPT_DIR/../fast-evolutionary-evaluation/arc_json_to_fee.py" \
    "$ARC_JSON_PATH" \
    --output-dir "$TRAINING_DIR" \
    --task-ids "$TASK_ID"

# Testing dir uses the same files (each .txt has both training pairs and the test pair)
cp "$TRAINING_DIR/$TASK_ID.txt" "$TESTING_DIR/$TASK_ID.txt"

echo "Task files written to $TRAINING_DIR"

# ------------------------------------------------------------------ #
# 2. Run the GP                                                        #
# ------------------------------------------------------------------ #
PYTHONPATH="$FEE_LIB" \
RUNNING_TASK=arc \
ARC_TASK_ID="$TASK_ID" \
ARC_JSON_PATH="$ARC_JSON_PATH" \
DATA_FOLDER="$DATA_FOLDER" \
POPULATION_SIZE="$POP_SIZE" \
MUTATION_PROBABILITY=0.50 \
CROSSOVER_PROBABILITY=0.50 \
LLM_POPULATION_GENERATION=False \
    $PYTHON "$SCRIPT_DIR/gp.py"
