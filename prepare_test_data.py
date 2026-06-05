"""
prepare_test_data.py
--------------------
Convert learnable-dsl/data/test.json into FEE .txt format, one directory
per task so each task is solved independently by the GP.

Output structure (under DATA_FOLDER, default ./data):

  data/
    1d_hollow_48 / training / 1d_hollow_48.txt
                 / testing  / 1d_hollow_48.txt
    1d_hollow_12 / training / 1d_hollow_12.txt
                 / testing  / 1d_hollow_12.txt
    ...

Each .txt file is in FEE format.  training/ and testing/ contain the same
file — FEE uses the embedded test pair for evaluation.

Usage:
    python prepare_test_data.py [--json PATH] [--data-dir PATH] [--list-tasks]
"""

import argparse
import json
import os


# ------------------------------------------------------------------ #
# FEE .txt conversion                                                 #
# ------------------------------------------------------------------ #

def grid_to_lines(grid):
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    lines = [f"{h} {w}"]
    for row in grid:
        lines.append(" ".join(str(v) for v in row))
    return lines


def task_to_training_txt(task):
    """Training file: N-1 training pairs for fitness; last training pair held out as embedded test."""
    train = task["train"]
    fit_pairs = train[:-1] if len(train) > 1 else train
    held_out = train[-1]
    lines = [str(len(fit_pairs))]
    for pair in fit_pairs:
        lines += grid_to_lines(pair["input"])
        lines += grid_to_lines(pair["output"])
    lines += grid_to_lines(held_out["input"])
    lines += grid_to_lines(held_out["output"])
    lines += grid_to_lines(held_out["input"])
    return "\n".join(lines) + "\n"


def task_to_testing_txt(task):
    """Testing file: all training pairs + real test pair."""
    train, test = task["train"], task["test"]
    lines = [str(len(train))]
    for pair in train:
        lines += grid_to_lines(pair["input"])
        lines += grid_to_lines(pair["output"])
    tp = test[0]
    lines += grid_to_lines(tp["input"])
    lines += grid_to_lines(tp["output"])
    lines += grid_to_lines(tp["input"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default=os.path.join(
            os.path.dirname(__file__), "../learnable-dsl/data/test.json"
        ),
        help="Path to test.json",
    )
    ap.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Root output directory (default: ./data)",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task ids (one per line) and exit",
    )
    args = ap.parse_args()

    with open(args.json) as f:
        tasks = json.load(f)

    task_ids = sorted(tasks.keys())

    if args.list_tasks:
        for tid in task_ids:
            print(tid)
        return

    total = 0
    for tid in task_ids:
        task = tasks[tid]
        for split, txt in (
            ("training", task_to_training_txt(task)),
            ("testing",  task_to_testing_txt(task)),
        ):
            d = os.path.join(args.data_dir, tid, split)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{tid}.txt"), "w") as f:
                f.write(txt)
            total += 1

    print(f"Written {total} .txt files for {len(task_ids)} tasks under {args.data_dir}/")
    for tid in task_ids:
        print(f"  {tid}")


if __name__ == "__main__":
    main()
