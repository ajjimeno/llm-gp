"""
eval_stage4.py
--------------
Retrospective evaluation of Stage 4 (LLM selection using task name).

For each spurious selection in results_test_json.txt (Training=1.0, Testing<1.0),
we pair the spurious program with the known-correct program (from the partition
results), then ask the LLM to choose between them using only:
  - the task name
  - the training pairs (inputs + outputs)
  - the two candidate programs

Usage:
    LLM_NAME=openrouter python3 eval_stage4.py

The LLM model is set via LLM_NAME env var (openrouter | ollama | qwen | deepseek).
"""

import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
from llm import get_model
from structural_select import load_training_pairs

load_dotenv()

# ── Known correct programs per task type (from partition run) ─────────────────
CORRECT_PROGRAMS = {
    "1d_denoising_1c":  "dsl_clear_noise()",
    "1d_denoising_mc":  "dsl_recolor_to_dominant()",
    "1d_fill":          "dsl_fill_bbox()",
    "1d_flip":          "dsl_flip_horizontal_bbox()",
    "1d_hollow":        "dsl_hollow()",
    "1d_mirror":        "dsl_reflect_about_marker()",
    "1d_move_1p":       "dsl_move_right()",
    "1d_move_2p":       "dsl_move_right_n(get2())",
    "1d_move_3p":       "dsl_move_right_n(get3())",
    "1d_pcopy_1c":      "prog2(dsl_expand_point(), dsl_regroup_by_color())",
    "1d_pcopy_mc":      "prog2(dsl_expand_point(), dsl_regroup_by_color())",
    "1d_padded_fill":   "dsl_padded_fill()",
    # move_dp and move_2p_dp require longer programs — skip for now
}

WRAP = "prog2(dsl_segment(), prog2(prog2({inner}, dsl_render()), status_end()))"

def format_grid(row):
    """Format a 1D pixel row for display."""
    non_zero = [(i, v) for i, v in enumerate(row) if v != 0]
    w = len(row)
    display = "  " + " ".join(str(v) if v != 0 else "·" for v in row)
    return display

def build_prompt(task_id, task_type, spurious_inner, correct_inner, train_pairs):
    """Build the LLM prompt for Stage 4 selection."""
    sp = WRAP.format(inner=spurious_inner)
    cp = WRAP.format(inner=correct_inner)

    pairs_text = ""
    for k, (in_row, out_row) in enumerate(train_pairs):
        pairs_text += f"\n  Example {k+1}:\n"
        pairs_text += f"    Input:  {format_grid(in_row)}\n"
        pairs_text += f"    Output: {format_grid(out_row)}\n"

    return f"""You are evaluating two programs for an ARC visual reasoning task.
The task is called: "{task_id}"

Both programs achieve perfect accuracy on the training examples shown below,
but only one truly captures the rule. The other is coincidentally correct.

Training examples (· = background pixel):
{pairs_text}

Program A:
  {sp}

Program B:
  {cp}

Based on the task name "{task_type}" and the transformation shown in the examples,
which program is semantically correct?

Answer with ONLY "A" or "B" followed by one sentence explaining why."""


def extract_inner(wrapped_prog):
    """Strip the segment/render wrapper to get the inner program."""
    m = re.search(r'prog2\(dsl_segment\(\), prog2\(prog2\((.+), dsl_render\(\)\), status_end\(\)\)\)$',
                  wrapped_prog)
    return m.group(1) if m else wrapped_prog


def run_evaluation():
    llm_name = os.getenv("LLM_NAME", "openrouter")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    results_file = os.path.join(os.path.dirname(__file__), "results_test_json.txt")

    # Load spurious cases
    spurious_cases = []
    with open(results_file) as f:
        for line in f:
            m = re.match(r'^(\S+):\s+.*Training\|([0-9.]+)\|Testing\|([0-9.]+)', line)
            if not m: continue
            task_id, tr, te = m.group(1), float(m.group(2)), float(m.group(3))
            task_type = re.sub(r'_\d+$', '', task_id)
            prog_m = re.search(r'Top\|(prog2\([^|]+\))\|Length', line)
            if tr >= 1.0 and te < 1.0 and prog_m and task_type in CORRECT_PROGRAMS:
                spurious_cases.append({
                    'task_id': task_id,
                    'task_type': task_type,
                    'spurious_prog': prog_m.group(1),
                    'test_score': round(te, 3),
                })

    print(f"Evaluating Stage 4 on {len(spurious_cases)} spurious selections")
    print(f"LLM: {llm_name} (qwen/qwen-2.5-coder-32b-instruct via OpenRouter)")
    print()

    model = get_model(llm_name)

    results = []
    correct_count = 0
    skipped = 0

    for i, case in enumerate(spurious_cases):
        task_id   = case['task_id']
        task_type = case['task_type']
        spurious  = case['spurious_prog']
        correct   = WRAP.format(inner=CORRECT_PROGRAMS[task_type])

        # Load training pairs
        training_dir = os.path.join(data_dir, task_id, "training")
        train_pairs, _ = load_training_pairs(training_dir)
        if not train_pairs:
            skipped += 1
            continue

        spurious_inner = extract_inner(spurious)
        correct_inner  = CORRECT_PROGRAMS[task_type]

        # Randomise order (A/B assignment)
        import random
        if random.random() < 0.5:
            prog_a, prog_b = spurious_inner, correct_inner
            correct_letter = "B"
        else:
            prog_a, prog_b = correct_inner, spurious_inner
            correct_letter = "A"

        prompt = build_prompt(task_id, task_type, prog_a, prog_b, train_pairs)

        try:
            response = model(
                system_prompt="You are an expert in program synthesis and ARC visual reasoning.",
                user_prompt=prompt,
            )
            chosen = response.strip()[0].upper() if response.strip() else "?"
            llm_correct = (chosen == correct_letter)
            if llm_correct:
                correct_count += 1

            result = {
                'task_id':    task_id,
                'task_type':  task_type,
                'spurious':   spurious_inner[:60],
                'correct':    correct_inner[:60],
                'test_score': case['test_score'],
                'llm_chose':  chosen,
                'correct_was': correct_letter,
                'llm_correct': llm_correct,
                'response':   response[:200],
            }
            results.append(result)

            status = "✓" if llm_correct else "✗"
            print(f"[{i+1:3d}/{len(spurious_cases)}] {status} {task_id:<30} "
                  f"chosen={chosen} correct={correct_letter} test={case['test_score']}")

        except Exception as e:
            print(f"[{i+1:3d}] ERROR {task_id}: {e}")
            skipped += 1

    # Summary
    evaluated = len(results)
    print()
    print("=" * 60)
    print(f"Stage 4 Evaluation Results")
    print("=" * 60)
    print(f"Evaluated:  {evaluated}")
    print(f"Skipped:    {skipped}")
    print(f"Correct:    {correct_count}/{evaluated} = {correct_count/evaluated*100:.1f}%")
    print()

    # By task type
    by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        t = r['task_type']
        by_type[t]['total'] += 1
        if r['llm_correct']:
            by_type[t]['correct'] += 1
    print("By task type:")
    for t in sorted(by_type):
        c = by_type[t]['correct']
        n = by_type[t]['total']
        print(f"  {t:<25} {c}/{n} = {c/n*100:.0f}%")

    # Save results
    out_file = os.path.join(os.path.dirname(__file__), "eval_stage4_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    run_evaluation()
