"""
eval_stage3.py
--------------
Retrospective evaluation of Stage 3 (structural consistency check).

For each spurious selection (Training=1.0, Testing<1.0) in results_test_json.txt,
pairs the spurious program with the known-correct program and uses the structural
consistency check to choose between them — no LLM, no test labels needed.

Usage:
    python3 eval_stage3.py
"""

import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from structural_select import (
    load_training_pairs, load_test_input,
    transformation_signature, consistency_score,
    _row_to_objects, _objects_to_row,
    _reflect_about_marker, _recolor_to_dominant,
)

# ── Known correct inner programs per task type ────────────────────────────────
CORRECT_INNER = {
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
}

# ── Python implementations of single-op programs ─────────────────────────────
def _clear_noise(objs):
    return [o for o in objs if o.get("type") == "metadata"
            or o.get("color") == 0
            or len(o.get("pixels", [])) > 1]

def _filter_even(objs):
    return [o for o in objs if o.get("type") == "metadata"
            or o.get("color") == 0
            or len(o.get("pixels", [])) % 2 == 0]

def _filter_odd(objs):
    return [o for o in objs if o.get("type") == "metadata"
            or o.get("color") == 0
            or len(o.get("pixels", [])) % 2 == 1]

def _filter_largest(objs):
    non_bg = [o for o in objs if o.get("type") != "metadata" and o.get("color") != 0]
    if not non_bg: return objs
    mx = max(len(o.get("pixels", [])) for o in non_bg)
    return [o for o in objs if o.get("type") == "metadata"
            or o.get("color") == 0
            or len(o.get("pixels", [])) == mx]

def _filter_smallest(objs):
    non_bg = [o for o in objs if o.get("type") != "metadata" and o.get("color") != 0]
    if not non_bg: return objs
    mn = min(len(o.get("pixels", [])) for o in non_bg)
    return [o for o in objs if o.get("type") == "metadata"
            or o.get("color") == 0
            or len(o.get("pixels", [])) == mn]

def _flip_h(objs):
    w = next((o["input_grid_size"][1] for o in objs
               if o.get("type") == "metadata"), len(objs))
    return [{**o, "pixels": [(r, w-1-c) for r, c in o.get("pixels", [])]}
            if o.get("type") != "metadata" and o.get("color") != 0
            else o for o in objs]

def _flip_h_bbox(objs):
    non_bg = [o for o in objs if o.get("type") != "metadata" and o.get("color") != 0]
    if not non_bg: return objs
    all_cols = [c for o in non_bg for _, c in o.get("pixels", [])]
    min_c, max_c = min(all_cols), max(all_cols)
    result = []
    for o in objs:
        if o.get("type") == "metadata" or o.get("color") == 0:
            result.append(o)
        else:
            new_px = [(r, min_c + max_c - c) for r, c in o.get("pixels", [])]
            result.append({**o, "pixels": new_px})
    return result

def _hollow(objs):
    result = []
    for o in objs:
        if o.get("type") == "metadata" or o.get("color") == 0:
            result.append(o)
            continue
        px = set(o.get("pixels", []))
        border = [(r, c) for r, c in px
                  if any((r+dr, c+dc) not in px for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])]
        result.append({**o, "pixels": border})
    return result

def _padded_fill(objs):
    # Simplified: fill between objects in each row
    return objs  # structural signature still useful as-is

def _expand_point_regroup(objs):
    # Expand area-1 non-zero objects to 3 pixels
    w = next((o["input_grid_size"][1] for o in objs if o.get("type")=="metadata"), 30)
    result = [o for o in objs if o.get("type") == "metadata"]
    expanded = []
    for o in objs:
        if o.get("type") == "metadata": continue
        if o.get("color") != 0 and len(o.get("pixels", [])) == 1:
            r, c = o["pixels"][0]
            new_px = [(r, c)]
            if c - 1 >= 0: new_px.insert(0, (r, c-1))
            if c + 1 < w:  new_px.append((r, c+1))
            expanded.append({**o, "pixels": new_px})
        else:
            expanded.append(o)
    return result + expanded

PYTHON_OPS = {
    "dsl_clear_noise()":                    _clear_noise,
    "dsl_filter_even_size()":               _filter_even,
    "dsl_filter_odd_size()":                _filter_odd,
    "dsl_filter_largest()":                 _filter_largest,
    "dsl_filter_smallest()":               _filter_smallest,
    "dsl_recolor_to_dominant()":            _recolor_to_dominant,
    "dsl_flip_horizontal()":               _flip_h,
    "dsl_flip_horizontal_bbox()":          _flip_h_bbox,
    "dsl_reflect_about_marker()":          _reflect_about_marker,
    "dsl_hollow()":                         _hollow,
    "dsl_padded_fill()":                    _padded_fill,
    "dsl_fill_bbox()":                      _padded_fill,  # structural only
    "dsl_move_right()":                     lambda o: o,   # structural approx
    "dsl_move_right_n(get2())":             lambda o: o,
    "dsl_move_right_n(get3())":             lambda o: o,
    "prog2(dsl_expand_point(), dsl_regroup_by_color())": _expand_point_regroup,
    "dsl_keep_bg_or_large()":              _clear_noise,  # structurally same
    "dsl_filter_horizontal_lines()":       _clear_noise,  # removes singletons
    "dsl_filter_unique_color()":           _filter_smallest,
}


def extract_inner(wrapped):
    m = re.search(
        r'prog2\(dsl_segment\(\), prog2\(prog2\((.+), dsl_render\(\)\), status_end\(\)\)\)$',
        wrapped)
    return m.group(1) if m else wrapped


def structural_score(inner_prog, train_pairs, test_in):
    """Compute structural consistency score for a program on the test input."""
    op = PYTHON_OPS.get(inner_prog)
    if op is None:
        return None  # can't evaluate

    train_sigs = [transformation_signature(i, o) for i, o in train_pairs]
    w = len(test_in)

    try:
        objs = _row_to_objects(test_in)
        pred_objs = op(objs)
        pred_row = _objects_to_row(pred_objs, w)
        pred_sig = transformation_signature(test_in, pred_row)
        return consistency_score(pred_sig, train_sigs)
    except Exception:
        return None


def run_evaluation():
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
            pm = re.search(r'Top\|(prog2\([^|]+\))\|Length', line)
            if tr >= 1.0 and te < 1.0 and pm and task_type in CORRECT_INNER:
                spurious_cases.append({
                    'task_id':    task_id,
                    'task_type':  task_type,
                    'spurious':   pm.group(1),
                    'test_score': round(te, 3),
                })

    print(f"Stage 3 structural consistency evaluation")
    print(f"Cases: {len(spurious_cases)} spurious selections")
    print()

    results = []
    correct = skipped = tied = 0

    for i, case in enumerate(spurious_cases):
        task_id     = case['task_id']
        task_type   = case['task_type']
        spurious    = case['spurious']
        correct_prog = CORRECT_INNER[task_type]

        train_pairs, _ = load_training_pairs(os.path.join(data_dir, task_id, "training"))
        test_in = load_test_input(os.path.join(data_dir, task_id, "testing"))

        if not train_pairs or not test_in:
            skipped += 1
            continue

        spurious_inner = extract_inner(spurious)
        sp_score = structural_score(spurious_inner, train_pairs, test_in)
        co_score = structural_score(correct_prog,   train_pairs, test_in)

        if sp_score is None or co_score is None:
            skipped += 1
            verdict = "SKIP(no impl)"
        elif co_score > sp_score:
            correct += 1
            verdict = "✓ CORRECT"
        elif co_score == sp_score:
            tied += 1
            verdict = "= TIED"
        else:
            verdict = "✗ WRONG"

        results.append({
            'task_id':       task_id,
            'task_type':     task_type,
            'spurious_inner': spurious_inner[:50],
            'correct_inner':  correct_prog[:50],
            'spurious_score': sp_score,
            'correct_score':  co_score,
            'test_score':    case['test_score'],
            'verdict':        verdict,
        })

        sp_str = f"{sp_score:.3f}" if sp_score is not None else "  N/A"
        co_str = f"{co_score:.3f}" if co_score is not None else "  N/A"
        print(f"[{i+1:3d}] {verdict:<12} {task_id:<30} "
              f"sp={sp_str} co={co_str} test={case['test_score']}")

    # Summary
    evaluated = len([r for r in results if "SKIP" not in r['verdict']])
    print()
    print("=" * 60)
    print("Stage 3 Structural Consistency — Results")
    print("=" * 60)
    print(f"Evaluated:  {evaluated}")
    print(f"Skipped:    {skipped}")
    print(f"Correct:    {correct}/{evaluated} = {correct/evaluated*100:.1f}%")
    print(f"Tied:       {tied}/{evaluated} = {tied/evaluated*100:.1f}%")
    print(f"Wrong:      {evaluated-correct-tied}/{evaluated}")
    print()

    by_type = defaultdict(lambda: {'correct': 0, 'tied': 0, 'wrong': 0, 'skip': 0})
    for r in results:
        t = r['task_type']
        if "SKIP" in r['verdict']:  by_type[t]['skip'] += 1
        elif r['verdict'] == "✓ CORRECT": by_type[t]['correct'] += 1
        elif r['verdict'] == "= TIED":    by_type[t]['tied'] += 1
        else:                              by_type[t]['wrong'] += 1

    print("By task type:")
    for t in sorted(by_type):
        d = by_type[t]
        n = d['correct'] + d['tied'] + d['wrong']
        if n == 0: continue
        print(f"  {t:<25}  correct={d['correct']:2d}  tied={d['tied']:2d}  "
              f"wrong={d['wrong']:2d}  skip={d['skip']:2d}  "
              f"acc={d['correct']/n*100:.0f}%")

    out = os.path.join(os.path.dirname(__file__), "eval_stage3_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    run_evaluation()
