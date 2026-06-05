"""
structural_select.py
--------------------
Post-hoc selection among Training=1.0 candidate programs using structural
consistency: does the candidate's prediction on the test input follow the
same transformation pattern as the training pairs?

Works without knowing the expected test output.
"""

import glob
import os
import shutil
import tempfile


# ── 1D object detection ───────────────────────────────────────────────────────

def find_objects_1d(row):
    """Connected non-zero, same-colour runs in a 1D grid row."""
    segs, i = [], 0
    while i < len(row):
        if row[i] != 0:
            j = i
            while j < len(row) and row[j] == row[i]:
                j += 1
            segs.append({"start": i, "end": j - 1, "color": row[i], "size": j - i})
            i = j
        else:
            i += 1
    return segs


# ── Transformation signature ──────────────────────────────────────────────────

def transformation_signature(in_row, out_row):
    """
    Extract a dict of structural features describing how in_row became out_row.
    All features are comparable across pairs of the same task type.
    """
    i_segs = find_objects_1d(in_row)
    o_segs = find_objects_1d(out_row)

    px_in  = sum(s["size"] for s in i_segs)
    px_out = sum(s["size"] for s in o_segs)
    max_in  = max((s["size"] for s in i_segs), default=0)
    max_out = max((s["size"] for s in o_segs), default=0)

    # Do singleton objects (area=1) stay at the same positions?
    in_singletons  = frozenset(s["start"] for s in i_segs if s["size"] == 1)
    out_singletons = frozenset(s["start"] for s in o_segs if s["size"] == 1)
    singletons_stay = (in_singletons == out_singletons)

    # Do the sizes of objects match (same multiset of sizes)?
    in_sizes  = tuple(sorted(s["size"] for s in i_segs))
    out_sizes = tuple(sorted(s["size"] for s in o_segs))

    return {
        "n_objects_in":      len(i_segs),
        "n_objects_out":     len(o_segs),
        "n_delta":           len(o_segs) - len(i_segs),
        "px_in":             px_in,
        "px_out":            px_out,
        "px_delta":          px_out - px_in,
        "max_size_in":       max_in,
        "max_size_out":      max_out,
        "max_preserved":     max_in == max_out,
        "size_multiset_preserved": in_sizes == out_sizes,
        "singletons_in":     sum(1 for s in i_segs if s["size"] == 1),
        "singletons_out":    sum(1 for s in o_segs if s["size"] == 1),
        "singletons_stay":   singletons_stay,
        "colors_in":         frozenset(s["color"] for s in i_segs),
        "colors_out":        frozenset(s["color"] for s in o_segs),
        "colors_preserved":  (frozenset(s["color"] for s in i_segs) ==
                              frozenset(s["color"] for s in o_segs)),
    }


def consistency_score(pred_sig, train_sigs):
    """
    Average fraction of features in pred_sig that match each training signature.
    Features chosen to be discriminating across common 1D ARC task types.
    """
    FEATURES = [
        "n_delta",
        "px_delta",
        "max_preserved",
        "size_multiset_preserved",
        "colors_preserved",
        "singletons_out",
        "singletons_stay",
        "n_objects_out",
    ]
    if not train_sigs:
        return 0.0
    total = 0.0
    for ts in train_sigs:
        match = sum(1 for f in FEATURES if pred_sig.get(f) == ts.get(f))
        total += match / len(FEATURES)
    return total / len(train_sigs)


# ── File parsing ──────────────────────────────────────────────────────────────

def _parse_pairs(lines):
    """Parse FEE file lines → list of (in_row, out_row), test_in_row."""
    n = int(lines[0])
    pairs, idx = [], 1
    for _ in range(n):
        h, w = map(int, lines[idx].split())
        in_row = list(map(int, lines[idx + 1].split()))
        idx += 1 + h
        h2, _ = map(int, lines[idx].split())
        out_row = list(map(int, lines[idx + 1].split()))
        idx += 1 + h2
        pairs.append((in_row, out_row))
    # held-out / test input
    h, _ = map(int, lines[idx].split())
    test_in = list(map(int, lines[idx + 1].split()))
    return pairs, test_in


def load_training_pairs(training_dir):
    """Return all (in_row, out_row) pairs from the training file, plus held-out."""
    txts = glob.glob(os.path.join(training_dir, "*.txt"))
    if not txts:
        return [], []
    with open(txts[0]) as f:
        lines = [l.rstrip("\n") for l in f.readlines()]
    pairs, held_in = _parse_pairs(lines)
    return pairs, held_in


def load_test_input(testing_dir):
    """Return the test input row from the testing file."""
    txts = glob.glob(os.path.join(testing_dir, "*.txt"))
    if not txts:
        return []
    with open(txts[0]) as f:
        lines = [l.rstrip("\n") for l in f.readlines()]
    _, test_in = _parse_pairs(lines)
    return test_in


# ── Main selector ─────────────────────────────────────────────────────────────

def structural_select(population, training_dir, testing_dir,
                      simulator_module, wrap_fn, verbose=False):
    """
    Among all Training=1.0 programs in the population, return the one whose
    prediction on the test input is most structurally consistent with the
    transformation seen in the training pairs.

    Uses only the test INPUT (not the expected test output).
    """
    # Load training pairs (for signatures) and test input (for prediction)
    train_pairs, held_in = load_training_pairs(training_dir)
    test_in = load_test_input(testing_dir)

    all_pairs = train_pairs + [(held_in, None)]  # held-out input only
    if not test_in:
        return None

    # Compute training transformation signatures
    train_sigs = [transformation_signature(inp, out)
                  for inp, out in train_pairs if out is not None]
    if not train_sigs:
        return None

    # Unique Training=1.0 candidates
    seen, candidates = set(), []
    for ind in population:
        prog_str = str(ind[0])
        if ind[1] >= 1.0 and prog_str not in seen:
            seen.add(prog_str)
            candidates.append(ind)
    if not candidates:
        return None

    # Run each candidate on the test input
    tmp_dir = tempfile.mkdtemp()
    try:
        # Write a minimal FEE file with just the test input as both fit and test
        # (we only need it to run the program through FEE)
        # Use one dummy training pair + test input as held-out
        dummy_in  = train_pairs[0][0] if train_pairs else test_in
        dummy_out = train_pairs[0][1] if train_pairs else test_in
        w = len(test_in)

        def row_to_block(row):
            return [f"1 {len(row)}", " ".join(map(str, row))]

        file_lines = ["1"]                          # 1 fit pair
        file_lines += row_to_block(dummy_in)        # fit input
        file_lines += row_to_block(dummy_out)       # fit output
        file_lines += row_to_block(test_in)         # test input
        file_lines += row_to_block(dummy_out)       # test output (dummy)
        file_lines += row_to_block(test_in)         # test input repeated

        with open(os.path.join(tmp_dir, "struct.txt"), "w") as f:
            f.write("\n".join(file_lines) + "\n")

        s_tmp = simulator_module.Runner(tmp_dir)
        prog_strs = [wrap_fn(str(ind[0])) for ind in candidates]
        # We can't get the output grid directly from FEE, so we use a workaround:
        # run each program on the real training file and collect outputs via FEE scoring
        # Instead, let's compute predictions by running on an isolated test runner.
        scores = s_tmp.run(prog_strs)
    finally:
        shutil.rmtree(tmp_dir)

    # For each candidate, predict the output by running on the test
    # We need the actual output grid. Use a per-candidate temp dir approach:
    results = []
    for ind, prog_str_wrapped in zip(candidates, prog_strs):
        # Run program and infer predicted output from a single-pair FEE file
        # where the test input IS the evaluation input.
        # We recover the predicted row by building a test file where the
        # "expected" output is all zeros — the score will be 0 if anything
        # non-zero is produced, but we can't get the actual grid back from FEE.
        # Instead, use a Python-level simulation from the learned_dsl primitives.
        # Simpler: skip grid recovery and score consistency using training-only info.
        pass

    # ── Fallback: use training-pair consistency only (no test grid needed) ──
    # For each candidate, compute how consistently it transforms training inputs.
    # The key insight: a program that is structurally consistent on training pairs
    # is more likely to be consistent on the test input.
    # This doesn't require running on the test input at all.

    # Compute signatures for each candidate on each training pair
    candidate_scores = []
    for ind in candidates:
        prog_str = wrap_fn(str(ind[0]))
        pair_sigs = []
        for inp, out in train_pairs:
            tmp_dir2 = tempfile.mkdtemp()
            try:
                file_lines = ["1"]
                file_lines += row_to_block(inp)
                file_lines += row_to_block(out)
                file_lines += row_to_block(inp)
                file_lines += row_to_block(out)
                file_lines += row_to_block(inp)
                with open(os.path.join(tmp_dir2, "t.txt"), "w") as f:
                    f.write("\n".join(file_lines) + "\n")
                # We can't easily get predicted output from FEE
                # so we'll use a simpler approach
            finally:
                shutil.rmtree(tmp_dir2)

    # ── Better approach: use Python DSL to predict output ────────────────────
    # Import the Python implementations from learned_dsl.json to get predictions
    return _structural_select_python(candidates, train_pairs, test_in,
                                     train_sigs, wrap_fn, verbose)


def _structural_select_python(candidates, train_pairs, test_in,
                               train_sigs, wrap_fn, verbose):
    """
    Use Python-level program execution to predict test output, then score
    structural consistency. Falls back to first candidate if prediction fails.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from primitives import (
        filter_largest, filter_smallest, filter_unique_color,
        filter_horizontal_lines, filter_vertical_lines, identity,
        move_right, move_left, move_up, move_down,
        flip_horizontal, flip_vertical, rotate_cw_90,
        recolor_to_color, filter_by_color,
    )
    from runner_dsl import set_pset_dsl

    # Build a simple Python executor for single-primitive programs
    PYTHON_OPS = {
        "dsl_clear_noise":          lambda objs: [o for o in objs
                                                   if o.get("type") == "metadata"
                                                   or o.get("color") == 0
                                                   or len(o.get("pixels", [])) > 1],
        "dsl_filter_even_size":     lambda objs: [o for o in objs
                                                   if o.get("type") == "metadata"
                                                   or o.get("color") == 0
                                                   or len(o.get("pixels", [])) % 2 == 0],
        "dsl_filter_odd_size":      lambda objs: [o for o in objs
                                                   if o.get("type") == "metadata"
                                                   or o.get("color") == 0
                                                   or len(o.get("pixels", [])) % 2 == 1],
        "dsl_filter_largest":       lambda objs: filter_largest(objs),
        "dsl_filter_smallest":      lambda objs: filter_smallest(objs),
        "dsl_filter_unique_color":  lambda objs: filter_unique_color(objs),
        "dsl_filter_horizontal_lines": lambda objs: filter_horizontal_lines(objs),
        "dsl_flip_horizontal":      lambda objs: flip_horizontal(objs),
        "dsl_identity":             lambda objs: identity(objs),
        "dsl_keep_bg_or_large":     lambda objs: [o for o in objs
                                                   if o.get("type") == "metadata"
                                                   or o.get("color") == 0
                                                   or len(o.get("pixels", [])) > 1],
        "dsl_recolor_to_dominant":  lambda objs: _recolor_to_dominant(objs),
        "dsl_reflect_about_marker": lambda objs: _reflect_about_marker(objs),
    }

    best, best_score = candidates[0], -1.0

    for ind in candidates:
        prog_str = str(ind[0])
        # Extract the inner operation name (works for single-op programs)
        import re
        m = re.search(r'prog2\(dsl_segment\(\), prog2\(prog2\((dsl_\w+)\(\)', prog_str)
        if not m:
            continue
        op_name = m.group(1)
        op = PYTHON_OPS.get(op_name)
        if op is None:
            continue

        # Predict output on test input
        test_objs = _row_to_objects(test_in)
        try:
            pred_objs = op(test_objs)
        except Exception:
            continue
        pred_row = _objects_to_row(pred_objs, len(test_in))
        pred_sig = transformation_signature(test_in, pred_row)

        score = consistency_score(pred_sig, train_sigs)

        if verbose:
            print(f"  {op_name:<35} pred={pred_row}")
            _print_sig(pred_sig)
            print(f"  → consistency score: {score:.3f}\n")

        if score > best_score:
            best_score = score
            best = ind

    return best


# ── Python-level object helpers ───────────────────────────────────────────────

def _row_to_objects(row):
    """Convert a 1D pixel row to a list of object dicts (learnable-dsl format)."""
    objs = [{"type": "metadata", "input_grid_size": (1, len(row)),
              "color": 0, "pixels": []}]
    i = 0
    while i < len(row):
        if row[i] != 0:
            j = i
            while j < len(row) and row[j] == row[i]:
                j += 1
            objs.append({"type": "object", "color": row[i],
                          "pixels": [(0, c) for c in range(i, j)],
                          "area": j - i,
                          "center": (0, (i + j - 1) / 2),
                          "input_grid_size": (1, len(row))})
            i = j
        else:
            # background segment
            j = i
            while j < len(row) and row[j] == 0:
                j += 1
            objs.append({"type": "object", "color": 0,
                          "pixels": [(0, c) for c in range(i, j)],
                          "area": j - i,
                          "input_grid_size": (1, len(row))})
            i = j
    return objs


def _objects_to_row(objs, width):
    """Render objects to a 1D row. Non-zero objects painted last so they
    overwrite background, matching FEE's render_to_grid behaviour."""
    row = [0] * width
    # background first, non-zero last (same order as FEE C++)
    for obj in objs:
        if obj.get("type") == "metadata" or obj.get("color") != 0:
            continue
        for r, c in obj.get("pixels", []):
            if 0 <= c < width:
                row[c] = 0
    for obj in objs:
        if obj.get("type") == "metadata" or obj.get("color") == 0:
            continue
        for r, c in obj.get("pixels", []):
            if 0 <= c < width:
                row[c] = obj["color"]
    return row


def _recolor_to_dominant(objs):
    non_meta = [o for o in objs if o.get("type") != "metadata" and o.get("color") != 0]
    if not non_meta:
        return objs
    dominant = max(non_meta, key=lambda o: o.get("area", len(o.get("pixels", []))))["color"]
    return [{**o, "color": dominant} if o.get("type") != "metadata" and o.get("color") != 0
            else o for o in objs]


def _reflect_about_marker(objs):
    marker_col = None
    for obj in objs:
        if obj.get("type") == "metadata" or obj.get("color") == 0:
            continue
        if len(obj.get("pixels", [])) == 1:
            marker_col = obj["pixels"][0][1]
            break
    if marker_col is None:
        return objs
    result = []
    for obj in objs:
        if obj.get("type") == "metadata" or obj.get("color") == 0 or \
                len(obj.get("pixels", [])) == 1:
            result.append(obj)
        else:
            new_pixels = [(r, 2 * marker_col - c) for r, c in obj["pixels"]]
            result.append({**obj, "pixels": new_pixels})
    return result


def _print_sig(sig):
    print(f"    n_delta={sig['n_delta']}  px_delta={sig['px_delta']}  "
          f"max_preserved={sig['max_preserved']}  "
          f"colors_preserved={sig['colors_preserved']}  "
          f"singletons_out={sig['singletons_out']}")
