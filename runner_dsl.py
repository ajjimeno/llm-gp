"""
runner_dsl.py
-------------
Defines the DEAP primitive set for ARC DSL programs evaluated by
fast-evolutionary-evaluation's SimulatorCPU.

Program structure:
    prog2(dsl_segment(), prog2(<dsl_ops...>, dsl_render()))

Execution by FEE:
    dsl_segment() - segments current grid into objects
    dsl_* ops     - filter / transform the object list
    dsl_render()  - renders objects back to output grid

All DSL ops use the dsl_ prefix as required by FEE's instruction map.
Color-parameterized ops (dsl_filter_color, dsl_recolor) take an RInteger
argument (supplied by get0..get9 terminals).

Learned primitives:
    Primitives learned by learnable-dsl can be loaded from a JSON file via
    load_learned_primitives().  Pass mdl_threshold to filter by MDL gain
    (None = load all, 0.0 = positive-gain only, etc.).  These are added to
    the DEAP grammar as no-arg Operations; C++ implementations in FEE are
    required for actual program execution.
"""

import json
import os
from deap import base, creator, gp

import gp_algorithm

# Default path to the learned DSL JSON produced by learnable-dsl.
_LEARNED_DSL_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "learnable-dsl", "learned_dsl.json"
)


class Operation(object):
    """A void operation on shared grid / DSL state."""
    pass


class RInteger(object):
    """A register integer (color constant 0-9)."""
    pass


class Boolean(object):
    """A boolean value (0 or 1) produced by comparison primitives."""
    pass


def load_learned_primitives(path=None, mdl_threshold=None):
    """Load learned primitives from a learned_dsl.json file.

    Parameters
    ----------
    path : str or None
        Path to learned_dsl.json.  Defaults to ../learnable-dsl/learned_dsl.json
        relative to this file.
    mdl_threshold : float or None
        If given, only primitives with mdl_gain >= mdl_threshold are returned.
        None means return all entries (including those without an mdl_gain field).

    Returns
    -------
    list of dict  — each dict has at least "name", "code", and optionally "mdl_gain".
    """
    path = path or _LEARNED_DSL_DEFAULT_PATH
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return []

    if mdl_threshold is not None:
        data = [
            d for d in data
            if d.get("mdl_gain") is not None and d["mdl_gain"] >= mdl_threshold
        ]
    return data


def set_pset_dsl(learned_path=None, mdl_threshold=None):
    """Return a DEAP PrimitiveSetTyped with all FEE DSL primitives."""
    pset = gp.PrimitiveSetTyped("MAIN", [], Operation)

    # ------------------------------------------------------------------ #
    # No-argument transforms                                              #
    # ------------------------------------------------------------------ #
    for name in [
        "dsl_move_up",
        "dsl_move_down",
        "dsl_move_left",
        "dsl_move_right",
        "dsl_rotate_cw_90",
        "dsl_flip_vertical",
        "dsl_flip_horizontal",
        "dsl_filter_largest",
        "dsl_filter_smallest",
        "dsl_filter_unique_color",
        "dsl_filter_rectangles",
        "dsl_filter_squares",
        "dsl_filter_horizontal_lines",
        "dsl_filter_vertical_lines",
        "dsl_identity",
        "dsl_reset",
    ]:
        pset.addPrimitive(lambda: None, [], Operation, name=name)

    # ------------------------------------------------------------------ #
    # Color-parameterised transforms  (take one RInteger color arg)       #
    # ------------------------------------------------------------------ #
    pset.addPrimitive(lambda c: None, [RInteger], Operation, name="dsl_filter_color")
    pset.addPrimitive(lambda c: None, [RInteger], Operation, name="dsl_recolor")

    # ------------------------------------------------------------------ #
    # Color constants (terminals for RInteger)                            #
    # ------------------------------------------------------------------ #
    for i in range(10):
        pset.addPrimitive(lambda: None, [], RInteger, name=f"get{i}")

    # Returns the number of non-metadata objects as an integer (RInteger).
    # Useful as a color argument: e.g. dsl_recolor(dsl_n_objects()).
    pset.addPrimitive(lambda: None, [], RInteger, name="dsl_n_objects")

    # Returns the gap between the non-singleton objects and the singleton (marker) object,
    # capped to [0,9]. Useful for mirror and move-to-marker tasks.
    pset.addPrimitive(lambda: None, [], RInteger, name="dsl_get_marker_distance")

    # ------------------------------------------------------------------ #
    # Denoising (from extracted_dsl.json: clear_small_nonblack_color)   #
    # ------------------------------------------------------------------ #
    # dsl_clear_noise: set single-pixel (area==1) non-background objects to 0
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_clear_noise")

    # ------------------------------------------------------------------ #
    # Color exclusion filter (from sleep_audit: remove_non_metadata_by_color family)
    # ------------------------------------------------------------------ #
    # dsl_filter_not_color: keep all objects EXCEPT those with the given color
    pset.addPrimitive(lambda c: None, [RInteger], Operation, name="dsl_filter_not_color")

    # dsl_recolor_color: recolor objects of color src to dst; all others unchanged
    pset.addPrimitive(lambda s, d: None, [RInteger, RInteger], Operation, name="dsl_recolor_color")

    # ------------------------------------------------------------------ #
    # Segmentation variant                                                 #
    # ------------------------------------------------------------------ #
    # dsl_regroup_by_color: merge all disconnected same-color objects into one per color
    # (ColorPartition-style perception applied after initial segmentation)
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_regroup_by_color")

    # ------------------------------------------------------------------ #
    # Compositionality: complement-merge with initial snapshot             #
    # ------------------------------------------------------------------ #
    # dsl_union_input: append non-zero initial objects whose color is NOT in current.
    # Pattern: filter_color(X) → transform → dsl_union_input()
    # = transform color-X objects and restore all other initial non-zero objects.
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_union_input")

    # dsl_union_all_input: restore ALL non-zero initial objects unconditionally.
    # Use for fill-with-same-color: prog3(dsl_filter_color(get0()), prog2(dsl_filter_smallest(), dsl_recolor(getX())), dsl_union_all_input())
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_union_all_input")

    # ------------------------------------------------------------------ #
    # Dominant-color recolor (from sleep_audit: 1d_denoising_mc programs) #
    # ------------------------------------------------------------------ #
    # dsl_recolor_to_dominant: recolor all non-zero objects to the color of
    # the non-zero object with the largest total pixel area
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_recolor_to_dominant")

    # ------------------------------------------------------------------ #
    # Parameterised moves (move N steps in one op)                       #
    # ------------------------------------------------------------------ #
    for name in ["dsl_move_right_n", "dsl_move_left_n", "dsl_move_up_n", "dsl_move_down_n"]:
        pset.addPrimitive(lambda c: None, [RInteger], Operation, name=name)

    # ------------------------------------------------------------------ #
    # Conditional branching                                               #
    # ------------------------------------------------------------------ #
    # dsl_comparison(Boolean, Operation, Operation) → Operation
    # Executes op1 if the boolean is true (non-zero), op2 otherwise.
    pset.addPrimitive(
        lambda b, op1, op2: None,
        [Boolean, Operation, Operation],
        Operation,
        name="dsl_comparison",
    )
    # dsl_bigger_thanR(a, b) → Boolean: true if a > b
    pset.addPrimitive(
        lambda a, b: None, [RInteger, RInteger], Boolean, name="dsl_bigger_thanR"
    )
    # dsl_equalR(a, b) → Boolean: true if a == b
    pset.addPrimitive(
        lambda a, b: None, [RInteger, RInteger], Boolean, name="dsl_equalR"
    )

    # ------------------------------------------------------------------ #
    # Data-driven color terminal                                          #
    # ------------------------------------------------------------------ #
    # dsl_get_max_color() → RInteger: most frequent non-zero color in input
    pset.addPrimitive(lambda: None, [], RInteger, name="dsl_get_max_color")

    # ------------------------------------------------------------------ #
    # Hollow and bbox-flip                                                #
    # ------------------------------------------------------------------ #
    # dsl_hollow: keep only border pixels of each non-background object
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_hollow")

    # dsl_flip_horizontal_bbox: flip within bounding box (not grid centre)
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_flip_horizontal_bbox")

    # ------------------------------------------------------------------ #
    # Fill operations                                                     #
    # ------------------------------------------------------------------ #
    # dsl_fill_bbox: fill the bounding box of each non-zero color group.
    # Multiple same-color objects are merged into one filled rectangle.
    # Solves 1d_fill: two same-color dots → filled segment between them.
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_fill_bbox")

    # dsl_padded_fill: fill the space between consecutive pairs of objects
    # per row with the first object's color.
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_padded_fill")

    # ------------------------------------------------------------------ #
    # Gravity (move until touching another object or grid edge)           #
    # ------------------------------------------------------------------ #
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_gravity_right")
    pset.addPrimitive(lambda: None, [], Operation, name="dsl_gravity_left")

    # ------------------------------------------------------------------ #
    # Looping                                                             #
    # ------------------------------------------------------------------ #
    # dsl_loop(n, Operation): repeat Operation exactly n times (n = RInteger 0-9)
    #pset.addPrimitive(lambda n, op: None, [RInteger, Operation], Operation, name="dsl_loop")

    # ------------------------------------------------------------------ #
    # Sequencing                                                          #
    # ------------------------------------------------------------------ #
    pset.addPrimitive(
        lambda a, b: None, [Operation, Operation], Operation, name="prog2"
    )
    pset.addPrimitive(
        lambda a, b, c: None,
        [Operation, Operation, Operation],
        Operation,
        name="prog3",
    )

    # ------------------------------------------------------------------ #
    # Halt (FEE executes the program in a 200-iteration loop;             #
    # status_end() breaks out after the first pass)                       #
    # ------------------------------------------------------------------ #
    pset.addPrimitive(lambda: None, [], Operation, name="status_end")

    # ------------------------------------------------------------------ #
    # Learned primitives (from learnable-dsl)                             #
    # ------------------------------------------------------------------ #
    # repeat(objects, func, n) → same as dsl_loop; FEE opcode 235.
    pset.addPrimitive(lambda n, op: None, [RInteger, Operation], Operation, name="dsl_repeat")

    # filter_size_n(objects, n) → FEE opcode 263; must be pre-registered as [RInteger] → Operation.
    pset.addPrimitive(lambda c: None, [RInteger], Operation, name="dsl_filter_size_n")

    # filter_by_color(objects, color) → same C++ as dsl_filter_color; FEE opcode 202.
    pset.addPrimitive(lambda c: None, [RInteger], Operation, name="dsl_filter_by_color")

    # recolor_to_color(objects, color) → same C++ as dsl_recolor; FEE opcode 220.
    pset.addPrimitive(lambda c: None, [RInteger], Operation, name="dsl_recolor_to_color")

    # All three above are pre-registered; the loader loop skips them automatically.
    _SKIP_LEARNED = set()  # nothing left to skip explicitly

    for entry in load_learned_primitives(path=learned_path, mdl_threshold=mdl_threshold):
        name = entry.get("name", "").strip()
        if not name or name in _SKIP_LEARNED:
            continue
        dsl_name = f"dsl_{name}" if not name.startswith("dsl_") else name
        # Skip if already registered (base primitives or dsl_repeat pre-registered above)
        if dsl_name in pset.mapping:
            continue
        # Register AU/LLM-generalized primitives with their persisted free-
        # parameter arity ("n_params" in learned_dsl.json — see learnable-dsl
        # main.py's sleep phase) so the grammar can actually supply RInteger
        # arguments (get0..get9) instead of uniformly registering every
        # learned primitive as a no-arg Operation, which made parameterized
        # candidates structurally uninvocable by the GP search.
        n_params = entry.get("n_params") or 0
        if n_params == 1:
            pset.addPrimitive(lambda c: None, [RInteger], Operation, name=dsl_name)
        elif n_params == 2:
            pset.addPrimitive(lambda a, b: None, [RInteger, RInteger], Operation, name=dsl_name)
        elif n_params >= 3:
            pset.addPrimitive(lambda a, b, c: None, [RInteger, RInteger, RInteger], Operation, name=dsl_name)
        else:
            pset.addPrimitive(lambda: None, [], Operation, name=dsl_name)

    return pset
