"""
prompts_dsl.py
--------------
LLM prompting for ARC DSL programs.  Parallel to prompts.py (list tasks).
"""

import json
import os
import random
import re

from dotenv import load_dotenv
from tqdm import tqdm

from llm import get_model
from logger_config import getLogger

load_dotenv()
logger = getLogger(__name__)


def _check():
    from programs_check_dsl import check_programs, get_primitive_tree, get_valid_program, wrap_with_segment_render
    return check_programs, get_primitive_tree, get_valid_program, wrap_with_segment_render


def clean_output(string):
    return (
        string.replace("```python\n", "")
        .replace("```\n", "")
        .replace("\n", "")
        .replace("```", "")
        .replace(" ", "")
        .replace("```json\n", "")
        .replace("plaintext", "")
    )


def extract_python_code(text, prefix="python"):
    match = re.search(rf"```{prefix}(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


system_prompt = (
    "You are an expert assistant specialised in program synthesis and "
    "evolutionary algorithms for visual reasoning tasks."
)

dsl_functions = """
These are the available DSL primitives for transforming ARC grids.
You write only the INNER transform pipeline — the system automatically wraps it as:
  prog2(dsl_segment(), prog2(prog2(<your_program>, dsl_render()), status_end()))

Do NOT include dsl_segment() or dsl_render() in your output.

--- Sequencing and looping ---
prog2(Operation, Operation): executes the first operation then the second.
prog3(Operation, Operation, Operation): executes three operations in sequence.
dsl_loop(RInteger, Operation): repeats the given Operation exactly N times (N supplied as a colour constant get0()..get9() or dsl_n_objects()).

--- No-argument transforms (each modifies the current object list) ---
dsl_move_up(): shifts all object pixels one row up.
dsl_move_down(): shifts all object pixels one row down.
dsl_move_left(): shifts all object pixels one column left.
dsl_move_right(): shifts all object pixels one column right.
dsl_rotate_cw_90(): rotates all object pixels 90 degrees clockwise.
dsl_flip_vertical(): reflects all object pixels vertically.
dsl_flip_horizontal(): reflects all object pixels horizontally.
dsl_identity(): no-op, passes objects through unchanged.
dsl_reset(): restores the object list to the initial segmented state.
dsl_clear_noise(): removes single-pixel (area==1) non-background objects entirely (denoising).
dsl_recolor_to_dominant(): recolours all non-background objects to the colour of the non-background object with the largest total pixel area (multi-colour denoising).
dsl_hollow(): removes interior pixels from each non-background object, keeping only border pixels (pixels with at least one absent 4-connected neighbour). In 1-D this reduces each segment to its two endpoints.
dsl_flip_horizontal_bbox(): flips all non-background objects horizontally within their combined bounding box. Unlike dsl_flip_horizontal() (which mirrors around the grid centre), this preserves the objects' horizontal extent and only reverses their left-right order.
dsl_gravity_right(): moves each non-background object as far right as possible without overlapping other non-background objects or going out of bounds. Objects are stacked: the rightmost hits the wall first, others stack against it.
dsl_gravity_left(): symmetric — moves each non-background object as far left as possible.
dsl_regroup_by_color(): merges all disconnected same-colour objects into a single object per colour; after this, filter_largest compares colour groups rather than connected components.
dsl_union_input(): appends non-zero initial objects whose colour does NOT appear in the current object list. Use after filtering and transforming a subset to restore the untouched objects. Pattern: prog3(dsl_filter_color(X), transform, dsl_union_input()).
dsl_union_all_input(): restores ALL non-zero initial objects unconditionally (ignores colour overlap). Use for fill-with-same-colour: after filling background holes with colour X, the original shape (also colour X) is blocked by dsl_union_input's colour gate — use this instead.

--- Filters (keep only objects that match a criterion) ---
dsl_filter_largest(): keeps only the largest object(s) by pixel count.
dsl_filter_smallest(): keeps only the smallest object(s) by pixel count.
dsl_filter_unique_color(): keeps only objects whose colour appears exactly once.
dsl_filter_rectangles(): keeps only objects that are solid rectangles.
dsl_filter_squares(): keeps only objects that are solid squares.
dsl_filter_horizontal_lines(): keeps only objects where all pixels share the same row.
dsl_filter_vertical_lines(): keeps only objects where all pixels share the same column.
dsl_filter_color(ColorInt): keeps only objects with the given colour.
dsl_filter_not_color(ColorInt): keeps all objects EXCEPT those with the given colour (inverse of dsl_filter_color).
dsl_recolor(ColorInt): recolours all remaining objects to the given colour.
dsl_recolor_color(SrcColorInt, DstColorInt): recolours only objects whose colour == SrcColorInt to DstColorInt; all other objects pass through unchanged.

--- Parameterised moves (move N steps in one operation) ---
dsl_move_right_n(ColorInt): shifts all objects N columns right (N = colour constant 0-9).
dsl_move_left_n(ColorInt): shifts all objects N columns left.
dsl_move_up_n(ColorInt): shifts all objects N rows up.
dsl_move_down_n(ColorInt): shifts all objects N rows down.

--- Colour constants and data-driven terminals (RInteger) ---
get0() through get9(): return the integer 0 through 9 respectively.
dsl_get_max_color(): returns the most frequent non-zero colour in the input grid as an integer. Use wherever a colour constant is expected.
dsl_n_objects(): returns the number of non-metadata objects in the current list as an integer.

--- Conditional branching ---
dsl_comparison(Boolean, Operation, Operation): if the Boolean is true, executes the first Operation; otherwise executes the second. Enables programs that behave differently depending on properties of the current state.
dsl_bigger_thanR(RInteger, RInteger) → Boolean: true if the first integer is greater than the second.
dsl_equalR(RInteger, RInteger) → Boolean: true if the two integers are equal.

--- Valid example inner programs ---
<example>dsl_filter_largest()</example>
<example>prog2(dsl_filter_unique_color(), dsl_move_right())</example>
<example>dsl_recolor(get1())</example>
<example>prog2(dsl_filter_color(get2()), dsl_recolor(get5()))</example>
<example>prog2(dsl_filter_largest(), prog2(dsl_move_down(), dsl_move_right()))</example>
<example>dsl_clear_noise()</example>
<example>dsl_recolor_to_dominant()</example>
<example>prog2(dsl_filter_not_color(get0()), dsl_recolor_to_dominant())</example>
<example>dsl_filter_not_color(get2())</example>
<example>dsl_recolor_color(get3(), get1())</example>
<example>prog2(dsl_filter_largest(), dsl_recolor_color(get2(), get5()))</example>
<example>dsl_move_right_n(get3())</example>
<example>prog2(dsl_filter_largest(), dsl_move_right_n(get2()))</example>
<example>dsl_regroup_by_color()</example>
<example>prog2(dsl_regroup_by_color(), dsl_filter_largest())</example>
<example>prog3(dsl_filter_color(get3()), dsl_move_right_n(get2()), dsl_union_input())</example>
<example>prog3(dsl_filter_color(get1()), dsl_recolor(get5()), dsl_union_input())</example>
<example>dsl_comparison(dsl_bigger_thanR(dsl_n_objects(), get1()), dsl_recolor_to_dominant(), dsl_clear_noise())</example>
<example>dsl_comparison(dsl_equalR(dsl_n_objects(), get1()), dsl_recolor(dsl_get_max_color()), dsl_filter_largest())</example>
<example>dsl_filter_color(dsl_get_max_color())</example>
<example>prog2(dsl_filter_color(dsl_get_max_color()), dsl_move_right_n(get3()))</example>
<example>dsl_comparison(dsl_bigger_thanR(dsl_n_objects(), get3()), dsl_filter_largest(), dsl_identity())</example>
<example>dsl_loop(get3(), dsl_move_right())</example>
<example>prog2(dsl_filter_largest(), dsl_loop(get2(), dsl_move_down()))</example>
<example>dsl_loop(dsl_n_objects(), dsl_move_right())</example>
<example>prog3(dsl_filter_color(get0()), prog2(dsl_filter_smallest(), dsl_recolor(get3())), dsl_union_input())</example>
<example>prog3(dsl_filter_color(get0()), prog2(dsl_filter_smallest(), dsl_recolor(get3())), dsl_union_all_input())</example>
<example>dsl_hollow()</example>
<example>prog2(dsl_filter_largest(), dsl_hollow())</example>
<example>dsl_flip_horizontal_bbox()</example>
<example>dsl_gravity_right()</example>
<example>dsl_gravity_left()</example>
<example>prog2(dsl_filter_color(get3()), dsl_gravity_right())</example>

--- Rules ---
* Do NOT write dsl_segment() or dsl_render() — they are added automatically.
* prog2 always takes exactly two Operation arguments.
* dsl_filter_color, dsl_filter_not_color, dsl_recolor, and parameterised moves require exactly one colour constant argument (get0()..get9()).
* dsl_recolor_color requires exactly two colour constant arguments: dsl_recolor_color(get2(), get5()) recolours all colour-2 objects to colour 5.
* dsl_regroup_by_color and dsl_union_input take no arguments.
* dsl_union_input restores initial non-zero objects NOT of the current colours — use it as the last step after filtering and transforming a subset.
* dsl_union_all_input restores ALL initial non-zero objects regardless of colour overlap — use it when the fill colour equals the shape colour.
* Background (colour 0) is now part of the object list. dsl_filter_color(get0()) selects background regions; dsl_filter_smallest() then selects interior holes (smallest region). dsl_filter_largest() and dsl_filter_smallest() automatically skip background when non-zero objects are present.
* dsl_loop takes exactly two arguments: an RInteger (count) and an Operation (body). Count is capped internally — loops with count 0 are no-ops.
* dsl_comparison takes exactly three arguments: a Boolean, then two Operations (then-branch, else-branch).
* dsl_bigger_thanR and dsl_equalR take exactly two RInteger arguments and return a Boolean — they cannot be used as Operations directly.
* dsl_get_max_color() and dsl_n_objects() return RInteger values — use them as colour arguments, not as Operations.
* No other primitives exist. Do not invent new ones.
"""


def _format_grid(grid):
    return "\n".join("  " + " ".join(str(v) for v in row) for row in grid)


def get_arc_training_examples(task_data: dict) -> str:
    """Format an ARC task dict for inclusion in an LLM prompt."""
    lines = []
    for i, pair in enumerate(task_data["train"]):
        lines.append(f"Training example {i + 1}:")
        lines.append("  Input:")
        lines.append(_format_grid(pair["input"]))
        lines.append("  Output:")
        lines.append(_format_grid(pair["output"]))
    test = task_data["test"][0]
    lines.append("Test input (apply the discovered rule to produce the output):")
    lines.append(_format_grid(test["input"]))
    return "\n".join(lines)


class DSLGeneticPrompting:
    """LLM prompting for ARC DSL GP.  Drop-in replacement for GeneticPrompting."""

    def __init__(self, task_data: dict):
        """
        Args:
            task_data: ARC task dict with 'train' and 'test' keys.
        """
        self.model = get_model(model_name=os.getenv("LLM_NAME"))
        self.task_data = task_data
        self._examples_str = get_arc_training_examples(task_data)

    def get_best_description(self, retries=5):
        descriptions = "\n".join(
            f"<description id='{i}'>{self._get_problem_description()}</description>"
            for i in range(retries)
        )
        user_prompt = f"""
Given the following descriptions of an ARC visual-reasoning task, select the
description that best and most concisely explains the transformation rule.
Output only the chosen/combined description — no additional text.

***descriptions***
{descriptions}

***examples***
{self._examples_str}
"""
        description = self.model(system_prompt=system_prompt, user_prompt=user_prompt)
        logger.info(description)
        return description

    def _get_problem_description(self):
        user_prompt = f"""
Analyse the following ARC training examples (input → output grids) and describe
the underlying transformation rule in plain language.
Be concise. Identify what objects are selected, how they are moved or recoloured,
and what disappears.

{self._examples_str}

Your analysis:
"""
        desc = self.model(system_prompt=system_prompt, user_prompt=user_prompt)
        logger.info(desc)
        return desc

    def get_problem_programs(self, description):
        check_programs, get_primitive_tree, get_valid_program, wrap_with_segment_render = _check()

        count = 0
        population = []
        max_count = 80
        pbar = tqdm(total=max_count)

        while count < max_count:
            pbar.update(1)
            pbar.set_description(f"generating program {count}/{max_count}")
            try:
                output = self.model(
                    system_prompt=system_prompt,
                    user_prompt=f"""
Using the explanation below, write 20 candidate INNER DSL transform pipelines.
Do NOT include dsl_segment() or dsl_render() — they are added automatically.
Express programs as a tree of function calls using parentheses and commas.
Do not add any primitive not listed in the specification.

<explanation>
{description}
</explanation>

Output as JSON — a list of dicts with an "example" key, one inner program per entry.
No explanation, no enumeration, no extra keys.

[
  {{"example": "dsl_filter_largest()"}}
]

{dsl_functions}
""",
                )

                logger.info(f"LLM output: {output}")

                output_list = extract_python_code(output, prefix="json") or \
                              extract_python_code(output, prefix="") or \
                              output

                population += [
                    e[list(e.keys())[0]]
                    for e in json.loads(output_list)
                ]

                population = check_programs(population)
                logger.info(f"Filtered programs: {population}")

                if len(population) > 30:
                    break
            except Exception as e:
                logger.error(f"Error: {e}")

            count += 1

        pbar.close()
        return [get_primitive_tree(p) for p in set(population)]

    def _get_guided_mutation_prompt(self, description, individual, score):
        if score >= 1.0:
            objective = (
                "The program already achieves perfect training accuracy (1.0). "
                "Your goal is NOT to improve the score (it cannot go higher) but to verify "
                "semantic correctness: does this program solve the task for the RIGHT reason, "
                "or is it coincidentally correct on the training examples? "
                "If it is coincidentally correct — i.e. its logic does not match the task "
                "description — replace it with a semantically correct program that will "
                "generalise to unseen examples."
            )
        else:
            objective = (
                f"The program achieves training accuracy {score:.4f} / 1.0. "
                "Your goal is to improve this score."
            )

        return f"""
Mutate the following DSL inner program for the ARC task.
The system wraps it automatically as: prog2(dsl_segment(), prog2(<your_program>, dsl_render()))
Do NOT include dsl_segment() or dsl_render() in your output.

Current inner program:
{individual}

Current accuracy: {score:.4f} / 1.0

{objective}

Task description:
<explanation>
{description}
</explanation>

Rules:
1. Identify a subtree that seems wrong or redundant given the task description.
2. Replace it with a better subtree using only the available primitives.
3. Output only the mutated inner program — no explanation.
4. Use {random.randint(0, 9999999)} as your random seed.

{dsl_functions}
"""

    def get_guided_mutation_program(self, description, individual):
        check_programs, get_primitive_tree, get_valid_program, wrap_with_segment_render = _check()

        individual = (str(individual[0]), individual[1])
        for _ in range(3):
            try:
                program = clean_output(
                    self.model(
                        system_prompt=system_prompt,
                        user_prompt=self._get_guided_mutation_prompt(
                            description, individual[0], individual[1]
                        ),
                    )
                )
                byte_program = get_valid_program(program)
                if not byte_program:
                    logger.info(f"Invalid mutation: {program}")
                    raise ValueError(program)
                logger.info(f"Mutated program: {program}")
                return str(byte_program)
            except Exception:
                pass
        return None
