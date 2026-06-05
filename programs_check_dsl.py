"""
programs_check_dsl.py
---------------------
DEAP toolbox for ARC DSL programs (parallel to programs_check.py for list tasks).

Usage:
    from programs_check_dsl import get_primitive_tree, get_valid_program, toolbox
"""

import logging
import operator

from deap import base, creator, gp, tools

import gp_algorithm
from runner_dsl import set_pset_dsl

gp.genGrow = gp_algorithm.genGrow
gp.generate = gp_algorithm.generate

logger = logging.getLogger(__name__)

pset = set_pset_dsl()

creator.create("FitnessMaxDSL", base.Fitness, weights=(1.0, 0.000000000000005))
creator.create(
    "IndividualDSL", gp_algorithm.PrimitiveTree, fitness=creator.FitnessMaxDSL
)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register(
    "individual", tools.initIterate, creator.IndividualDSL, toolbox.expr
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=120)
)
toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=120)
)


def wrap_with_segment_render(middle: str) -> str:
    """Wrap an evolved sub-program with dsl_segment(), dsl_render(), and status_end().

    Produces:
        prog2(dsl_segment(), prog2(prog2(<middle>, dsl_render()), status_end()))

    status_end() is required because FEE executes the program root 200 times
    in a loop — without it, the DSL ops would be applied 200 times, shifting
    objects off the grid.  status_end() halts the loop after one pass.
    """
    return (
        f"prog2(dsl_segment(), prog2(prog2({middle}, dsl_render()), status_end()))"
    )


def get_primitive_tree(program: str) -> gp_algorithm.PrimitiveTree:
    return creator.IndividualDSL.from_string(program, pset)


def get_valid_program(program: str):
    try:
        return get_primitive_tree(program)
    except Exception as e:
        logger.debug("Invalid program %s: %s", program, e)
        return None


def check_programs(programs):
    """Filter a list of inner program strings to only those parseable by the pset."""
    return [p for p in programs if get_valid_program(p) is not None]
