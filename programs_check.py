import logging
import operator

from deap import base, creator, gp, tools

import gp_algorithm
from runner import Runner, set_pset

gp.genGrow = gp_algorithm.genGrow
gp.generate = gp_algorithm.generate

logger = logging.getLogger(__name__)

runner = Runner()
pset = set_pset(runner, is_arc=False)

creator.create("FitnessMin", base.Fitness, weights=(1.0, 0.000000000000005))
creator.create("Individual", gp_algorithm.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)

toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

"""
Koza, J. R. (1992). Genetic Programming: On the Pro-
gramming of Computers by Means of Natural Selection.
MIT Press, Cambridge, MA, USA.
"""
toolbox.decorate(
    "mate",
    gp.staticLimit(key=operator.attrgetter("height"), max_value=120),
)  # 120, 17, default, 30 most of tests
toolbox.decorate(
    "mutate",
    gp.staticLimit(key=operator.attrgetter("height"), max_value=120),
)  # 17, default, 30 most of tests


def get_primitive_tree(program) -> gp_algorithm.PrimitiveTree:
    return creator.Individual.from_string(program, pset)


def get_valid_program(program) -> bool:
    try:
        return get_primitive_tree(program)
    except Exception as e:
        print(e, program)
        return None


def check_programs(programs) -> list[str]:
    return [
        str(creator.Individual.from_string(p, pset))
        for p in programs
        if get_valid_program(p)
    ]
