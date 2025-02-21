import operator

import gp_algorithm
from deap import base, creator, gp
from runner import Runner, set_pset

runner = Runner()
pset = set_pset(runner, is_arc=False)

creator.create("FitnessMin", base.Fitness, weights=(1.0, 0.000000000000005))
creator.create("Individual", gp_algorithm.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)

toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

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


def get_valid_program(program):
    try:
        return creator.Individual.from_string(program, pset)
    except Exception as e:
        print(e, program)
        return None


def check_programs(programs):
    return [
        str(creator.Individual.from_string(p, pset))
        for p in programs
        if get_valid_program(p)
    ]


def get_program_length(program):
    return len(creator.Individual.from_string(program, pset))
