import gp_algorithm
from deap import base, creator
from runner import Runner, set_pset

creator.create("FitnessMin", base.Fitness, weights=(1.0, 0.000000000000005))
creator.create("Individual", gp_algorithm.PrimitiveTree, fitness=creator.FitnessMin)


def is_valid_program(pset, program):
    try:
        creator.Individual.from_string(program, pset)
        return True
    except Exception as e:
        print(e)
        return False


def check_programs(programs):
    runner = Runner()
    pset = set_pset(runner, is_arc=False)

    return [
        str(creator.Individual.from_string(p, pset))
        for p in programs
        if is_valid_program(pset, p)
    ]
