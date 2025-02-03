from glob import glob
from os.path import exists

import gp_algorithm
from deap import base, creator, gp, tools
from runner import Runner, set_pset


def get_population(population_size):
    pset = set_pset(Runner(), is_arc=False)

    creator.create("FitnessMin", base.Fitness, weights=(1.0, 0.000000000000005))
    creator.create("Individual", gp_algorithm.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(population_size)

    return [str(p) for p in pop]


if __name__ == "__main__":
    print(get_population(100))
