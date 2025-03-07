import numpy as np


def _population_statistics(name, values):
    return f"{name}|min|{np.min(values)}|max|{np.max(values)}|mean|{np.mean(values)}|std|{np.std(values)}"


def population_height_statistics(population):
    heights = [program.height for program, _ in population]
    return _population_statistics("population_height", heights)


def population_length_statistics(population):
    lengths = [len(program) for program, _ in population]
    return _population_statistics("population_length", lengths)


def population_performance_statistics(population):
    fitness_scores = [fitness for _, fitness in population]
    return _population_statistics("population_performance", fitness_scores)
