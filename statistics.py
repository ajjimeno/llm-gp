import numpy as np


def _population_statistics(values):
    return f"min|{np.min(values)}|max|{np.max(values)}|mean|{np.mean(values)}|std|{np.std(values)}"


def population_length_statistics(population):
    lengths = [len(program) for program, _ in population]
    return _population_statistics(lengths)


def population_performance_statistics(population):
    fitness_scores = [fitness for _, fitness in population]
    return _population_statistics(fitness_scores)
