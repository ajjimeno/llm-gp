import numpy as np
from programs_check import get_program_length


def population_length_statistics(population):
    lengths = [get_program_length(individual[0]) for individual in population]
    return f"min|{np.min(lengths)}|max|{np.max(lengths)}|mean|{np.mean(lengths)}|std|{np.std(lengths)}"


def population_performance_statistics(population):
    performance = [individual[1] for individual in population]
    return f"min|{np.min(performance)}|max|{np.max(performance)}|mean|{np.mean(performance)}|std|{np.std(performance)}"
