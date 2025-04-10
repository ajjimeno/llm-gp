from programs_check import toolbox


def get_population(population_size):
    pop = toolbox.population(population_size)
    return pop


if __name__ == "__main__":
    print(get_population(100))
