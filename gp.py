import datetime
import itertools
import json
import os
import random
import sys

import SimulatorCPU as simulator
from deap import gp
from dotenv import load_dotenv
from tqdm import tqdm

from initial_population import get_population
from programs_check import (
    get_program_length,
    check_program_length,
    get_valid_program,
    toolbox,
)
from prompts import GeneticPrompting
from selection import selStochasticUniversalSampling

load_dotenv()


def get_top_individual(population):
    max_score = population[0][1]
    min_max_length = get_program_length(population[0][0])
    elitism_individual = population[0]

    for individual in population[1:]:
        if individual[1] > max_score:
            max_score = individual[1]
            min_max_length = get_program_length(individual[0])
            elitism_individual = individual
            continue

        if individual[1] == max_score:
            n_length = get_program_length(individual[0])
            if min_max_length > n_length:
                min_max_length = n_length
                elitism_individual = individual

    return min_max_length, elitism_individual


if __name__ == "__main__":

    population_size = int(os.getenv("POPULATION_SIZE", 300))

    mutation_probability = float(os.getenv("MUTATION_PROBABILITY", 0.50))
    crossover_probability = float(os.getenv("CROSSOVER_PROBABILITY", 0.50))

    llm_mutation_probability = float(os.getenv("LLM_MUTATION_PROBABILITY", 0.99))
    llm_elite_mutation = bool(os.getenv("LLM_ELITE_MUTATION", True))

    task = os.getenv("RUNNING_TASK")
    running_mode = os.getenv("RUNNING_MODE")

    print(f"Task: {task}, running_mode = {running_mode}")

    if task not in [
        "count",
        "inverse",
        "sorted",
        "max-min",
        "mixed",
        "mixed-100",
    ] or running_mode not in ["full", "initial"]:
        raise ValueError("Revise task and running mode")

    prompting = GeneticPrompting()

    description = prompting.get_problem_description(task)

    population = prompting.get_problem_programs(description)

    if running_mode == "initial":
        with open(f"programs-{task}-{datetime.datetime.now()}.txt", "w") as f:
            json.dump(population, f)

        sys.exit(0)

    s = simulator.Runner(f"{os.getenv('DATA_FOLDER')}/{task}/training")

    population += get_population(population_size)

    population = list(zip(population, s.run(population)))

    print(population)

    print(sorted(population, key=lambda x: x[1], reverse=True)[:5])

    elitism_individual = None

    for i in tqdm(range(1500)):
        print(f"Epoch {i}")

        min_max_length, elitism_individual = get_top_individual(population)

        population = selStochasticUniversalSampling(
            [
                individual
                for individual in population
                if check_program_length(individual[0], min_max_length + 25)
            ],
            k=population_size,
        )

        print(f"Top: {elitism_individual}")

        with open("programs.txt", "w") as f:
            for individual in population:
                print(individual, file=f)

        for i in tqdm(range(0, len(population), 2)):
            if random.random() > crossover_probability:
                if i + 1 >= len(population):
                    break

                individual1 = population[i]
                individual2 = population[i + 1]

                program1, program2 = toolbox.mate(
                    get_valid_program(individual1[0]), get_valid_program(individual2[0])
                )

                program1 = str(program1)
                program2 = str(program2)

                population[i] = (program1, s.run([program1])[0])
                population[i + 1] = (program2, s.run([program2])[0])

        for i in tqdm(range(len(population))):
            if random.random() > mutation_probability:
                individual = population[i]

                new_program = str(toolbox.mutate(get_valid_program(individual[0]))[0])

                population[i] = (new_program, s.run([new_program])[0])

        for i in tqdm(range(len(population))):
            if random.random() > llm_mutation_probability:
                individual = population[i]
                new_program = prompting.get_guided_mutation_program(
                    description, individual
                )

                if new_program != individual[0]:
                    population.append((new_program, s.run([new_program])[0]))

        if llm_elite_mutation:
            new_program = prompting.get_guided_mutation_program(
                description, elitism_individual
            )

            if new_program != elitism_individual[0]:
                population.append((new_program, s.run([new_program])[0]))

        population.append(elitism_individual)
