from gp_algorithm import PrimitiveTree
import json
from logger_config import getLogger
import numpy as np
import os
import random
from statistics import (
    population_height_statistics,
    population_length_statistics,
    population_performance_statistics,
)
import sys

gp_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../gp")
# Get the current PYTHONPATH (if any)
current_path = os.environ.get("PYTHONPATH", "")

# Add new path to PYTHONPATH environment variable
os.environ["PYTHONPATH"] = f"{gp_path}:{current_path}"

# Also add to sys.path for the current process
sys.path.insert(0, gp_path)

import SimulatorCPU as simulator

from dotenv import dotenv_values, load_dotenv
from tqdm import tqdm

from initial_population import get_population
from programs_check import (
    get_primitive_tree,
    toolbox,
)
from prompts import GeneticPrompting

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from selection import selStochasticUniversalSampling

load_dotenv()


def str2bool(value) -> bool:
    return value.lower() == "true"


def get_top_individual(population):
    program, max_score = population[0]
    min_max_length = len(program)
    elitism_individual = population[0]

    for individual in population[1:]:
        program, fitness = individual
        if fitness > max_score:
            max_score = fitness
            min_max_length = len(program)
            elitism_individual = individual
            continue

        if fitness == max_score:
            n_length = len(program)
            if min_max_length > n_length:
                min_max_length = n_length
                elitism_individual = individual

    return min_max_length, elitism_individual 


def evaluate_population(population, simulator):
    negative_indices = [
        i for i, (program, score) in enumerate(population) if score == -1
    ]
    negative_individuals = [
        program for _, (program, score) in enumerate(population) if score == -1
    ]

    if not negative_individuals:
        return population

    updated_values = simulator.run(negative_individuals)

    for i, new_value in zip(negative_indices, updated_values):
        population[i] = (population[i][0], new_value)

    return population


if __name__ == "__main__":

    logger = getLogger(__name__)

    population_size = int(os.getenv("POPULATION_SIZE", 300))

    mutation_probability = float(os.getenv("MUTATION_PROBABILITY", 0.50))
    crossover_probability = float(os.getenv("CROSSOVER_PROBABILITY", 0.50))

    llm_population_generation = str2bool(
        os.getenv("LLM_POPULATION_GENERATION", "False")
    )

    if llm_population_generation:
        llm_mutation_probability = float(os.getenv("LLM_MUTATION_PROBABILITY", 0.99))
        llm_elite_mutation = str2bool(os.getenv("LLM_ELITE_MUTATION", "False"))
    else:
        llm_mutation_probability = 1.0
        llm_elite_mutation = False

    task = os.getenv("RUNNING_TASK")

    logger.info(json.dumps(dotenv_values()))
    logger.info(task)

    if task not in [
        "count",
        "inverse",
        "sorted",
        "max-min",
        "mixed",
        "mixed-100",
    ]:
        raise ValueError("Revise task")

    population = []

    if llm_population_generation:
        prompting = GeneticPrompting()

        description = prompting.get_best_description(task)

        population = prompting.get_problem_programs(description)

    s = simulator.Runner(f"{os.getenv('DATA_FOLDER')}/{task}/training")
    s_testing = simulator.Runner(f"{os.getenv('DATA_FOLDER')}/{task}/testing")

    population += get_population(population_size)

    population = list(zip(population, s.run(population)))

    elitism_individual = None

    for epoch in tqdm(range(5)):
        min_max_length, elitism_individual = get_top_individual(population)

        logger.info(
            f"Epoch|{epoch}|Top|{elitism_individual[0]}|Length|{len(elitism_individual[0])}|Height|{elitism_individual[0].height}|Training|{elitism_individual[1]}|Testing|{s_testing.run([elitism_individual[0]])[0]}"
        )

        for function in [
            population_performance_statistics,
            population_length_statistics,
            population_height_statistics,
        ]:
            logger.info(f"Epoch|{epoch}|{function(population)}")

        mean_score = np.mean([individual[1] for individual in population])

        with open("programs.txt", "w") as f:
            for individual in population:
                print(str(individual), file=f)

        population = [
            (PrimitiveTree(program), score)
            for program, score in selStochasticUniversalSampling(
                [
                    individual
                    for individual in population
                    if len(individual[0]) < min_max_length + 25
                ],
                k=population_size,
            )
        ]

        for i in tqdm(range(0, len(population), 2)):
            if random.random() > crossover_probability:
                if i + 1 >= len(population):
                    break

                individual1, _ = population[i]
                individual2, _ = population[i + 1]

                program1, program2 = toolbox.mate(individual1, individual2)

                population[i] = (program1, -1)
                population[i + 1] = (program2, -1)

        for i in tqdm(range(len(population))):
            if random.random() > mutation_probability:
                individual, _ = population[i]

                (new_program,) = toolbox.mutate(individual)

                population[i] = (new_program, -1)

        evaluate_population(population, s)

        for i in tqdm(range(len(population))):
            if random.random() > llm_mutation_probability:
                individual = population[i]
                new_program = prompting.get_guided_mutation_program(
                    description, individual
                )

                if new_program:
                    new_score = s.run([new_program])[0]

                    logger.info(
                        f"Epoch|{epoch}|mutation|initial|{str(individual[0])}|mutated|{new_program}|{new_score}|mean|{mean_score}"
                    )

                    #if new_score >= mean_score:
                    if new_program != str(individual[0]):
                        population.append(
                            (get_primitive_tree(new_program), new_score)
                        )

        if llm_elite_mutation:
            new_program = prompting.get_guided_mutation_program(
                description, elitism_individual
            )
        
            if new_program:
                new_score = s.run([new_program])[0]

                logger.info(
                    f"Epoch|{epoch}|mutation|initial|{str(elitism_individual[0])}|mutated|{new_program}|{new_score}|mean|{mean_score}"
                )

                #if new_score >= mean_score:
                if new_program != str(elitism_individual[0]):
                    population.append(
                        (get_primitive_tree(new_program), new_score)
                    )

    
        population.append(elitism_individual)
