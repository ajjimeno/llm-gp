import datetime
import json
import os
import random
import sys

import SimulatorCPU as simulator
from dotenv import load_dotenv
from tqdm import tqdm

from initial_population import get_population
from programs_check import check_programs
from prompts import GeneticPrompting

load_dotenv()

population_size = 300

task = os.getenv("RUNNING_TASK")
running_mode = os.getenv("RUNNING_MODE")

if task not in [
    "count",
    "inverse",
    "sorted",
    "max-min",
    "mixed",
] or running_mode not in ["full", "initial"]:
    raise ValueError("Revise task and running mode")

prompting = GeneticPrompting()

description = prompting.get_problem_description(task)

population = prompting.get_problem_programs(description)

if running_mode == "initial":
    with open(f"programs-{task}-{datetime.datetime.now()}.txt", "w") as f:
        json.dump(population, f)

    sys.exit(0)

s = simulator.Runner(f"/home/antonio/Documents/data/experiments/{task}/training")

population += get_population(population_size)

population = list(zip(population, s.run(population)))

print(population)

print(sorted(population, key=lambda x: x[1], reverse=True)[:5])

for i in tqdm(range(1500)):
    print(f"Epoch {i}")

    mutations = prompting.get_guided_mutation_programs(description, population)

    xovers = prompting.get_guided_x_over_programs(description, population)

    new_population = [individual for individual in (mutations + xovers)]

    random.shuffle(new_population)

    new_population = check_programs(new_population)

    with open("programs.txt", "w") as f:
        for individual in new_population:
            print(individual, file=f)

    new_population = population + list(zip(new_population, s.run(new_population)))

    sorted_population = sorted(new_population, key=lambda x: x[1], reverse=True)

    print(sorted_population[:5])

    # Selection
    population = sorted_population[:population_size]

    random.shuffle(population)
