import json
import os
import random
import sys
from statistics import (
    population_height_statistics,
    population_length_statistics,
    population_performance_statistics,
)

import numpy as np

from gp_algorithm import PrimitiveTree
from logger_config import getLogger

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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from selection import selStochasticUniversalSampling

load_dotenv()

_task_env = os.getenv("RUNNING_TASK", "")
IS_ARC = _task_env not in ["count", "inverse", "sorted", "max-min", "mixed", "mixed-100"]

if IS_ARC:
    from programs_check_dsl import get_primitive_tree, get_valid_program, toolbox, wrap_with_segment_render
    from prompts_dsl import DSLGeneticPrompting
else:
    from initial_population import get_population
    from programs_check import get_primitive_tree, get_valid_program, toolbox
    from prompts import GeneticPrompting
    from read_experiments import get_previous_programs


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


def loo_select(population, training_dir, simulator_module, wrap_fn):
    """Leave-one-out validation: among Training=1.0 programs, return the one
    that generalises best across all LOO splits of the training data.

    Parses all pairs from the FEE training file (fit pairs + held-out pair),
    creates N temporary runners (each leaving one pair out), evaluates every
    unique Training=1.0 candidate, and returns the program with the highest
    min LOO score (ties broken by average, then by length).
    """
    import tempfile, shutil, glob

    # Find the single .txt training file in training_dir
    txt_files = glob.glob(os.path.join(training_dir, "*.txt"))
    if not txt_files:
        return None
    training_file = txt_files[0]

    # Parse all pairs from the FEE file
    with open(training_file) as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    # FEE format: line 0 = n_fit_pairs, then n_fit_pairs × (h w / rows / h w / rows),
    # then held_out = (h w / rows / h w / rows / h w / rows)
    n_fit = int(lines[0])
    pairs = []  # list of (input_block, output_block) each as list of strings

    idx = 1
    for _ in range(n_fit):
        h, w = map(int, lines[idx].split())
        in_block  = [lines[idx]] + [lines[idx + 1 + r] for r in range(h)]
        idx += 1 + h
        h2, w2 = map(int, lines[idx].split())
        out_block = [lines[idx]] + [lines[idx + 1 + r] for r in range(h2)]
        idx += 1 + h2
        pairs.append((in_block, out_block))

    # Held-out pair
    if idx < len(lines) and lines[idx].strip():
        h, w = map(int, lines[idx].split())
        in_block  = [lines[idx]] + [lines[idx + 1 + r] for r in range(h)]
        idx += 1 + h
        h2, w2 = map(int, lines[idx].split())
        out_block = [lines[idx]] + [lines[idx + 1 + r] for r in range(h2)]
        pairs.append((in_block, out_block))

    N = len(pairs)
    if N < 2:
        return None  # nothing to cross-validate

    # Collect unique Training=1.0 candidates (as wrapped strings)
    candidates = list({wrap_fn(str(ind[0])): ind
                       for ind in population if ind[1] >= 1.0}.items())
    if not candidates:
        return None

    # Evaluate each candidate on every LOO split
    loo_scores = {prog_str: [] for prog_str, _ in candidates}

    for held_out_idx in range(N):
        fit = [p for i, p in enumerate(pairs) if i != held_out_idx]
        held = pairs[held_out_idx]

        # Build the FEE file content for this split
        file_lines = [str(len(fit))]
        for in_b, out_b in fit:
            file_lines.extend(in_b)
            file_lines.extend(out_b)
        in_b, out_b = held
        file_lines.extend(in_b)   # held-out input
        file_lines.extend(out_b)  # held-out output
        file_lines.extend(in_b)   # held-out input repeated (FEE format)

        tmp_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp_dir, "loo.txt"), "w") as f:
                f.write("\n".join(file_lines) + "\n")
            s_loo = simulator_module.Runner(tmp_dir)
            prog_strs = [p for p, _ in candidates]
            scores = s_loo.run(prog_strs)
            for prog_str, score in zip(prog_strs, scores):
                loo_scores[prog_str].append(score)
        finally:
            shutil.rmtree(tmp_dir)

    # Rank by: min LOO score desc, then avg LOO score desc, then length asc
    def rank_key(item):
        prog_str, ind = item
        scores = loo_scores[prog_str]
        return (-min(scores), -sum(scores) / len(scores), len(ind[0]))

    best_prog_str, best_ind = min(candidates, key=rank_key)
    return best_ind


def evaluate_population(population, simulator, to_program=str):
    negative_indices = [
        i for i, (program, score) in enumerate(population) if score == -1
    ]
    negative_individuals = [
        to_program(program)
        for _, (program, score) in enumerate(population)
        if score == -1
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

    data_folder = os.getenv("DATA_FOLDER")

    population = []

    if IS_ARC:
        # ARC mode: task is a path to a directory of pre-converted .txt task files
        # (produced by arc_json_to_fee.py).  DATA_FOLDER/arc/training/ and
        # DATA_FOLDER/arc/testing/ both contain the same .txt files; FEE uses the
        # embedded test pair for fitness.
        training_dir = f"{data_folder}/{task}/training"
        testing_dir  = f"{data_folder}/{task}/testing"

        # Load the raw ARC JSON so the LLM can see the grids
        arc_json_path = os.getenv("ARC_JSON_PATH")
        arc_task_id   = os.getenv("ARC_TASK_ID")

        if llm_population_generation:
            with open(arc_json_path) as f:
                all_tasks = json.load(f)
            task_data = all_tasks[arc_task_id]
            prompting = DSLGeneticPrompting(task_data)
            description = prompting.get_best_description()
            population = prompting.get_problem_programs(description)

        s         = simulator.Runner(training_dir)
        s_testing = simulator.Runner(testing_dir)

        # Random initial population from the DSL pset.
        # Programs are stored as inner trees (without segment/render);
        # wrap_with_segment_render is applied at every evaluation call.
        llm_inners = [get_primitive_tree(str(p)) for p in population if get_valid_program(str(p))]
        random_inners = [toolbox.individual() for _ in range(population_size)]
        all_inners = llm_inners + random_inners

        wrapped_strs = [wrap_with_segment_render(str(ind)) for ind in all_inners]
        scores = s.run(wrapped_strs)
        population = list(zip(all_inners, scores))

    else:
        # Original list-task mode
        if llm_population_generation:
            prompting = GeneticPrompting()
            description = prompting.get_best_description(task)
            population = prompting.get_problem_programs(description)

        s         = simulator.Runner(f"{data_folder}/{task}/training")
        s_testing = simulator.Runner(f"{data_folder}/{task}/testing")

        population += get_population(population_size)
        population = [get_valid_program(program) for program in get_previous_programs(task)]
        population = list(zip(population, s.run(population)))

    elitism_individual = None

    max_epochs = int(os.getenv("MAX_EPOCHS", 1500))
    for epoch in tqdm(range(max_epochs)):
        min_max_length, elitism_individual = get_top_individual(population)

        top_prog_str = wrap_with_segment_render(str(elitism_individual[0])) if IS_ARC else str(elitism_individual[0])
        logger.info(
            f"Epoch|{epoch}|Top|{top_prog_str}|Length|{len(elitism_individual[0])}|Height|{elitism_individual[0].height}|Training|{elitism_individual[1]}|Testing|{s_testing.run([top_prog_str])[0]}"
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

        filtered = [
            individual
            for individual in population
            if len(individual[0]) < min_max_length + 25
        ]
        if not filtered:
            filtered = population

        population = [
            (PrimitiveTree(program), score)
            for program, score in selStochasticUniversalSampling(
                filtered,
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

        evaluate_population(
            population, s,
            to_program=wrap_with_segment_render if IS_ARC else str,
        )

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

                    # if new_score >= mean_score:
                    if new_program != str(individual[0]):
                        population.append((get_primitive_tree(new_program), new_score))

        if llm_elite_mutation:
            new_program = prompting.get_guided_mutation_program(
                description, elitism_individual
            )

            if new_program:
                new_score = s.run([new_program])[0]

                logger.info(
                    f"Epoch|{epoch}|mutation|initial|{str(elitism_individual[0])}|mutated|{new_program}|{new_score}|mean|{mean_score}"
                )

                # if new_score >= mean_score:
                if new_program != str(elitism_individual[0]):
                    population.append((get_primitive_tree(new_program), new_score))

        population.append(elitism_individual)

    # ------------------------------------------------------------------ #
    # Post-hoc LOO validation: among all Training=1.0 programs in the     #
    # final population, pick the one that generalises best across all      #
    # leave-one-out splits of the training data.                           #
    # Only applied in IS_ARC mode (has a single training file per task).  #
    # ------------------------------------------------------------------ #
    if IS_ARC and elitism_individual is not None and elitism_individual[1] >= 1.0:
        loo_winner = loo_select(
            population, training_dir, simulator,
            wrap_fn=wrap_with_segment_render,
        )
        if loo_winner is not None:
            logger.info(
                f"LOO|elite_before|{wrap_with_segment_render(str(elitism_individual[0]))}|"
                f"loo_winner|{wrap_with_segment_render(str(loo_winner[0]))}"
            )
            elitism_individual = loo_winner
