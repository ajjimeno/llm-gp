import json
import os
import random
from glob import glob


def read_matrix(file_obj):
    num_rows = int(next(file_obj).split()[0])
    return [[int(n) for n in next(file_obj).split()] for _ in range(num_rows)]


def get_training_examples(problem):
    input_folder = os.path.join(
        "/home/antonio/Documents/data/experiments", problem, "training"  # noqa
    )

    examples = []

    filenames = glob(os.path.join(input_folder, "*.txt"))
    random.shuffle(filenames)

    for filename in filenames[:100]:
        instance = []
        with open(filename) as f:
            num_examples = int(next(f))  # Read the number of examples

            for _ in range(num_examples):
                example = {}

                def read_matrix(file_obj):
                    num_rows = int(next(file_obj).split()[0])
                    matrix = []
                    for _ in range(num_rows):
                        row = [int(n) for n in next(file_obj).split()]
                        matrix.append(row)
                    return matrix

                example["input"] = read_matrix(f)
                example["output"] = read_matrix(f)

                if len(example["input"]) == 1:
                    example["input"] = example["input"][0]
                if len(example["output"]) == 1:
                    example["output"] = example["output"][0]
                instance.append(example)

        examples.append(instance)

    return json.dumps(examples, indent=4)
