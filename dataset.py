import json
import os
import random
from glob import glob


def read_matrix(file_obj):
    num_rows = int(next(file_obj).split()[0])
    return [[int(n) for n in next(file_obj).split()] for _ in range(num_rows)]


def get_training_examples(problem):
    input_folder = os.path.join(os.getenv("DATA_FOLDER"), problem, "training")  # noqa

    examples = []

    filenames = glob(os.path.join(input_folder, "*.txt"))
    random.shuffle(filenames)

    for filename in filenames[:100]:
        with open(filename) as f:
            # training
            training = []

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
                training.append(example)

            # testing
            testing = {"input": read_matrix(f), "output": read_matrix(f)}

        examples.append({"training": training, "testing": testing})

    return json.dumps(examples)
