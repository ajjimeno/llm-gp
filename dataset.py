from glob import glob


def get_training_examples(problem):
    input_folder = (
        f"/home/antonio/Documents/data/experiments/{problem}/training/"  # noqa
    )

    examples = []

    for filename in glob(f"{input_folder}/*.txt"):
        with open(filename) as f:
            data = f.read().split("\n")

        pointer = 0

        instance = []
        for i in range(int(data[0])):
            example = {}

            pointer += 1
            input = []
            for j in range(int(data[pointer].split()[0])):
                pointer += 1
                input.append([int(n) for n in data[pointer].split()])

            if len(input) == 1:
                example["input"] = input[0]
            else:
                example["input"] = input

            output = []

            pointer += 1

            for j in range(int(data[pointer].split()[0])):
                pointer += 1
                output.append([int(n) for n in data[pointer].split()])

            if len(output) == 1:
                example["output"] = output[0]
            else:
                example["output"] = input

            instance.append(example)

        examples.append(instance)

    result = "["

    for example in examples:
        result += '{"example:"' + str(example) + "},\n"

    result += "]"

    return result
