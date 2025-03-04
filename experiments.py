import os
import subprocess
from itertools import product


def generate_filename_from_args(arg_dict, iter, prefix="experiment"):
    """
    Generates a filename from an argument dictionary.
    """
    filename_parts = [prefix]
    for key, value in arg_dict.items():
        # Handle different value types (lists, booleans, etc.)
        if isinstance(value, list):
            value_str = "-".join(map(str, value))  # Join list elements with '-'
        else:
            value_str = str(value)

        # Sanitize key and value for filename safety
        key_sanitized = "".join(c if c.isalnum() else "_" for c in key)
        value_sanitized = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in value_str
        )

        filename_parts.append(f"{key_sanitized}_{value_sanitized}")

    return "_".join(filename_parts) + f"_iter_{iter}.log"


def generate_env_file(arguments, filename=".env"):
    """
    Generates a .env file with given arguments.
    """
    with open(filename, "w") as f:
        for k, v in arguments.items():
            print(f"{k}={v}", file=f)


def run_experiments(arg_dict, iter=10):
    """
    Runs experiments with all combinations of arguments.
    """
    for i in range(iter):
        keys = list(arg_dict.keys())
        values = list(arg_dict.values())
        value_combinations = product(*values)

        for combo in value_combinations:
            args = dict(zip(keys, combo))
            print(args)
            filename = generate_filename_from_args(args, i)

            generate_env_file(args)

            print(f"Running experiment with {args}")

            with open(filename, "w") as outfile:
                try:
                    subprocess.run(
                        ["python", "gp.py"], check=True, stdout=outfile
                    )  # call your script
                except subprocess.CalledProcessError as e:
                    print(f"Experiment failed: {e}")
                finally:
                    os.remove(".env")  # remove the temporary .env file


if __name__ == "__main__":
    arg_variables = {
        "RUNNING_TASK": [
            "count",
            "inverse",
            "sorted",
            "max-min",
        ],
        "DATA_FOLDER": ["./data/experiments"],
        "POPULATION_SIZE": [300, 3000],
    }

    run_experiments(arg_variables, iter=10)
