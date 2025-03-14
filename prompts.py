import json
import os
import random
import re

from dotenv import load_dotenv
from tqdm import tqdm

from dataset import get_training_examples
from llm import get_model
from logger_config import getLogger
from programs_check import check_programs, get_primitive_tree, get_valid_program


load_dotenv()

logger = getLogger(__name__)


def clean_output(string):
    return (
        string.replace("```python\n", "")
        .replace("```\n", "")
        .replace("\n", "")
        .replace("```", "")
        .replace(" ", "")
        .replace("```json\n", "")
        .replace("```\n", "")
        .replace("plaintext", "")
    )


def extract_python_code(text, prefix="python"):
    match = re.search(rf"```{prefix}(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


system_prompt = (
    "You are an expert assistant that is an expert in evolutionary algorithms."
)


functions = """

These are the available primitives to build the output function group by different properties describe at the beginning of the group
The primitives are defined by the name, the types of attributes and a description of what they do.
A primitive returns a value if it is mentioned explicitly.
No value from the training primitives are used for the testing primitives.

Numbers from 0 to 9 use the following primitives:

get0(): return 0
get1(): return 1
get2(): return 2
get3(): return 3
get4(): return 4
get5(): return 5
get6(): return 6
get7(): return 7
get8(): return 8
get9(): return 9

These are primitives that an be used to control the flow, which can be used with training and testing primitives:

loop (Integer, Operation): iterates using the integer value and executes Operation
prog2 (Operation, Operation), executes two operations so it always needs two arguments or the interpreter fails, e.g, prog2(testing_output_move_right(), testing_output_move_right())
comparison (Boolean, Operation, Operation): the Boolean value is used to execute the first Operation if true or the second Operation otherwise

Comparison of elements from the training set, they should be used with functions expected to return an integer from the training set of functions.
The functions that do not return a value should not be used.

bigger_thanW(Integer, Integer): true if the first integer is bigger than the second one, only used with values returned using training functions. Do not use any get number function, e.g. get6().
equalW(Integer, Integer): true if the two integers are equal, only used with values returned using training functions

These are functions that work on the training input list:

input_next(): moves to the next training example list. No value is returned.
input_previous(): moves to the previous training example list. No value is returned.
input_max(): returns the maximum value of the current training input example.
input_min(): returns the minimum value of the current training input example.
input_read(): returns the value at the current position of the training input list.
input_move_left(): moves the current position in the training input list to the left. No value is returned.
input_move_right(): moves the current position in the training input list to the right. No value is returned.
get_length_input_x(): returns an integer with the length of the training input list.
reset_input_position(): resets the position to the beginning of the training input list. No value is returned.

These are primitives that work on the training output list:

output_read(): returns the value at the current position of the training output list
get_length_output_x(): returns an integer with the length of the training output list
output_move_left(): moves the current position in the training output list to the left
output_move_right(): moves the current position in the training output list to the right
reset_output_position(): resets the position to the beginning of the training output list
bigger_than_output_next(): returs true if the next value is larger in the training output list


Comparison of elements from the testing set, they should be used with primitives expected to return an integer from the testing set of primitives.
The primitives that do not return a value should not be used.

bigger_thanR(Integer, Integer): true if the first integer is bigger than the second one, only used with values returned using testing primitives. Do not use any get number primitive, e.g. get6().
equalR(Integer, Integer): true if the two integers are equal, only used with values returned using testing primitives

The following primitives work on the testing input list:

testing_input_max (): returns the maximum value of the testing input list
testing_input_min (): returns the minimum value of the testing input list
testing_input_read (): returns the current value of the testing input list
get_testing_length_input_x (): returns an integer with the length of the testing input list
testing_input_move_right (): moves the pointer to the list to the right, but does not come back to the initial position if overflown
testing_reset_input_position (): sets the position of the pointer to zero

The following primitives work on the testing output list:

testing_output_read (): returns the current value of the testing output list
testing_output_write(Integer): writes the Integer in the current position of the testing output list. Only values or properties read from the testing input or output list can be used valid inputs for this primitive. 
get_testing_length_output_x (): returns an integer the length of the testing list
bigger_than_testing_output_next (): returs true if the next value is larger in the testing output list
swap_testing_output_next (): interchanges the current value in the test output list with the next one of the test output list. No value is returned.
testing_output_move_right (): moves the pointer to the list to the right, but does not come back to the initial position if overflown. No value is returned.
testing_reset_output_position (): sets the position of the pointer to zero. No value is returned.

The following are examples of functions that are incorrect:

1. The one below because testing_output_write does not return any value:
<example>bigger_thanR(output_read(),testing_output_write(testing_input_min()))</example>
2. The one below because a result from a training primitive is used in a testing primitive:
<example>equalW(input_max(),testing_output_read())</example>
3. The one below uses equalR, when it should use equalW for the training primitives:
<example>equalR(input_max(),output_read())</example>

When writing the functions consider that the evaluation is done on the testing output, there should be a primitive that writes to the output used in the generated functions.
When writing the functions, the training set should be used to decide what is the action to do on the testing output list.

The following functions are only an example of valid syntax and is not an example of functions to be written.

<example>
comparison(equalW(input_min(), input_read()),prog2(testing_output_move_right(), swap_testing_output_next()),loop(get_testing_length_output_x(), swap_testing_output_next()))
</example>
<example>
comparison(equalW(input_min(), input_read()),prog2(testing_output_move_right(), swap_testing_output_next()),loop(get_testing_length_output_x(), testing_output_write(testing_input_max())))
</example>
<example>
prog2(testing_reset_output_position(), testing_output_write(testing_input_read()))
</example>
<example>
loop(get_testing_length_output_x(), testing_output_move_right())
</example>

"""


class GeneticPrompting:
    def __init__(self):
        self.model = get_model(model_name=os.getenv("LLM_NAME"))


    def get_best_description(self, task, retries=5):
        descriptions = "\n".join([f"<description id='{i}'>" + self.get_problem_description(task) + "</description>" for i in range(retries)])

        user_prompt = f"""
            ** Task: Given the following descriptions and examples, select the description that better explains the examples and summarise it in the output.
            Only output the new description. A description is better if it is simpler and if it manages to explain the examples.

            "***descriptions***
            {descriptions}

            ***examples***
            {get_training_examples(task)}

            """


        description = self.model(
            system_prompt=system_prompt, user_prompt =user_prompt
        )

        logger.info(description)

        return description


    def get_problem_description(self, task):
        problem_description = f"""
**Task:** Analyze the following training and testing input-output list examples to determine the underlying transformation function.

**Format:** List of lists with dictionaries that represent examples: `[input_list] -> [output_list]`

**Constraints:**

* The transformation from input to output may vary across examples, but it is similar.
* Within a single example, the transformation is consistently applied.
* Identify patterns, including those that select, modify, or rearrange elements.
* Provide a detailed explanation of the transformation rule(s) for each example, and a general explanation of all the examples.
* Provide multiple possible explanations if the data permits.

**Examples:**

{get_training_examples(task)}

** Analysis **

Your analysis starts here:
"""

        description = self.model(
            system_prompt=system_prompt, user_prompt=problem_description
        )

        logger.info(description)

        return description

    def get_problem_programs(self, description):
        count = 0

        population = []
        # TODO: shold we parameterise this constant in the config
        max_count = 80
        pbar = tqdm(total=max_count)

        while count < max_count:
            pbar.update(1)
            pbar.set_description(f"obtaining problem program {count} of {max_count}")
            try:
                output = self.model(
                    system_prompt=system_prompt,
                    user_prompt="""
                Get inspiration from the following explanation of the problem:

                <explanation>
                """
                    + description
                    + """
                </explanation>

                Write 20 candidate functions inspired to solve the problem above, trying to capture different aspects.
                The functions should be expressed as a tree structure using parenthesis and commas to separate attributes where needed, without any python code.
                No other python code or function should be added than the function, do not add any number or function that is not mentioned in the list of valid functions.
                The functions should identify a relation between the input and output lists in the training set and apply it to the lists of the testing set.
                Review the new functions, so they are fully compliant with the function specification.
                Functions will be evaluated on their performance on the testing set.

                The format of the output should follow the same JSON format below.
                Do not add any explanation, just the function in the example attribute.
                Do not enclose the functions with parenthesis and do not enumerate them.
                Try to diversify the functions used in the generated functions.

                This is an example of output functions.
                This is the format that should be used to generate the functions, it is a list of dictionaries in which the example attribute has one of the functions.

                [
                {"example": "comparison(equal(input_min(), input_max),prog2(testing_output_move_right(), swap_testing_output_next()),loop(get_testing_length_output_x(), testing_output_write(testing_input_max())))"}
                ]

                """
                    + functions,
                )

                logger.info(f"Population generation output: {output}")

                output_list = extract_python_code(output, prefix="json")

                if not output_list:
                    output_list = extract_python_code(output, prefix="")

                if not output_list:
                    output_list = output

                population += [
                    e[list(e.keys())[0]]
                    #e["example"]
                    for e in json.loads(output_list)
                    # if not is_individual_invalid(e["example"])
                ]

                population = check_programs(population)
                logger.info(f"Filtered programs: {population}")

                if len(population) > 30:
                    break
            except Exception as e:
                logger.error(f"Error: {e}")

            count += 1
            logger.info(f"Trying again {count}")

        return [get_primitive_tree(individual) for individual in set(population)]

    def _get_guided_mutation_prompt(self, description, individual, score):
        return f"""
    Write a new function using the following function, the explanation of the task, and the primitive specification below to build the function tree, do not add any number or primitive that is not mentioned in the list of valid primitives.
    Output only the mutated function tree, formatted as a single line with parentheses.
    
    A mutation is defined as the following steps:
    
    1. Examine the function tree branches with respect to the description of the task
    2. Identify a subtree that contributes to a significant error in the function's output based on the task description, otherwise search for a subtree that appears redundant
    3. Create a new valid and improved subtree using the available primitives and valid primitive arguments and replace the identified subtree
    
    Use number {random.randint(0,9999999)} as random seed for the generation of the mutation.
    The tree mutation should be inspired by the function accuracy and the the description of the task mentioned below.
    The mutated function has to be different to the function below.

    This is the function that has to be mutated.
    Do not show any description of the steps done during the mutation, only the mutated function needs to appear in the output.
    
    {individual}

    When evaluated, the function above has an accuracy of {score} of 1.0.
    Accuracy is calculated as proportion of correct elements in the testing output list. 

    Ensure the mutated function uses only the provided primitives and arguments. Do not introduce any new symbols or operations.
    Review the new function, so it is fully compliant with the primitives.

    {functions}

    The task has the following description, which can be used to guide the mutated function to get a higher score.

    <explanation>
    {description}
    </explanation>
    """

    def get_guided_mutation_program(self, description, individual):
        count = 0

        individual = (str(individual[0]), individual[1])
        while count < 3:
            try:
                new_program = self._get_guided_mutation_program(description, individual)
                return new_program
            except Exception:
                count += 1

        return None

    def _get_guided_mutation_program(self, description, individual):
        program = clean_output(
            self.model(
                system_prompt=system_prompt,
                user_prompt=self._get_guided_mutation_prompt(
                    description, individual[0], individual[1]
                ),
            )
        )

        byte_program = get_valid_program(program)

        if not byte_program:
            logger.info(f"Failed program: {program}")
            raise ValueError(program)

        logger.info(f"Generated program: {program}")
        return str(byte_program)


if __name__ == "__main__":
    prompting = GeneticPrompting()

    for task in ["sorted", "count", "inverse", "max-min", "mixed"]:
        description = prompting.get_best_description(task)

        #with open(f"programs-{task}.json", "w") as f:
        #    json.dump(prompting.get_problem_programs(description), f)

        break
