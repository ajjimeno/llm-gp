#!/usr/bin/env python
# coding: utf-8

# In[1]:

import datetime

import codeop
import json
import os
import random
import re

import SimulatorCPU as simulator
from dotenv import load_dotenv
from tqdm import tqdm

from dataset import get_training_examples
from llm import get_model
from programs_check import check_programs
import sys

load_dotenv()

task = os.getenv("RUNNING_TASK")
running_mode=os.getenv("RUNNING_MODE")

if task not in ["count", "inv-sort"] or running_mode not in ["full", "initial"]:
    raise ValueError ("Revise task and running mode")


system_prompt = (
    "You are an expert assistant that is an expert in evolutionary algorithms."
)

model = get_model(model_name=os.getenv("LLM_NAME"))


def extract_python_code(text, prefix="python"):
    match = re.search(rf"```{prefix}(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def is_syntax_correct(individual):
    try:
        codeop.compile_command(individual)
        return True
    except:
        return False


def is_individual_invalid(individual):
    return (
        "True" in individual
        or "False" in individual
        or "Boolean" in individual
        or "[" in individual
        or "notbigger" in individual
        or "plaintext" in individual
        or not is_syntax_correct(individual)
    )


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


# In[4]:


# In[5]:


functions = """

        When writing the functions consider that the evaluation is done on the testing output, there should be a function that writes to the output used in the generated functions.
        When writing the functions, the training set should be used to decide what is the action to do on the testing output list.
        
        The following example checks the values in the training set to understand what needs to be done on the testing output list. Programs should follow this.

        comparison(equalW(input_min(), input_read()),prog2(testing_output_move_right(), swap_testing_output_next()),loop(get_testing_length_output_x(), testing_output_write(testing_input_max())))

These are the available functions to build the output function group by different properties describe at the beginning of the group
The functions are defined by the name, the types of attributes and a description of what they do.
A function returns a value if it is mentioned explicitly.
No value from the training functions are used for the testing functions.

Numbers from 0 to 9 use the following functions:

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

These are functions that an be used to control the flow, which can be used with training and testing functions:

loop (Integer, Operation): iterates using the integer value and executes Operation
prog2 (Operation, Operation), executes two operations so it always needs two arguments or the interpreter fails, e.g, prog2(testing_output_move_right(), testing_output_move_right())
comparison (Boolean, Operation, Operation): the Boolean value is used to execute the first Operation if true or the second Operation otherwise

Comparison of elements from the training set, they should be used with functions expected to return an integer from the training set of functions.
The functions that do not return a value should not be used.

bigger_thanW(Integer, Integer): true if the first integer is bigger than the second one, only used with values returned using training functions
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

These are functions that work on the training output list:

output_read(): returns the value at the current position of the training output list
get_length_output_x(): returns an integer with the the length of the training output list
output_move_left(): moves the current position in the training output list to the left
output_move_right(): moves the current position in the training output list to the right
reset_output_position(): resets the position to the beginning of the training output list
bigger_than_output_next(): returs true if the next value is larger in the training output list


Comparison of elements from the testing set, they should be used with functions expected to return an integer from the testing set of functions.
The functions that do not return a value should not be used.

bigger_thanR(Integer, Integer): true if the first integer is bigger than the second one, only used with values returned using testing functions
equalR(Integer, Integer): true if the two integers are equal, only used with values returned using testing functions

The following functions work on the testing input list:

testing_input_max (): returns the maximum value of the testing input list
testing_input_min (): returns the minimum value of the testing input list
testing_input_read (): returns the current value of the testing input list
get_testing_length_input_x (): returns an integer with the length of the testing input list
testing_input_move_right (): moves the pointer to the list to the right, but does not come back to the initial position if overflown
testing_reset_input_position (): sets the position of the pointer to zero

The following functions work on the testing output list:

testing_output_read (): returns the current value of the testing output list
testing_output_write(Integer): writes the Integer in the current position of the testing output list. Only values or properties read from the testing input or output list can be used valid inputs for this function. 
get_testing_length_output_x (): returns an integer the length of the testing list
bigger_than_testing_output_next (): returs true if the next value is larger in the testing output list
swap_testing_output_next (): interchanges the current value in the test output list with the next one of the test output list. No value is returned.
testing_output_move_right (): moves the pointer to the list to the right, but does not come back to the initial position if overflown. No value is returned.
testing_reset_output_position (): sets the position of the pointer to zero. No value is returned.

"""

problem_description = (
    """
Given the following examples, one per line, that show training input and output examples of a program as lists.
Each example has several instances in which the program does the same thing to write the output list base on the input list.
Across examples, the action done to write the output list does not need to be the same.

The training examples include different tasks and the program identifies them and decides what actions need to be taken on the output list.
The test example will have a training set that will be used to decide what action to take on the test list.

Identify a complete explanation that solves all the examples, considering that there might be more than one explanation, e.g. some examples might select odd numbers and other even numbers from the input list.
Some examples might require doing the opposite of other examples, identify edge cases and make sure you have an explanation for them.
Write more than one proposal explanation if needed.
Before generating any explanation, revise your understanding of the problem with all the examples.

<examples>
"""
    + get_training_examples(task)
    + "</examples>"
)


# Show as many examples showing how the proposed solutions works, at least 10 examples.\n


# In[6]:

description = model(system_prompt=system_prompt, user_prompt=problem_description)

print(description)


count = 0

population = []

while True:
    try:
        output = model(
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
        Functions will be evaluated on their performance on the test set.

        
        The format of the output should follow the same JSON format below. Do not enclose the functions with parenthesis and do not enumerate them.
        Try to diversify the functions used in the generated functions.

        This is an example of output functions:

        [
        {"example": "comparison(equal(input_min(), input_max),prog2(testing_output_move_right(), swap_testing_output_next()),loop(get_testing_length_output_x(), testing_output_write(testing_input_max())))"}
        ]

        """
            + functions,
        )

        print(output)

        output_list = extract_python_code(output, prefix="json")

        if not output_list:
            output_list = extract_python_code(output, prefix="")

        if not output_list:
            output_list = output

        print(output_list)

        population += [
            e["example"]
            for e in json.loads(output_list)
            # if not is_individual_invalid(e["example"])
        ]

        print(population)

        population = check_programs(population)
        print(population)

        if len(population) > 20:
            break
    except Exception as e:
        print(e)

    count += 1
    print(f"Trying again {count}")

if running_mode == "initial":
    with open(f"programs-{task}-{datetime.datetime.now()}.txt", "w") as f:
        json.dump(population, f)

    sys.exit(0)


# In[14]:

# s = simulator.Runner("/home/antonio/Documents/data/experiments/sorted/training")
s = simulator.Runner(f"/home/antonio/Documents/data/experiments/{task}/training")

population = list(zip(population, s.run(population)))

print(population)


# In[15]
def get_mutation_prompt(individual):
    return f"""
Write a mutation to the following function using the function specification below to build the function tree, do not add any number or function that is not mentioned in the list of valid functions.
Use this number {random.randint(0,9999999)} as random seed for the generation of the mutation. 
The function is expressed as a tree structure using parenthesis, without any python code.
{individual}
No other python code or function should be added than the function.
Just output the mutated function.
Review the new function, so it is fully compliant with the functions.
Try to balance training and testing functions and combine functions for present in the input function.

{functions}
"""


def get_guided_mutation_prompt(individual, score):
    return (
        f"""
Write a mutation using the following function using the function specification below to build the function tree, do not add any number or function that is not mentioned in the list of valid functions.
Use this number {random.randint(0,9999999)} as random seed for the generation of the mutation.
The function is expressed as a tree structure using parenthesis, without any python code.
The mutation can be inspired by the function evaluation score and the the description of the task mentioned below.
If something is strange, consider adding functions that check the training data for action to be done on the testing output list.

This is the input function. 
{individual}

When evaluated, the function above has a score of {score} of 1.0.

No other python code or function should be added than the function.
Just output the mutated function.
Review the new function, so it is fully compliant with the functions.
Try to balance training and testing functions and combine functions for present in the input function.

{functions}

The task has the following description, which can be used to guide the mutated function to get a higher score.

<explanation>
"""
        + description
        + """
</explanation>
"""
    )


# In[18]:


def get_x_over_prompt(individual1, individual2):
    return f"""
Write a crossover function using the two functions below and build a new function tree, do not add any number or function that is not mentioned in the list of valid functions.
Use this number {random.randint(0,9999999)} as random seed for the generation of the crossover. 

The functions are expressed as a tree structure using parenthesis, without any python code.
{individual1}

{individual2}
No other python code or function should be added than the function.
Just output the cross over function.
Review the new function and the output, so it is fully compliant with the valid functions.

{functions}
"""


def get_guided_x_over_prompt(individual1, score1, individual2, score2):
    return (
        f"""
Write a crossover function using the two functions below and build a new function tree, do not add any number or function that is not mentioned in the list of valid functions.
Use this number {random.randint(0,9999999)} as random seed for the generation of the crossover. 
The functions are expressed as a tree structure using parenthesis, without any python code.

This is the first function:

{individual1}

This is the second function:

{individual2}

Each function is evaluated against their performance on the test set with float values from 0 to 1.
The firstfunction above has a score of {score1} and the second has a score of {score2}.

No other python code or function should be added than the function.
Just output the cross over function.
Review the new function and the output, so it is fully compliant with the valid functions.

{functions}

The task has the following description, which can be used to guide the crossover of the functions to get a higher score.

<explanation>
"""
        + description
        + """
</explanation>
"""
    )


# In[ ]:

for i in tqdm(range(100)):
    mutations = [
        clean_output(
            model(
                system_prompt=system_prompt,
                user_prompt=get_guided_mutation_prompt(individual[0], individual[1]),
            )
        )
        for individual in tqdm(population, position=0, leave=True)
        if random.random() > 0.5
    ]

    xovers = [
        clean_output(
            model(
                system_prompt=system_prompt,
                user_prompt=get_guided_x_over_prompt(p1[0], p1[1], p2[0], p2[1]),
            )
        )
        for p1, p2 in tqdm(
            list(zip(population[1:], population[0:-1])), position=0, leave=True
        )
        if random.random() > 0.5
    ]

    # population = population + mutations + xovers

    new_population = [clean_output(individual) for individual in (mutations + xovers)]

    random.shuffle(new_population)

    # population = [
    #    individual for individual in population if not is_individual_invalid(individual)
    # ]

    new_population = check_programs(new_population)

    with open("programs.txt", "w") as f:
        for individual in new_population:
            print(individual, file=f)

    new_population = population + list(zip(new_population, s.run(new_population)))

    sorted_population = sorted(new_population, key=lambda x: x[1], reverse=True)

    print(sorted_population[:5])

    # Selection
    population = sorted_population[:300]

    random.shuffle(population)
