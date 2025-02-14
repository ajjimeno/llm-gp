import math
import random

import numpy as np


def selStochasticUniversalSampling(individuals, k):
    """Select the *k* individuals among the input *individuals*.
    The selection is made by using a single random value to sample all of the
    individuals by choosing them at evenly spaced intervals. The list returned
    contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :return: A list of selected individuals.

    This function is an optimised version from the code provided by DEAP

    This function uses the :func:`~random.uniform` function from the python base
    :mod:`random` module.
    """
    fits = [math.exp(ind[1] * 50) - 1 for ind in individuals]

    sum_fits = sum(fits)

    distance = sum_fits / float(k)
    start = random.uniform(0, distance)

    print(
        f"max: {np.max(fits)} min: {np.min(fits)} sum: {sum_fits} distance: {distance} start: {start}"
    )

    points = [start + i * distance for i in range(k)]

    chosen = []
    i = 0

    sum_ = fits[i]
    for p in points:
        while sum_ < p:
            i += 1
            sum_ += fits[i]

        chosen.append(individuals[i])

    return chosen


if __name__ == "__main__":
    individuals = [("p", 0.9), ("a", 0.1)]
    print(selStochasticUniversalSampling(individuals, k=300))
