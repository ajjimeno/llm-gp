""" This module contains the implementation of the genetic programming algorithm """

""" which are based on the DEAP library to improve the speed and memory use """
""" For more information about DEAP: https://deap.readthedocs.io/en/master/ """
import copy
import pickle
import random
import re
import sys
from array import array
from collections import deque
from inspect import isclass

from deap import tools
from deap.gp import Primitive, Terminal


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or (
            depth >= min_ and random.random() < pset.terminalRatio
        )
        # (depth >= min_ and depth <= max_)
        # AJY

    return generate(pset, min_, max_, condition, type_)


def generate(pset, min_, max_, condition, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        # AJY
        # if condition(height, depth):
        if condition(height, depth) and len(pset.terminals[type_]) > 0:
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                print(pset.terminals)
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add "
                    "a terminal of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add "
                    "a primitive of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def varAndCrossOver(offspring, toolbox, cxpb, start_index, end_index):
    for i in range(start_index, end_index, 2):
        if i >= len(offspring):
            break
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i]
            )
            del offspring[i - 1].fitness.values, offspring[i].fitness.values


def varAndMutation(offspring, toolbox, mutpb, start_index, end_index):
    """Part of an evolutionary algorithm applying only the variation part"""
    for i in range(start_index, end_index):
        if i > len(offspring):
            break
        if random.random() < mutpb:
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values


import threading


def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    n_offspring = len(offspring)
    n_threads = min(n_offspring, 20)
    chunk_size = int(n_offspring / n_threads)

    threads = []

    # Apply crossover and mutation on the offspring
    for i in range(n_threads):
        start_index = i * chunk_size + 1
        end_index = n_offspring if (i == n_threads - 1) else (i + 1) * chunk_size

        t = threading.Thread(
            target=varAndCrossOver,
            args=(offspring, toolbox, cxpb, start_index, end_index),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    threads = []

    for i in range(n_threads):
        start_index = i * chunk_size
        end_index = n_offspring if (i == n_threads - 1) else (i + 1) * chunk_size

        t = threading.Thread(
            target=varAndMutation,
            args=(offspring, toolbox, mutpb, start_index, end_index),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    """
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    """
    return offspring


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    cp=None,
    checkpoint=None,
    verbose=__debug__,
):

    if cp:
        # A file name has been given, then load the data from the file
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        start_gen = 1
        logbook = tools.Logbook()

    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                    contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
            evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
    Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    if halloffame is not None and len(halloffame) > 0:
        print("Removing from hall of fame")
        population.append(halloffame[0])
        while len(halloffame) > 0:
            print("Removing from hall of fame")
            halloffame.remove(-1)
        print(len(halloffame))

    print("Initial evaluation")
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population]  # if not ind.fitness.valid]
    print("Evaluation")
    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    fitnesses = toolbox.evaluate(invalid_ind)

    print("Merging")
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("halloffame update")
    if halloffame is not None:
        halloffame.update(population)

    print("logbook")
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    k = len(population)

    print("Begin generation")

    lp = 25

    # Begin the generational process
    for gen in range(start_gen, ngen + 1):

        l = len(halloffame[0])
        np = [p for p in population if len(p) <= (l + lp)]
        # np = population

        print(len(np))
        print("Select")
        # Select the next generation individuals
        offspring = toolbox.select(np, k)
        # offspring = toolbox.select(np, min(k, len(population)))
        # offspring = toolbox.select(population, len(population))

        # print("X-over and mutation")
        # Vary the pool of individuals
        # offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        best_inds = []

        for i in range(len(halloffame)):
            best_inds.append(toolbox.clone(halloffame[i]))
            # offspring = [toolbox.clone(halloffame[i]), *offspring]

        print("X-over and mutation")
        # print(offspring[0])
        # print(halloffame[i])
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        offspring.append(halloffame[0])
        halloffame.remove(-1)

        print("Preparing Evaluation")
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print("Evaluation")

        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        # strings = toolbox.map(str, invalid_ind)
        fitnesses = toolbox.evaluate(invalid_ind)
        # fitnesses = [toolbox.evaluate.remote(ind) for ind in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        for ind in best_inds:
            offspring.append(ind)

        # Replace the current population by the offspring
        population[:] = offspring

        print("Statistics")
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # pr.disable()
        # pr.print_stats()
        # sys.exit(-1)

        # AJ
        [print(halloffame[i], flush=True) for i in range(len(halloffame))]

        # Evaluate
        [
            print(
                "Exp|gen|{}|length|{}|height|{}|accTraining|{}|accTesting|{}".format(
                    gen,
                    len(halloffame[i]),
                    halloffame[i].height,
                    halloffame[i].fitness.values[0],
                    toolbox.evalTest(halloffame[i]),
                ),
                flush=True,
            )
            for i in range(len(halloffame))
        ]
        # print(toolbox.evalTest(halloffame[0]))

        print("Checkpoint")
        # AJ: Write after each iteration
        if checkpoint:
            cp = dict(
                population=population,
                generation=gen,
                halloffame=halloffame,
                logbook=logbook,
                rndstate=random.getstate(),
            )

            with open(checkpoint, "wb") as cp_file:
                pickle.dump(cp, cp_file)

    return population, logbook


class PrimitiveTree(array):
    """Tree specifically formatted for optimization of genetic
    programming operations. The tree is represented with a
    list where the nodes are appended in a depth-first order.
    The nodes appended to the tree are required to
    have an attribute *arity* which defines the arity of the
    primitive. An arity of 0 is expected from terminals nodes.
    """

    object_ids = {}
    ids_objects = {}

    def map_object(o):
        try:
            return PrimitiveTree.object_ids[o.name]
        except:
            PrimitiveTree.object_ids[o.name] = len(PrimitiveTree.object_ids)
            PrimitiveTree.ids_objects[PrimitiveTree.object_ids[o.name]] = o

            return PrimitiveTree.object_ids[o.name]

    def __new__(cls, content):
        if isinstance(content, PrimitiveTree):
            return super(PrimitiveTree, cls).__new__(cls, "b", [c for c in content])
        else:
            return super(PrimitiveTree, cls).__new__(
                cls, "b", [cls.map_object(p) for p in content]
            )

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [PrimitiveTree.ids_objects[i] for i in array.__getitem__(self, key)]
        else:
            return PrimitiveTree.ids_objects[array.__getitem__(self, key)]

    def __setitem__(self, key, val):
        # Check for most common errors
        # Does NOT check for STGP constraints
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError(
                    "Invalid slice object (try to assign a %s"
                    " in a tree of size %d). Even if this is allowed by the"
                    " list object slice setter, this should not be done in"
                    " the PrimitiveTree context, as this may lead to an"
                    " unpredictable behavior for searchSubtree or evaluate."
                    % (key, len(self))
                )
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError(
                    "Invalid slice assignation : insertion of"
                    " an incomplete subtree is not allowed in PrimitiveTree."
                    " A tree is defined as incomplete when some nodes cannot"
                    " be mapped to any position in the tree, considering the"
                    " primitives' arity. For instance, the tree [sub, 4, 5,"
                    " 6] is incomplete if the arity of sub is 2, because it"
                    " would produce an orphan node (the 6)."
                )
        elif val.arity != self[key].arity:
            raise ValueError(
                "Invalid node replacement with a node of a" " different arity."
            )

        if isinstance(key, slice):
            array.__setitem__(
                self, key, array("b", [PrimitiveTree.map_object(p) for p in val])
            )
        else:
            array.__setitem__(self, key, PrimitiveTree.map_object(val))

    def __str__(self):
        """Return the expression in a human readable string."""
        string = ""
        stack = []
        for node in self:
            node = PrimitiveTree.ids_objects[node]
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)

        return string

    @classmethod
    def from_string(cls, string, pset):
        """Try to convert a string expression into a PrimitiveTree given a
        PrimitiveSet *pset*. The primitive set needs to contain every primitive
        present in the expression.

        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == "":
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if token in pset.mapping:
                primitive = pset.mapping[token]

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError(
                        "Primitive {} return type {} does not "
                        "match the expected one: {}.".format(
                            primitive, primitive.ret, type_
                        )
                    )

                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if type_ is None:
                    type_ = type(token)

                if not issubclass(type(token), type_):
                    raise TypeError(
                        "Terminal {} type {} does not "
                        "match the expected one: {}.".format(token, type(token), type_)
                    )

                expr.append(Terminal(token, False, type_))
        return cls(expr)

    @property
    def height(self):
        """Return the height of the tree, or the depth of the
        deepest node.
        """
        stack = [0]
        max_depth = 0
        for elem in self:
            elem = PrimitiveTree.ids_objects[elem]
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth

    @property
    def root(self):
        """Root of the tree, the element 0 of the list."""
        return self[0]

    def searchSubtree(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)


if __name__ == "__main__":
    p = Primitive("mul", (int, int), int)
    a = PrimitiveTree([p])

    print(a[0].name)

    a[0] = p

    print(a[0].name)
    print(array.__dict__)
