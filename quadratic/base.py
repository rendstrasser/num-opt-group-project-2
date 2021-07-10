from quadratic.quadratic_problem import QuadraticProblem

import numpy as np


def minimize_quadratic_problem(problem: QuadraticProblem) -> np.ndarray:
    """
    A problem - assumed to be in standard form - is optimized.
    """

    x = find_x0(problem)
    m = len(problem.b)  # number of constraints -> size of basis

    # the basis consists of x0 elements which are not 0. all others should be 0
    x0_args_sorted = np.argsort(x)
    basis = np.argsort(x)[-m:]

    # TODO implement algorithm 13.1
    while True:
        break

    return x


def find_x0(problem: QuadraticProblem) -> np.ndarray:
    raise NotImplementedError
