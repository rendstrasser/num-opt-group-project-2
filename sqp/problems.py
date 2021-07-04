"""
File with functions and routines to construct MinimizationProblems.
"""

import numpy as np

from shared.constraints import Constraint
from shared.minimization_problem import MinimizationProblem


def create_example_18_3_problem():
    """
    Example 18.3 from the book
    """

    def f(x):
        return np.e ** (np.prod(x)) - 1 / 2 * (x[0] ** 3 + x[1] ** 3 + 1) ** 2

    def c1(x):
        return np.sum(x ** 2) - 10

    def c2(x):
        return x[1] * x[2] - 5 * x[3] * x[4]

    def c3(x):
        return x[0] ** 3 + x[1] ** 3 + 1

    x0 = np.array((-1.71, 1.59, 1.82, -0.763, -0.763))
    solution = np.array((-1.8, 1.7, 1.9, -0.8, -0.8))
    constraints = np.array((Constraint(c1, is_equality=True),
                            Constraint(c2, is_equality=True),
                            Constraint(c3, is_equality=True)))

    return MinimizationProblem(f=f, constraints=constraints, solution=solution, x0=x0)
