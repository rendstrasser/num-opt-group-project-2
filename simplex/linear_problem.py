"""
LinearProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Sequence, Callable

import numpy as np

from shared.constraints import LinearConstraint, combine_linear, LinearCallable
from shared.minimization_problem import LinearConstraintsProblem


@dataclass
class LinearProblem(LinearConstraintsProblem):
    """
    Linear problem as described in the book at 13.1.
    Constraints don't explicitly contain the x >= 0 case.
    """
    f: Callable[[np.ndarray], float] = field(init=False)
    c: np.ndarray
    bias: float = 0

    def __post_init__(self):
        self.f = lambda x: self.c @ x + self.bias

    @classmethod
    def phase_I_problem_from(cls, problem: LinearConstraintsProblem):
        n = problem.n
        m = len(problem.constraints)

        e_x = np.zeros(shape=n)
        e_z = np.ones(shape=m)
        e = np.concatenate((e_x, e_z))

        x0 = np.zeros(n)
        z0 = np.abs(problem.b)
        xz0 = np.concatenate((x0, z0))

        constraints = []
        for i, constraint in enumerate(problem.constraints):
            E_i = np.eye(m)[i]
            if problem.b[i] < 0:
                E_i = -E_i

            A_i = constraint.c.a
            a = np.concatenate(A_i, E_i)

            constraints.append(LinearConstraint(
                c=LinearCallable(a=a, b=constraint.c.b),
                is_equality=constraint.is_equality))

        return cls(c=e, constraints=constraints, x0=xz0, n=n+m, solution=None)

