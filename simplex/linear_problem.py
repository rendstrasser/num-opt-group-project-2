"""
LinearProblem and relevant functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from shared.constraints import LinearConstraint, LinearCallable
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
        super(LinearProblem, self).__post_init__()
        self.f = lambda x: self.c @ x + self.bias

    def to_standard_form(self, x_positive_constraints_assumed=True) -> LinearProblem:
        """
        Assumes that the given problem is given without x>=0 constraints on the input
        and converts it to standard form (13.41) where we assume these constraints, by
        adapting the objective and constraints as shown in page 357.

        Additionally, we just introduce slack variables for all the constraints, even
        if they have been equality constraints before already - it shouldn't matter as z=0
        will still be a valid solution for these.

        :return: problem in standard form wrt to the objective and the equality constraints,
        x>=0 constraints are not explicitly part of the problem
        """
        m = len(self.constraints)

        standard_constraints = []

        for i, constraint in enumerate(self.constraints):
            a = constraint.c.a
            e = np.eye(m)[i]

            # bring to standard form (13.41) by assuming x+, x-, z,
            # as shown in page 357
            new_a = np.concatenate((a, -a, e))
            if x_positive_constraints_assumed:
                # assume x >= 0 was already part before
                new_a = np.concatenate((a, e))
            
            standard_constraints.append(LinearConstraint(
                c=LinearCallable(a=new_a, b=constraint.c.b),
                is_equality=True))

        standard_c = np.concatenate((self.c, -self.c, np.zeros(m)))

        if x_positive_constraints_assumed:
            # assume x >= 0 was already part before
            standard_c = np.concatenate((self.c, np.zeros(m)))

        return LinearProblem(
            c=standard_c,
            n=len(standard_c),
            constraints=standard_constraints,
            x0=None,
            solution=None)

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
            a = np.concatenate((A_i, E_i))

            constraints.append(LinearConstraint(
                c=LinearCallable(a=a, b=constraint.c.b),
                is_equality=constraint.is_equality))

        return LinearProblem(c=e, constraints=constraints, x0=xz0, n=n+m, solution=None)

