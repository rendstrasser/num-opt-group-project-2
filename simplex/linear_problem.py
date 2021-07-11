"""
LinearProblem and relevant functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

import numpy as np

from shared.constraints import EquationType, LinearConstraint, LinearCallable
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

    @classmethod
    def from_positive_constrained_params(cls, 
            c: np.ndarray, 
            n: int, 
            constraints: Sequence[LinearConstraint], 
            solution: np.ndarray = None, 
            x0: np.ndarray = None,
            bias: float = 0):
        """
        Creates a linear problem based on the given params.
        It is assumed that 'constraints' does not explicitly contain positivity constraints, e.g. x_4 >= 0.
        Those are added by this method.
        """
        incl_positivity_constraints = constraints

        for i in range(n):
            e = np.eye(n)[i]

            incl_positivity_constraints = np.append(incl_positivity_constraints,
                                                    LinearConstraint(
                    c=LinearCallable(a=e, b=0),
                    equation_type=EquationType.GE))

        return LinearProblem(n=n, constraints=incl_positivity_constraints, x0=x0, solution=solution, c=c, bias=bias)

    def to_standard_form(self) -> Tuple[LinearProblem, np.ndarray]:
        """
        Converts a problem to standard form (13.41) by
        adapting the objective and constraints as shown in page 357.

        x is split into x+ and x- for all cases where we don't have positivity constraints on x.

        Additionally, we introduce slack variables for all inequality the constraints.
        Greater-than inequalities are multiplied with -1 before slack variables are added.

        The final result does not explicitly contain positivity constraints anymore but is set up in a way that 
        those can be assumed to be part of the theoretical problem.

        :return: problem in standard form wrt to the objective and the equality constraints,
        x>=0 constraints are not explicitly part of the problem
        """
        # number of slack variables
        
        standard_constraints, non_positive_constrained_indices, slack_var_count = super().standardized_constraints()

        # adapt c with new x^- and slack variables
        neg_c = -self.c[non_positive_constrained_indices]
        standard_c = np.concatenate((self.c, neg_c, np.zeros(slack_var_count)))

        return LinearProblem(
            c=standard_c,
            n=len(standard_c),
            constraints=standard_constraints,
            x0=None,
            solution=None), non_positive_constrained_indices

    @classmethod
    def phase_I_problem_from(cls, problem: LinearConstraintsProblem, standardized: bool) -> Tuple[LinearProblem, np.ndarray, int]:
        # standardize constraints (but not entire problem, as not necessary)
        if not standardized:
            standardized_constraints, non_positive_constrained_indices, slack_var_count = problem.standardized_constraints()
        else:
            standardized_constraints = problem.constraints
            non_positive_constrained_indices = []
            slack_var_count = 0

        # n of standardized constraints problem
        n = problem.n + len(non_positive_constrained_indices) + slack_var_count 
        m = len(standardized_constraints)

        e_x = np.zeros(shape=n)
        e_z = np.ones(shape=m)
        e = np.concatenate((e_x, e_z))

        x0 = np.zeros(n)
        z0 = np.abs(problem.b)
        xz0 = np.concatenate((x0, z0))

        constraints = []
        for i, constraint in enumerate(standardized_constraints):
            E_i = np.eye(m)[i]
            if problem.b[i] < 0:
                E_i = -E_i

            A_i = constraint.c.a

            a = np.concatenate((A_i, E_i))

            constraints.append(LinearConstraint(
                c=LinearCallable(a=a, b=constraint.c.b),
                equation_type=constraint.equation_type))

        return LinearProblem(c=e, constraints=constraints, x0=xz0, n=n+m, solution=None), non_positive_constrained_indices, slack_var_count

