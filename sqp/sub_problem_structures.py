import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence

from quadratic.quadratic_problem import QuadraticProblem
from shared.constraints import LinearConstraint, Constraint, LinearCallable


@dataclass
class SqpIterateQuadraticProblem(QuadraticProblem):
    """
    Represents the quadratic sub-problem for solving an iteration of SQP (18.11).

    Args:
        G: Lagrange function Hessian or approximation of Hessian at x
        c: Derivative of function value at current iterate x
        original_constraints: Constraint instances from original problem
        original_constraint_values: Values of all constraints c_i(x) at current iterate x
        original_constraints_jacobian: Jacobian of constraints at current iterate x
        initial_guess: Point assumed to be close to a feasible region.
"""

    n: int = field(init=False)
    constraints: Sequence[LinearConstraint] = field(init=False)
    x0: Optional[np.ndarray] = field(init=False)
    solution: Optional[np.ndarray] = field(init=False)

    original_constraints: Sequence[Constraint]
    original_constraint_values: np.ndarray
    original_constraint_jacobian: np.ndarray

    initial_guess: np.ndarray = None

    def __post_init__(self):
        self.n = len(self.c)

        self.constraints = np.array([self.build_constraint(constraint, c_i, c_i_grad)
                                     for constraint, c_i, c_i_grad
                                     in zip(
                                        self.original_constraints,
                                        self.original_constraint_values,
                                        self.original_constraint_jacobian)])

        self.x0 = None

        try:
            self.x0 = self.find_x0(self.initial_guess)
        except np.linalg.LinAlgError:
            pass

        self.solution = None

        super(QuadraticProblem, self).__post_init__()

    @staticmethod
    def build_constraint(
            original_constraint: Constraint,
            original_constraint_value: float,
            original_constraint_grad: np.ndarray) -> LinearConstraint:
        """
        Creates a linear constraint for a quadratic sub-problem based on a non-linear constraint (see 18.11).

        Args:
            original_constraint: Original non-linear constraint
            original_constraint_value: Value of constraint at x
            original_constraint_grad: Value of derivative of constraint at x

        Returns:
            Linear constraint for quadratic sub-problem
        """

        return LinearConstraint(
            c=LinearCallable(a=original_constraint_grad, b=-original_constraint_value),
            equation_type=original_constraint.equation_type)
