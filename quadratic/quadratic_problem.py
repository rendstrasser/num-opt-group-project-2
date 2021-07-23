"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from shared.constraints import LinearConstraint, LinearCallable, EquationType
from shared.minimization_problem import LinearConstraintsProblem


@dataclass(repr=False)
class QuadraticProblem(LinearConstraintsProblem):
    """Quadratic problem, such that

    Attributes:
        f (Callable): Is the function as defined in the book,
        c (np.ndarray): is a vector of weights,
        G (np.ndarray): is a non-indefinite matrix,
        constraints (Sequence[LinearConstraint]): is a collection of linear constraints, which can contain inequalities
    """
    f: Callable[[np.ndarray], float] = field(init=False)
    G: np.ndarray
    c: np.ndarray

    def __post_init__(self):
        super(QuadraticProblem, self).__post_init__()

        if np.any(np.linalg.eigvals(self.G) + 1e-5 < 0):
            raise ValueError("Non-positive-definite matrices not supported.")

        self.f = lambda x: 1 / 2 * x @ self.G @ x + x @ self.c

    @classmethod
    def from_params(cls, G: np.ndarray, c: np.ndarray, A: np.ndarray,
                    b: np.ndarray, equation_type_vec: Sequence[bool] = None,
                    solution: np.ndarray = None,
                    x0: float = None) -> 'QuadraticProblem':
        """
        Alternative constructor to return a quadratic problem from different params.

        Args:
            G: Matrix used in the quadratic term of the function.
            c: Vector used in the linear term of the function.
            A: Matrix representing the weights of the different linear constraints.
            b: Vector representing the b_i in a linear constraint.
            equation_type_vec (object): Vector specifying which of the constraints have which equality signs.
            solution: Solution of the problem.
            x0: Starting point.

        Returns:
            QuadraticProblem specified by aforementioned parameters.
        """
        equation_type_vec = equation_type_vec if equation_type_vec is not None else np.repeat(EquationType.EQ, len(b))
        constraints = [
            LinearConstraint(LinearCallable(a_i, b_i), equation_type)
            for a_i, b_i, equation_type in zip(A, b, equation_type_vec)
        ]
        return cls(n=len(c), G=G, constraints=constraints, c=c,
                   solution=solution, x0=x0)

    @property
    def is_inequality_constrained(self):
        """Return whether or not any of the imposed constraints are inequalities."""
        return any(constraint.equation_type is not EquationType.EQ for constraint in self.constraints)
