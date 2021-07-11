"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from shared.constraints import LinearConstraint, LinearCallable, InequalitySign
from shared.minimization_problem import LinearConstraintsProblem


@dataclass(repr=False)
class QuadraticProblem(LinearConstraintsProblem):
    """Quadratic problem, such that

    Attributes:
        f (Callable): Is the function as defined in the book,
        c (np.ndarray): is a vector of weights,
        G (np.ndarray): is a non-indefinite matrix,
        constraints (Sequence[LinearConstraint]): is a collection of linear constraints, which can contain inequalities,
        bias (float): is an offset/intercept/bias which translates the entire function up or down.
    """
    f: Callable[[np.ndarray], float] = field(init=False)
    G: np.ndarray
    c: np.ndarray
    bias: float = 0

    def __post_init__(self):
        super(QuadraticProblem, self).__post_init__()

        if {-1, 1}.issubset(np.linalg.eigvals(self.G)):
            raise ValueError("Indefinite Matrices are not supported.")

        self.f = lambda x: 1 / 2 * x @ self.G @ x + x @ self.c + self.bias

    @classmethod
    def from_params(cls, G: np.ndarray, c: np.ndarray, A: np.ndarray,
                    b: np.ndarray, equality_sign_vec: Sequence[bool] = None,
                    bias: float = 0, solution: np.ndarray = None,
                    x0: float = None) -> 'QuadraticProblem':
        """
        Alternative constructor to return a quadratic problem from different params.

        Args:
            G: Matrix used in the quadratic term of the function.
            c: Vector used in the linear term of the function.
            A: Matrix representing the weights of the different linear constraints.
            b: Vector representing the b_i in a linear constraint.
            equality_sign_vec: Vector specifying which of the constraints have which equality signs
            bias: Bias of the function.
            solution: Solution of the problem.
            x0: Starting point.

        Returns:
            QuadraticProblem specified by aforementioned parameters.
        """
        equality_sign_vec = equality_sign_vec if equality_sign_vec is not None else np.repeat(InequalitySign.EQUAL, len(b))
        constraints = [
            LinearConstraint(LinearCallable(a_i, b_i), equality_sign)
            for a_i, b_i, equality_sign in zip(A, b, equality_sign_vec)
        ]
        return cls(n=len(c), G=G, constraints=constraints, c=c, bias=bias,
                   solution=solution, x0=x0)

    @property
    def is_inequality_constrained(self):
        """Return whether or not any of the imposed constraints are inequalities."""
        return any(constraint.equality_type is not InequalitySign.EQUAL for constraint in self.constraints)
