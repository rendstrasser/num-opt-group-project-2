"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from shared.constraints import LinearConstraint, LinearCallable
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
                    b: np.ndarray, is_equality_vec: Sequence[bool] = None,
                    bias: float = 0, solution: np.ndarray = None,
                    x0: float = None) -> 'QuadraticProblem':
        is_equality_vec = is_equality_vec if is_equality_vec is not None else np.ones(len(b))
        constraints = [
            LinearConstraint(LinearCallable(a_i, b_i), is_equality)
            for a_i, b_i, is_equality in zip(A, b, is_equality_vec)
        ]
        return cls(n=len(b), G=G, constraints=constraints, c=c, bias=bias,
                   solution=solution, x0=x0)
