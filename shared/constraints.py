"""
File containing Implementations of basic constraints and their methods.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from simplex.simplex_problem import SimplexCallable


@dataclass
class Constraint:
    c: Callable[[np.ndarray], np.ndarray]
    is_equality: bool

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.c(x)

    def holds(self, x: np.ndarray) -> bool:
        return self(x) == 0 if self.is_equality else self(x) >= 0

    def is_active(self, x: np.ndarray) -> bool:
        """Check whether the constraint is active at point x, i.e. if c(x) == 0.

        Args:
            x: Point to evaluate the constraint at.
        """
        return self(x) == 0


@dataclass
class LinearCallable:
    """Specific callable to SimplexProblem. Keeps A and b accessible."""
    a: np.ndarray
    b: float

    def __call__(self, x: np.ndarray) -> float:
        return self.a @ x - self.b

@dataclass
class LinearConstraint(Constraint):
    c: LinearCallable


def combine_linear(linear_callables: Sequence[LinearCallable]) -> Tuple[np.ndarray, np.ndarray]:
    """Combine attributes of linear callables into matrix A and vector b.

    Args:
        linear_callables (Sequence[LinearCallable]): List of linear callables.
