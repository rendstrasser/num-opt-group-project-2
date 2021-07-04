"""
File containing Implementations of basic constraints and their methods.
"""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


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
    """Specific callable to keep a and b accessible."""
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

    Returns:
        Tuple[np.ndarray, np.ndarray]: A and b.
    """
    A = np.array([c.a for c in linear_callables])
    b = np.array([c.b for c in linear_callables])
    return A, b