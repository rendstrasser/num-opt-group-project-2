"""
File containing Implementations of basic constraints and their methods.
"""

from copy import copy
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Optional
from enum import Enum
from operator import __le__, __ge__, __eq__


import numpy as np


class EquationType(Enum):
    """Defines which operator/sign is in the middle of the equation."""
    EQ = __eq__
    LE = __le__
    GE = __ge__


@dataclass
class Constraint:
    c: Callable[[np.ndarray], float]
    equation_type: EquationType

    def __call__(self, x: np.ndarray) -> float:
        return self.c(x)

    def holds(self, x: np.ndarray) -> bool:
        return self.equation_type.value(self.c(x), 0)

    def is_active(self, x: np.ndarray) -> bool:
        """Check whether the constraint is active at point x, i.e. if c(x) == 0.

        Args:
            x: Point to evaluate the constraint at.
        """
        return self(x) == 0

    def try_get_positivity_constraint_idx(self) -> bool:
        """
        If this constraints represents a positivity constraint, e.g.,
        x_4 >= 0,
        then this method will return the index of the input vector which is positivity-constrained,
        which would be 4 in the example above.

        Returns -1 if not a positivity constraint.
        """
        if self.equality_type != InequalitySign.GREATER_THAN_OR_EQUAL:
            return -1

        non_zero_elems = np.nonzero(self.c.a)
        if len(non_zero_elems) != 1:
            return -1
        
        if self.c.a[non_zero_elems[0]] == 1:
            return non_zero_elems[0]

    def as_equality(self) -> 'LinearConstraint':
        """Return copy of the constraint, such that it is an equality."""
        new_constraint = copy(self)
        new_constraint.equation_type = EquationType.EQ
        return new_constraint


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

    def positivity_constraint_idx(self) -> Optional[int]:
        """
        If this constraints represents a positivity constraint, e.g.,
        x_4 >= 0,
        then this method will return the index of the input vector which is positivity-constrained,
        which would be 4 in the example above.

        Returns None if not a positivity constraint.
        """
        nonzero_indices = np.nonzero(self.c.a)
        if (self.equation_type == EquationType.GE
            and
                len(nonzero_indices) == 1
            and
                self.c.a[nonzero_indices[0]] == 1):
            return nonzero_indices[0]


def combine_linear(linear_callables: Sequence[LinearCallable]) -> Tuple[np.ndarray, np.ndarray]:
    """Combine attributes of linear callables into matrix A and vector b.

    Args:
        linear_callables (Sequence[LinearCallable]): List of linear callables.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A and b.
    """
    A = np.array([c.a for c in linear_callables], dtype=np.float64)
    b = np.array([c.b for c in linear_callables], dtype=np.float64)
    return A, b
