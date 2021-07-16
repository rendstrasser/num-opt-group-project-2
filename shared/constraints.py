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

        Returns:
            Optional[int]: The index of the variable which has the positivity constraint or None if not a positivity constraint.
        """
        if self.c.b != 0:
            # only for b=0 we can have a positivity constraint
            return None

        if self.equation_type == EquationType.EQ:
            # only inequality constraints can represent a positivity constraint
            return None

        nonzero_indices = np.nonzero(self.c.a)
        expected_non_zero_elem = 1 if self.equation_type == EquationType.GE else -1

        if len(nonzero_indices) == 1 and self.c.a[nonzero_indices[0]] == expected_non_zero_elem:
            return nonzero_indices[0]

    def equal_callables(self, other: 'Constraint') -> bool:
        return np.all(self.c.a == other.c.a) and self.c.b == other.c.b

    def as_ge_if_le(self) -> 'LinearConstraint':
        """Return copy of the constraint, such that it is an equality."""
        if self.equation_type != EquationType.LE:
            return self

        return LinearConstraint(LinearCallable(a=-self.c.a, b=-self.c.b), equation_type=EquationType.GE)


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
