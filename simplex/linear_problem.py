"""
LinearProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Sequence, Callable

import numpy as np

from shared.constraints import LinearConstraint, combine_linear
from shared.minimization_problem import LinearConstraintsProblem


@dataclass
class LinearProblem(LinearConstraintsProblem):
    """
    Linear problem as described in the book at 13.1.
    Constraints don't explicitly contain the x >= 0 case.
    """
    f: Callable[[np.ndarray], float] = field(init=False)
    c: np.ndarray
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    bias: float = 0

    def __post_init__(self):
        self.f = lambda x: self.c @ x + self.bias
        self.A, self.b = combine_linear([constraint.c for constraint in self.constraints])
