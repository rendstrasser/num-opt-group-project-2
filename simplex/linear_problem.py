"""
LinearProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Sequence, Callable

import numpy as np

from shared.constraints import LinearConstraint, combine_linear
from shared.minimization_problem import MinimizationProblem


@dataclass
class LinearProblem(MinimizationProblem):
    f: Callable[[np.ndarray], float] = field(init=False)
    c: np.ndarray
    constraints: Sequence[LinearConstraint]
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    bias: float = 0

    def __post_init__(self):
        self.f = lambda x: self.c @ x + self.bias
        self.A, self.b = combine_linear([constraint.c for constraint in self.constraints])
