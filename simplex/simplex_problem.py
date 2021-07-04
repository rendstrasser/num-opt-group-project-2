"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Sequence, Callable

import numpy as np

from shared.constraints import LinearConstraint, combine_linear
from shared.minimization_problem import MinimizationProblem


@dataclass
class SimplexProblem(MinimizationProblem):
    f: Callable[[np.ndarray], np.ndarray] = field(init=False)
    constraints: Sequence[LinearConstraint]
    c: np.ndarray

    def __post_init__(self):
        self.f = lambda x: np.inner(self.c, x)
        self.A, self.b = combine_linear([constraint.c for constraint in self.constraints])
