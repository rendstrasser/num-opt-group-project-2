"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Sequence, Callable

import numpy as np

from shared.constraints import LinearConstraint, combine_linear
from shared.minimization_problem import MinimizationProblem


@dataclass
class QuadraticProblem(MinimizationProblem):
    f: Callable[[np.ndarray], float] = field(init=False)
    G: np.ndarray
    c: np.ndarray
    constraints: Sequence[LinearConstraint]
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.f = lambda x: 1/2 * x @ self.G @ x + x @ self.c
        self.A, self.b = combine_linear([constraint.c for constraint in self.constraints])
