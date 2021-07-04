"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from shared.constraints import combine_linear
from shared.minimization_problem import LinearConstraintsProblem


@dataclass
class QuadraticProblem(LinearConstraintsProblem):
    f: Callable[[np.ndarray], float] = field(init=False)
    G: np.ndarray
    c: np.ndarray
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    bias: float = 0
    
    def __post_init__(self):
        self.f = lambda x: 1/2 * x @ self.G @ x + x @ self.c + self.bias
        self.A, self.b = combine_linear([constraint.c for constraint in self.constraints])
