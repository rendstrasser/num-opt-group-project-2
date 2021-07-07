"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from shared.minimization_problem import LinearConstraintsProblem


@dataclass
class QuadraticProblem(LinearConstraintsProblem):
    f: Callable[[np.ndarray], float] = field(init=False)
    G: np.ndarray
    c: np.ndarray
    bias: float = 0
    
    def __post_init__(self):
        super(QuadraticProblem, self).__post_init__()
        self.f = lambda x: 1/2 * x @ self.G @ x + x @ self.c + self.bias
