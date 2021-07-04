"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from shared.constraints import SimplexConstraint
from shared.minimization_problem import MinimizationProblem


@dataclass
class SimplexCallable:
    """Specific callable to SimplexProblem. Keeps A and b accessible."""
    A: np.ndarray
    b: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x + self.b


@dataclass
class SimplexProblem(MinimizationProblem):
    constraints: Sequence[SimplexConstraint]

    # Methods
