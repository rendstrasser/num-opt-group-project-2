"""
MinimizationProblem, relevant functions and implemented methods.
"""

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

from shared.constraints import Constraint, LinearConstraint
from shared.gradient_approximation import gradient_approximation, hessian_approximation

@dataclass
class MinimizationProblem:
    """
    Data class containing all necessary information of a minimization problem to support
    unconstrained optimization.

    Args:
        f (Callable): The function (objective) we are trying to minimize.
        n (int): Dimensionality of input x for function
        constraints (Sequence[Constraint]): Sequence of constraints.
        solution (np.ndarray): The solution to the minimization problem. May be None if unknown.
        x0 (np.ndarray): The starting point for the minimization procedure. May be None if unknown.
    """
    f: Callable[[np.ndarray], float]
    n: int
    constraints: Sequence[Constraint]
    x0: np.ndarray
    solution: np.ndarray

    # --- Objective function methods ---

    def active_set_at(self, x: np.ndarray) -> List[Constraint]:
        """Return list of active constraints at point x.

        Returns:
            List[Constraint]: List of active constraints.
        """
        return [c for c in self.constraints if c.is_active(x)]

    def calc_f_at(self, x: np.ndarray) -> float:
        return self.f(x)

    def calc_gradient_at(self, x: np.ndarray) -> np.ndarray:
        return self._central_difference_gradient(self.f, x)

    def calc_hessian_at(self, x: np.ndarray) -> np.ndarray:
        return self.hessian_approximation(self.f, x)

    # --- Constraint methods ---

    def calc_constraints_at(self, x: np.ndarray) -> np.ndarray:
        return np.array([c(x) for c in self.constraints])

    def calc_constraint_at(self, i: int, x: np.ndarray) -> float:
        return self.constraints[i](x)

    def calc_constraints_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        return np.array([gradient_approximation(c.c, x) for c in self.constraints])

    def calc_constraint_gradient_at(self, i: int, x: np.ndarray) -> np.ndarray:
        return gradient_approximation(self.constraints[i], x)

    # --- Lagrangian methods ---

    def calc_lagrangian_at(self, x, lambda_) -> float:
        assert len(lambda_) == len(self.constraints)

        result = self.calc_f_at(x)

        for i, lambda_i in enumerate(lambda_):
            result -= lambda_i * self.calc_constraint_at(i, x)

        return result

    # gradient wrt x
    def calc_lagrangian_gradient_at(self, x, lambda_) -> np.ndarray:

        def lagrangian(x_):
            return self.calc_lagrangian_at(x_, lambda_)

        return gradient_approximation(lagrangian, x)

    # hessian wrt x
    def calc_lagrangian_hessian_at(self, x, lambda_) -> np.ndarray:

        def lagrangian(x_):
            return self.calc_lagrangian_at(x_, lambda_)

        return hessian_approximation(lagrangian, x)

@dataclass
class LinearConstraintsProblem(MinimizationProblem):
    """
    Data class containing all necessary information of a minimization problem to support
    unconstrained optimization.

    Holds linar constraints in the form of Ax=b
    """
    constraints: Sequence[LinearConstraint]