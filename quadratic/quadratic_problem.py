"""
SimplexProblem and relevant functions.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from shared.constraints import LinearConstraint, LinearCallable, EquationType
from shared.minimization_problem import LinearConstraintsProblem
from simplex.base import minimize_linear_problem
from simplex.linear_problem import LinearProblem


@dataclass(repr=False)
class QuadraticProblem(LinearConstraintsProblem):
    """Quadratic problem, such that

    Attributes:
        f (Callable): Is the function as defined in the book,
        c (np.ndarray): is a vector of weights,
        G (np.ndarray): is a positive-definite matrix,
        constraints (Sequence[LinearConstraint]): is a collection of linear constraints, which can contain inequalities
    """
    f: Callable[[np.ndarray], float] = field(init=False)
    G: np.ndarray
    c: np.ndarray

    def __post_init__(self):
        super(QuadraticProblem, self).__post_init__()

        # +1e-5 because of approximation errors
        if np.any(np.linalg.eigvals(self.G) + 1e-5 <= 0):
            raise ValueError("Non-positive-definite matrices not supported.")

        self.f = lambda x: 1 / 2 * x @ self.G @ x + x @ self.c

    @classmethod
    def from_params(cls,
                    G: np.ndarray,
                    c: np.ndarray,
                    A: np.ndarray,
                    b: np.ndarray,
                    equation_type_vec: Sequence[bool] = None,
                    solution: np.ndarray = None,
                    x0: np.ndarray = None) -> 'QuadraticProblem':
        """
        Alternative constructor to return a quadratic problem from different params.

        Args:
            G: Matrix used in the quadratic term of the function.
            c: Vector used in the linear term of the function.
            A: Matrix representing the weights of the different linear constraints.
            b: Vector representing the b_i in a linear constraint.
            equation_type_vec (object): Vector specifying which of the constraints have which equality signs.
            solution: Solution of the problem.
            x0: Starting point.

        Returns:
            QuadraticProblem specified by aforementioned parameters.
        """
        equation_type_vec = equation_type_vec if equation_type_vec is not None else np.repeat(EquationType.EQ, len(b))
        constraints = [
            LinearConstraint(LinearCallable(a_i, b_i), equation_type)
            for a_i, b_i, equation_type in zip(A, b, equation_type_vec)
        ]
        return cls(n=len(c), G=G, constraints=constraints, c=c,
                   solution=solution, x0=x0)

    @property
    def is_inequality_constrained(self) -> bool:
        """Return whether or not any of the imposed constraints are inequalities."""
        return any(constraint.equation_type is not EquationType.EQ
                   for constraint in self.constraints)

    def find_x0(self, original_initial_guess: np.ndarray = None) -> np.ndarray:
        """Find initial solution

        Args:
            original_initial_guess: Point to start from.

        Returns:
            np.ndarray - Feasible point.
        """

        if self.x0 is not None:
            return self.x0

        standardized_constraints, standardizing_meta_info = self.standardized_constraints()
        standardized_n = standardizing_meta_info.calc_standardized_n()

        initial_guess = np.zeros(standardized_n)
        if original_initial_guess is not None:
            initial_guess[:self.n] = original_initial_guess

        z = np.array([self._compute_z_i(x=initial_guess, constraint=constraint)
                      for constraint in standardized_constraints])

        e = np.concatenate([np.zeros(len(initial_guess)), np.ones(len(z))])

        gamma = np.array([self._compute_gamma_i(x=initial_guess, constraint=constraint)
                          for constraint in standardized_constraints])

        # We bring this into a sort of standard form by having the gammas
        # being unit-vectors, quasi.
        sub_constraints = [LinearConstraint(c=LinearCallable(
            a=np.concatenate([constraint.c.a, unit_vector * gamma_i]),
            b=constraint.c.b),
            equation_type=constraint.equation_type
        )
            for constraint, unit_vector, gamma_i
            in zip(standardized_constraints, np.eye(len(standardized_constraints)), gamma)]

        x0 = np.concatenate([initial_guess, z])

        sub_problem = LinearProblem(constraints=sub_constraints,
                                    x0=x0,
                                    solution=None,
                                    c=e, n=len(x0))

        solution, _ = minimize_linear_problem(sub_problem, standardized=True)
        n = standardizing_meta_info.calc_standardized_n()
        x0 = solution[:n]

        return standardizing_meta_info.destandardize_x(x0)

    @staticmethod
    def _compute_gamma_i(x: np.ndarray, constraint: LinearConstraint) -> float:
        """Compute entry of gamma for the warm start method on page 473.

        Args:
            x: Point, usually being the initial guess.
            constraint: Some constraint.
        """
        a, b = constraint.c.a, constraint.c.b
        if constraint.equation_type == EquationType.EQ:
            sign = -np.sign(np.inner(a, x) - b)
            if sign == 0:
                # hardcode to 1, also fine and consistent with basic phase I problem
                sign = 1

            return sign
        else:
            return 1

    @staticmethod
    def _compute_z_i(x: np.ndarray, constraint: LinearConstraint) -> float:
        """Compute entry of z for the warm start method on page 473.

        Args:
            x: Point, usually being the initial guess.
            constraint: Some constraint.
        """
        a, b = constraint.c.a, constraint.c.b
        if constraint.equation_type == EquationType.EQ:
            return np.abs(np.inner(a, x) - b)
        else:
            return max(b - np.inner(a, x), 0)
