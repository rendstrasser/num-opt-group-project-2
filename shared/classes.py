import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class Constraint: 
    c: Callable[[np.ndarray], np.ndarray]
    equality: True # if false, then inequality

@dataclass
class MinimizationProblem:
    """
    Data class containing all necessary information of a minimization problem to support
    steepest descent, newton, quasi-newton and conjugate minimization.

    Args:
        f (Callable): The function (objective) we are trying to minimize.
        constraints (Sequence[Constraint]): Sequence of constraints.
        solutions (Sequence[np.ndarray]): The solutions(s) to the minimization problem.
                                         Might contain multiple if there are multiple local minimizers.
        x0 (np.ndarray): The starting point for the minimization procedure.
    """
    f: Callable[[np.ndarray], np.ndarray]
    constraints: Sequence[Constraint]
    solutions: Sequence[np.ndarray]
    x0: np.ndarray

    # --- Objective function methods ---

    def calc_f_at(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)

    def calc_gradient_at(self, x: np.ndarray) -> np.ndarray:
        return self._central_difference_gradient(self.f, x)

    def calc_hessian_at(self, x: np.ndarray) -> np.ndarray:
        return self.hessian_approximation(self.f, x)

    # --- Constraint methods ---

    def calc_constraints_at(self, x: np.ndarray) -> np.ndarray: 
        return np.array([c.c(x) for c in self.constraints])

    def calc_constraint_at(self, i: int, x: np.ndarray) -> float: 
        return self.constraints[i].c(x)

    def calc_constraints_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._central_difference_gradient(c.c, x) for c in self.constraints])

    def calc_constraint_gradient_at(self, i: int, x: np.ndarray) -> np.ndarray:
        return self._central_difference_gradient(self.constraints[i].c, x)

    # --- Lagrangian methods ---

    def calc_lagrangian_at(self, x, l) -> float:
        assert len(l) == len(self.constraints)

        result = self.calc_f_at(x)

        for i, l_i in enumerate(l):
            result -= l_i * self.calc_constraint_at(i, x)
        
        return result
    
    # gradient wrt x
    def calc_lagrangian_gradient_at(self, x, l) -> np.ndarray:
        def lagrangian(x):
            return self.calc_lagrangian_at(x, l)

        return self._central_difference_gradient(lagrangian, x)

    # hessian wrt x
    def calc_lagrangian_hessian_at(self, x, l) -> np.ndarray:
        def lagrangian(x):
            return self.calc_lagrangian_at(x, l)

        return self.hessian_approximation(lagrangian, x)

    # --- Gradient/hessian approximation implementation ---

    def _central_difference_gradient(self, f: Callable, x: np.ndarray) -> np.ndarray:
        """Approximate gradient as described in equation (8.7), called the 'central difference formula'.

        Args:
            x (np.ndarray): Function input.

        Returns:
            np.ndarray: Approximated gradient.
        """
        eps = self._find_epsilon(x)
        eps_vectors = np.eye(N=len(x)) * eps
        return np.array([
            (f(x + eps_vector) - f(x - eps_vector))/(2*eps) for eps_vector in eps_vectors
        ])

    def hessian_approximation(self, f: Callable, x: np.ndarray) -> np.ndarray:
        """Approximate Hessian based on equation (8.21) in the book.

        Args:
            x (np.ndarray): Point for which we approximate the function's Hessian.

        Returns:
            np.ndarray: Approximated Hessian.
        """
        eps = self._find_epsilon(x)
        eps_vectors = np.eye(N=len(x)) * eps
        
        hess = np.array([
            [self._hess_approx_num(f, x, eps_i, eps_j) for eps_i in eps_vectors]
            for eps_j in eps_vectors
        ]) / (eps**2)
        

        # If the hessian approximation is basically 0, we are already close.
        # Avoids SingularMatrix errors.
        if sum(abs(entry) for row in hess for entry in row) < 0.0001:
            return np.eye(len(x))

        return hess
        
    @staticmethod
    def _hess_approx_num(f: Callable, x: np.ndarray, eps_i: np.ndarray, eps_j: np.ndarray) -> float:
        return f(x + eps_i + eps_j) - f(x + eps_i) - f(x + eps_j) + f(x)

    @staticmethod
    def _find_epsilon(x: np.ndarray):
        """Find computational error of the datatype of x and return it's square-root, as in equation (8.6).

        Args:
            x (np.ndarray): Array of which the datatype is considered.
        """
        try:
            # Given the datatype of x, the below is the least number such that `1.0 + u != 1.0`.
            u = np.finfo(x.dtype).eps

        # x is an exact type, which throws an error; we use float64 instead, 
        # as it is often the default when performing operations on ints which map to floats.
        except (TypeError, ValueError): 
            
            u = np.finfo(np.float64).eps

        epsilon = np.sqrt(u)

        return epsilon