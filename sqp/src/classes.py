import numpy as np
from typing import Callable
from dataclasses import dataclass

@dataclass
class MinimizationProblemSettings:
    """
    Data class containing settings to activate/deactivate newly
    implemented features for project phase 2.

    Args:
        gradient_approximation_enabled (bool): Enables approximation of gradients as described in equation (8.7) in the book.
        hessian_approximation_enabled (bool): Enables approximation of hessians as described in equation (8.7) in the book.
        custom_matrix_inversion_enabled (bool): Enables custom matrix inversion TODO ref book
        variable_scaling_enabled (bool): Enables variable scaling inversion TODO ref book
        advanced_stopping_criteria_enabled (bool): Advanced stopping criteria enabled as described in the PDF of phase 2
    """
    gradient_approximation_enabled: bool = False
    hessian_approximation_enabled: bool = False
    custom_matrix_inversion_enabled: bool = True
    variable_scaling_enabled: bool = True
    advanced_stopping_criteria_enabled: bool = True


@dataclass
class MinimizationProblem:
    """
    Data class containing all necessary information of a minimization problem to support
    steepest descent, newton, quasi-newton and conjugate minimization.

    Args:
        A (np.ndarray): Matrix A, used for solving the problem
        b (np.ndarray): Vector b, used for solving the problem
        f (Callable): The function (objective) we are trying to minimize.
        solution (list): The solution(s) to the minimization problem.
                             Might contain multiple if there are multiple local minimizers.
        x0 (list): The starting point for the minimization procedure.
        gradient_f (Callable): The gradient function of f. Optional.
        hessian_f (Callable): The hessian function of f. Optional.
    """
    A: np.ndarray
    b: np.ndarray
    f: Callable[[np.ndarray], np.ndarray]
    solution: np.ndarray
    x0: np.ndarray
    settings: MinimizationProblemSettings = MinimizationProblemSettings()
    gradient_f: Callable[[np.ndarray], np.ndarray] = None
    hessian_f: Callable[[np.ndarray], np.ndarray] = None

    # Why do we pass f, when it is saved in the state?
    def calc_gradient_at(self, x: np.ndarray) -> np.ndarray:
        """Calculate gradient at point `x`. Uses an approximation, if the gradient is not explicitly known.

        Args:
            x (np.ndarray): Array representing some point in the domain of the function.

        Returns:
            np.ndarray: (Approxmiated or true) gradient at point `x`.
        """

        return (self.gradient_f or self._central_difference_gradient)(x)

    def _central_difference_gradient(self, x: np.ndarray) -> np.ndarray:
        """Approximate gradient as described in equation (8.7), called the 'central difference formula'.

        Args:
            x (np.ndarray): Function input.

        Returns:
            np.ndarray: Approximated gradient.
        """
        eps = self._find_epsilon(x)
        eps_vectors = np.eye(N=len(x)) * eps
        return np.array([
            (self.f(x + eps_vector) - self.f(x - eps_vector))/(2*eps) for eps_vector in eps_vectors
        ])

    def calc_hessian_at(self,
            x: np.ndarray) -> np.ndarray:
        return (self.hessian_f or self.hessian_approximation)(x)

    def hessian_approximation(self, x: np.ndarray) -> np.ndarray:
        """Approximate Hessian based on equation (8.21) in the book.

        Args:
            x (np.ndarray): Point for which we approximate the function's Hessian.

        Returns:
            np.ndarray: Approximated Hessian.
        """
        eps = self._find_epsilon(x)
        eps_vectors = np.eye(N=len(x)) * eps
        
        hess = np.array([
            [self._hess_approx_num(x, eps_i, eps_j) for eps_i in eps_vectors]
            for eps_j in eps_vectors
        ]) / (eps**2)
        

        # If the hessian approximation is basically 0, we are already close.
        # Avoids SingularMatrix errors.
        if sum(abs(x) for row in hess for x in row) < 0.0001: 
            return np.eye(len(x))

        return hess
        
    def _hess_approx_num(self, x: np.ndarray, eps_i: np.ndarray, eps_j: np.ndarray) -> float:
        f = self.f
        return f(x + eps_i + eps_j) - f(x + eps_i) - f(x + eps_j) + f(x)

    def _find_epsilon(self, x: np.ndarray):
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


@dataclass
class IterationState:
    """
    Data class containing the state of a direction calculation iteration
    within a minimization procedure,holding interesting data for consumers
    and for the next iteration of the direction calculation.

    Args:
        x (list): The approximated minimizer input for a specific iteration.
        direction (list): The calculated direction (p) of a specific iteration.
        gradient (list): The calculated gradient at x at a specific iteration.
    """
    x: np.ndarray
    direction: np.ndarray
    gradient: np.ndarray


@dataclass
class BfgsQuasiNewtonState(IterationState):
    """
    Data class containing the state of a direction calculation iteration
    within a BFGS quasi-newton minimization procedure, holding interesting data for consumers
    and for the next iteration of the direction calculation.

    Args:
        x (list): The approximated minimizer input for a specific iteration.
        direction (list): The calculated direction (p) of a specific iteration.
        gradient (list): The calculated gradient at x at a specific iteration.
        H: (2D list): The H that was used to calculate the direction (p).
                      Stored in the state to avoid recomputation at the next iteration.
    """
    H: np.ndarray
