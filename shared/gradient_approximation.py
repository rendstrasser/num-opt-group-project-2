from typing import Callable

import numpy as np


def gradient_approximation(f: Callable, x: np.ndarray) -> np.ndarray:
    """Approximate gradient as described in equation (8.7), called the 'central difference formula'.

    Args:
        x (np.ndarray): Function input.

    Returns:
        np.ndarray: Approximated gradient.
    """
    eps = _find_epsilon(x)
    eps_vectors = np.eye(N=len(x)) * eps
    return np.array([
        (f(x + eps_vector) - f(x - eps_vector)) / (2 * eps) for eps_vector in eps_vectors
    ])


def hessian_approximation(f: Callable, x: np.ndarray) -> np.ndarray:
    """Approximate Hessian based on equation (8.21) in the book.

    Args:
        f (Callable): Function to approximate the hessian of.
        x (np.ndarray): Point for which we approximate the function's Hessian.

    Returns:
        np.ndarray: Approximated Hessian.
    """
    eps = _find_epsilon(x)
    eps_vectors = np.eye(N=len(x)) * eps

    hess = np.array([
        [_hess_approx_num(f, x, eps_i, eps_j) for eps_i in eps_vectors]
        for eps_j in eps_vectors
    ]) / (eps ** 2)

    # If the hessian approximation is basically 0, we are already close.
    # Avoids SingularMatrix errors.
    if sum(abs(entry) for row in hess for entry in row) < 0.0001:
        return np.eye(len(x))

    return hess


def _hess_approx_num(f: Callable, x: np.ndarray, eps_i: np.ndarray, eps_j: np.ndarray) -> float:
    return f(x + eps_i + eps_j) - f(x + eps_i) - f(x + eps_j) + f(x)


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
