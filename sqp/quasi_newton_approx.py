import numpy as np
from typing import Optional

from shared.minimization_problem import MinimizationProblem


def damped_bfgs_updating(
        problem: MinimizationProblem,
        B: Optional[np.ndarray],
        x: np.ndarray,
        x_old: Optional[np.ndarray],
        lambda_) -> np.ndarray:
    """
    Approximates the Hessian of the Lagrange function based on Procedure 18.2.

    Args:
        problem: Problem that we want to minimize
        B: Previous approximation of Lagrange function Hessian
        x: Current approximated minimizer x
        x_old: Previous approximated minimizer x
        lambda_: Approximated lambda vector (Lagrange multipliers) at current iterate

    Returns:
        Approximation of Lagrange Hessian at the current iterate x
    """

    # if we have no previous iterate, just use the identity matrix to begin with
    if B is None or x_old is None:
        return np.eye(problem.n)

    s = x - x_old
    L_grad_x = problem.calc_lagrangian_gradient_at(x, lambda_)
    L_grad_xold = problem.calc_lagrangian_gradient_at(x_old, lambda_)
    y = L_grad_x - L_grad_xold

    s_B_s = np.inner(np.inner(s, B), s)

    if np.inner(s, y) >= 0.2 * s_B_s:
        theta = 1
    else:
        theta = (0.8 * s_B_s) / (s_B_s - np.inner(s, y))

    r = theta * y + (1 - theta) * (B @ s)

    B_new = B - (np.outer(B @ s, s) @ B) / s_B_s + np.outer(r, r) * 1 / np.inner(s, r)

    return B_new


def sr1(
        problem: MinimizationProblem,
        B: Optional[np.ndarray],
        x: np.ndarray,
        x_old: Optional[np.ndarray],
        lambda_: np.ndarray,
        delta=0.1) -> np.ndarray:
    """
    Approximates the Hessian of the Lagrange function based on the SR1 algorithm (6.24).

    Also ensures positive-definiteness of outcome by adding a multiple of the identity matrix to B if original SR1
    approximation is not positive-definite.

    Args:
        problem: Problem that we want to minimize
        B: Previous approximation of Lagrange function Hessian
        x: Current approximated minimizer x
        x_old: Previous approximated minimizer x
        lambda_: Approximated lambda vector (Lagrange multipliers) at current iterate
        delta: Factor for how much of the identity we add per positive-definitess check

    Returns:
        Approximation of Lagrange Hessian at the current iterate x
    """

    # if we have no previous iterate, just use the identity matrix to begin with
    if B is None or x_old is None:
        return np.eye(problem.n)

    s = x - x_old  # we would need to store x_old
    L_grad_x = problem.calc_lagrangian_gradient_at(x, lambda_)
    L_grad_xold = problem.calc_lagrangian_gradient_at(x_old, lambda_)
    y = L_grad_x - L_grad_xold

    y_minus_B_s = y - B @ s
    B_new = B + np.outer(y_minus_B_s, y_minus_B_s) / np.inner(y_minus_B_s, s)

    # according to page 538, adding sufficiently large multiple of identity
    # while criterion might not be sufficient enough (but hopefully is)
    n = B.shape[0]
    while not np.all(np.linalg.eigvals(B_new) > 0):
        B_new += delta * np.eye(n)

    return B_new
