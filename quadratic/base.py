from random import sample
from typing import Tuple

import numpy as np

from quadratic.quadratic_problem import QuadraticProblem
from simplex.base import find_x0

QP_MAX_ITER: int = 1_000


def min_eq_qp(problem: QuadraticProblem) -> np.ndarray:
    """Compute minimizer of equality constrained problem,
    by solving (16.4).

    Args:
        problem: Equality constrained problem.

    Returns:
        np.ndarray: Minimizer x_star, which is the solution to (16.4).
    """
    kkt, kkt_solution = kkt_matrix(problem)

    # TODO: Use some other solving technique.
    x_lambda = np.linalg.solve(kkt, kkt_solution)

    x = x_lambda[:len(problem.G)]
    return x


def min_ineq_qp(problem: QuadraticProblem) -> np.ndarray:
    x = find_x0(problem, standardized=False)

    active_set = problem.active_set_at(x, as_equalities=True)

    # Sample ~ 4/5 of the active constraints as equalities.
    working_eq_set = sample(active_set, k=np.ceil(len(active_set) * 0.8))

    c = problem.c
    G = problem.G

    for k in range(QP_MAX_ITER):
        subproblem = QuadraticProblem(
            G=G, c=G@x + c, constraints=working_eq_set, n=len(G)
        )
        p = min_eq_qp(subproblem)


def minimize_quadratic_problem(problem: QuadraticProblem) -> np.ndarray:
    """Compute minimizer of quadratic problem."""
    return min_ineq_qp(problem) if problem.is_inequality_constrained else min_eq_qp(problem)


def kkt_matrix(problem: QuadraticProblem) -> Tuple[np.ndarray, np.ndarray]:
    """Return KKT-matrix system as defined in equation (16.4).

    Args:
        problem: QuadraticProblem to compute the KKT-matrix from

    Returns:
        Tuple[np.ndarray, np.ndarray]: Left matrix [[G, -A.T], [A, 0]]
                                       and right vector [-c, b] of the equation.
    """

    A = problem.A
    G = problem.G
    c = problem.c
    b = problem.b

    min_A = min(A.shape)
    zero = np.zeros(shape=(min_A, min_A))

    left = np.block([
        [G, -A.T],
        [A, zero]
    ])

    right = np.block([-c, b])

    return left, right

