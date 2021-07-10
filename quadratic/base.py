from random import sample

import numpy as np

from quadratic.quadratic_problem import QuadraticProblem
from shared.solve_linear_system import solve

PROJECTED_CG_TOLERANCE = 1e-4


def minimize_quadratic_problem(problem: QuadraticProblem) -> np.ndarray:
    """
    A problem - assumed to be in standard form - is optimized.
    """

    if all(constraint.is_equality for constraint in problem.constraints):
        return projected_cg()
    else:
        x = find_x0(problem)
        m = len(problem.b)  # number of constraints -> size of basis

        # TODO: Consider better selection than random.
        working_set = sample(problem.active_set_at(x), k=5)

        # Just to see if things run; TODO: Implement stopping criterion.
        for _ in range(10):
            # Above (16.39a)
            g = problem.G @ x + problem.c

            subproblem = QuadraticProblem(n=len(x), constraints=working_set, G=problem.G, c=g, solution=None, x0=None)

        return x


def projected_cg(problem: QuadraticProblem):
    """Compute the the solution of an equality-constrained quadratic programming problem,
    as denoted in 'Algorithm 16.2 (Projected CG Method)'.

    Args:
        problem: QuadraticProblem where all constraints are equalities.
    """
    if not all(constraint.is_equality for constraint in problem.constraints):
        raise ValueError("Projected CG was supplied an inequality-constrained problem.")

    # Unpack problem.
    A = problem.A
    b = problem.b
    c = problem.c
    G = problem.G

    # Compute H as defined on page 462. Needs to be non-singular, hence identity.
    H = np.eye(G)

    x = solve(A, b)

    Z = NotImplemented

    # (16.33)
    P = Z @ np.linalg.inv(Z.T @ H @ Z) @ Z.T

    r = G@x + c
    g = P @ r
    d = -g

    while np.inner(r, g) < PROJECTED_CG_TOLERANCE:

        # Find step-length
        alpha = np.inner(r, g) / np.inner(d, G @ d)

        # Step along direction
        x = x + alpha * d

        # What does this do?
        r_new = r + alpha * G @ d

        # New .. gradient?
        g_new = P @ r_new

        beta = np.inner(r_new, g_new) / np.inner(r, g)

        d = -g_new + beta*d

        g, r = g_new, r_new

    return x


def find_x0(problem: QuadraticProblem) -> np.ndarray:
    raise NotImplementedError
