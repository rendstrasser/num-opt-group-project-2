
from random import sample
from typing import Tuple, List, Sequence

import numpy as np

from quadratic.quadratic_problem import QuadraticProblem
from simplex.base import find_x0
from shared.constraints import combine_linear, combine_linear_constraints, LinearConstraint, LinearCallable, EquationType

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


def transform_working_set_to_eq_constraints(working_set: Sequence[LinearConstraint]) -> Sequence[LinearConstraint]:
    return [LinearConstraint(
        c=LinearCallable(a=c.c.a, b=0),
        equation_type=c.equation_type) for c in working_set]


def min_ineq_qp(problem: QuadraticProblem) -> np.ndarray:
    x = find_x0(problem, standardized=False)

    active_set = problem.active_set_at(x, as_equalities=True)

    # Sample ~ 4/5 of the active constraints as equalities.
    working_set = sample(active_set, k=int(np.ceil(len(active_set) * 0.8)))

    c = problem.c
    G = problem.G

    for _ in range(QP_MAX_ITER):

        # Solve subproblem.
        g = G@x + c
        subproblem = QuadraticProblem(
            G=G, c=g,
            constraints=transform_working_set_to_eq_constraints(working_set),
            n=len(G), solution=None, x0=None
        )
        p = min_eq_qp(subproblem)

        if np.all(p == 0):
            A, _ = combine_linear([eq.c for eq in working_set])
            lambda_vec = np.linalg.solve(A, g)

            # in the working set we transformed all constraints into equalities, but we want to check for inequalities
            current_set = [const for const in problem.constraints if const in working_set]
            _is_eq_const = [eq.equation_type != EquationType.EQ for eq in current_set]
            lambda_vec = lambda_vec[_is_eq_const]

            if np.all(lambda_vec >= 0) and len(lambda_vec) != 0:
                return x
            else:
                least_lambda_index = np.argmin(lambda_vec)
                del working_set[least_lambda_index]
        else:
            # blocking constraints are all constraints of the problem that are not in the working set
            blocking_constraints = [constraint for constraint in problem.constraints if constraint not in working_set]
            alpha = compute_alpha(blocking_constraints, p, x)
            x += alpha*p
            if blocking_constraints:
                working_set.append(blocking_constraints.pop())

    raise TimeoutError(f"Solution not found within {QP_MAX_ITER} steps; current x = {x}")


def compute_alpha(blocking_constraints: List[LinearConstraint], p: np.ndarray, x: np.ndarray) -> float:
    """Compute alpha for Algorithm 16.3 as described in equation (16.41).

    Args:
        blocking_constraints: List of active constraints which are not in the working set.
        p: Solution to the subproblem (16.39)
        x: Current iterate.

    Returns:
           Alpha (float)
    """
    if not blocking_constraints:
        return 1
    A, b = combine_linear_constraints(blocking_constraints)
    return min(
        1,
        min((b - np.inner(a, x))/np.inner(a, p) for b, a in zip(b, A) if np.inner(a, p) < 0)
    )


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
