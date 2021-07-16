
from typing import Tuple, List, Sequence

import numpy as np

from quadratic.quadratic_problem import QuadraticProblem
from simplex.base import find_x0
from shared.constraints import combine_linear, EquationType, LinearConstraint, LinearCallable

QP_MAX_ITER: int = 1_000


def min_eq_qp(problem: QuadraticProblem) -> Tuple[np.ndarray, int]:
    """Compute minimizer of equality constrained problem,
    by solving (16.4).

    Args:
        problem: Equality constrained problem.

    Returns:
        Minimizer x_star, which is the solution to (16.4) and iteration count (hardcoded to 1 here).
    """
    kkt_solution = np.block([-problem.c, problem.b])
    x_lambda = solve_kkt_schur(problem, kkt_solution)

    x = x_lambda[:len(problem.G)]
    return x, 1


def transform_working_set_to_eq_constraints(working_set: Sequence[LinearConstraint]) -> Sequence[LinearConstraint]:
    return [LinearConstraint(
        c=LinearCallable(a=c.c.a, b=0),
        equation_type=c.equation_type) for c in working_set]


def min_ineq_qp(problem: QuadraticProblem) -> Tuple[np.ndarray, int]:
    x = find_x0(problem, standardized=False)

    active_set = problem.active_set_at(x, as_equalities=True)

    # Sample ~ 4/5 of the active constraints as equalities.
    #working_set = sample(active_set, k=int(np.ceil(len(active_set) * 0.8)))
    working_set = [active_set[0]]

    c = problem.c
    G = problem.G

    for i in range(QP_MAX_ITER):
        # Solve subproblem.
        g = G@x + c
        subproblem = QuadraticProblem(
            G=G, c=g,
            constraints=transform_working_set_to_eq_constraints(working_set),
            n=len(G), solution=None, x0=None
        )
        p, _ = min_eq_qp(subproblem)

        if np.all(p == 0):
            A, _ = combine_linear([eq.c for eq in working_set])
            lambda_vec = np.linalg.solve(A, g)

            # in the working set we transformed all constraints into equalities, but we want to check for inequalities
            current_set = [constr for constr in problem.constraints if
                           np.any([constr.equal_callables(working_constr) for working_constr in working_set])]
            _is_ineq_constr = [eq.equation_type != EquationType.EQ for eq in current_set]
            lambda_vec = lambda_vec[_is_ineq_constr]

            if np.all(lambda_vec >= 0) and len(lambda_vec) != 0:
                return x, i + 1
            else:
                least_lambda_index = np.argmin(lambda_vec)
                del working_set[least_lambda_index]
        else:
            # blocking constraints are all constraints of the problem that are not in the working set
            non_working_set_constraints = [constr for constr in problem.constraints if not
                                    np.any([constr.equal_callables(working_constr) for working_constr in working_set])]
            alpha, blocking_constraints = compute_alpha_and_blocking_constraints(non_working_set_constraints, p, x)
            x += alpha*p
            if blocking_constraints:
                working_set.append(blocking_constraints.pop())

    raise TimeoutError(f"Solution not found within {QP_MAX_ITER} steps; current x = {x}")


def compute_alpha_and_blocking_constraints(
        non_working_set_constraints: List[LinearConstraint],
        p: np.ndarray,
        x: np.ndarray) -> Tuple[float, List[LinearConstraint]]:
    """Compute alpha and blocking constraints for Algorithm 16.3 as described in equation (16.41).

    Args:
        non_working_set_constraints: List of constraints which are not in the working set.
        p: Solution to the subproblem (16.39)
        x: Current iterate.

    Returns:
           Alpha (float)
    """
    if not non_working_set_constraints:
        return 1, []
    non_working_set_constraints_as_callables = [constraint.c for constraint in non_working_set_constraints]
    A, b = combine_linear(non_working_set_constraints_as_callables)

    alpha = 1
    blocking_constraints = []
    for i, (b_i, a_i) in enumerate(zip(b, A)):
        blocking_term = (b_i - np.inner(a_i, x))/np.inner(a_i, p)
        if blocking_term < 1:
            blocking_constraints.append(non_working_set_constraints[i])
            alpha = min(alpha, blocking_term)

    return alpha, blocking_constraints


def minimize_quadratic_problem(problem: QuadraticProblem) -> Tuple[np.ndarray, int]:
    """Compute minimizer of quadratic problem."""
    return min_ineq_qp(problem) if problem.is_inequality_constrained else min_eq_qp(problem)


def solve_kkt_schur(problem: QuadraticProblem, kkt_solution: np.ndarray):
    """Computes solution to KKT-matrix equation using the Schur-complement method (page 455) using the inverse
    of the KKT-Matrix (16.16)

    Args:
        problem: QuadraticProblem to compute the inverse KKT-matrix
        kkt_solution: np.ndarray "b" in Ax=b

    Returns:
        np.ndarray: solution to Ax=b where A is the KKT-matrix and b the KKT-solution
    """

    def get_kkt_inv(problem: QuadraticProblem):
        """Return inverse of KKT-matrix as defined in 16.16.

        Args:
            problem: QuadraticProblem to compute the inverse KKT-matrix

        Returns:
            np.ndarray: inverse of KKT-matrix
        """

        A = problem.A
        G = problem.G

        G_inv = np.linalg.inv(G)
        AGAT = np.linalg.inv(A @ G_inv @ A.T)

        C = G_inv - G_inv @ A.T @ AGAT @ A @ G_inv
        E = G_inv @ A.T @ AGAT
        F = - AGAT

        kkt_inv = np.block([
            [C, E],
            [E.T, F]
        ])

        return kkt_inv

    kkt_inv = get_kkt_inv(problem)
    x_lambda = kkt_inv @ kkt_solution

    return x_lambda
