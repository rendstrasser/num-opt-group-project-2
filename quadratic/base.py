from typing import Tuple, List, Sequence, Optional

import numpy as np

from quadratic.quadratic_problem import QuadraticProblem
from shared.constraints import combine_linear, EquationType, LinearConstraint, LinearCallable
from shared.factorizations import qr_factorization_householder
from shared.solve_linear_system import solve_positive_definite

QP_MAX_ITER: int = 1_000


def minimize_quadratic_problem(original_problem: QuadraticProblem, use_nullspace_method=False) -> Tuple[np.ndarray, int]:
    """
    Compute minimizer of a quadratic problem.

    Args:
        original_problem: quadratic problem that we want to solve
        use_nullspace_method: setting that defines if the nullspace method should be used for solving
            equality-constrained problems instead of the schur method

    Returns:
        Tuple of the minimizer x and the number of iterations it took to find x
    """

    # we expect inequalities to be greater-than inequalities
    prepared_constraints = [c.as_ge_if_le() for c in original_problem.constraints]
    problem = QuadraticProblem(G=original_problem.G, c=original_problem.c, n=original_problem.n, x0=original_problem.x0,
                               solution=original_problem.solution, constraints=prepared_constraints)

    return min_ineq_qp(problem, use_nullspace_method) if problem.is_inequality_constrained \
        else min_eq_qp(problem, None, False)


def min_eq_qp(
        problem: QuadraticProblem,
        x: np.ndarray,
        use_nullspace_method: bool = False) -> Tuple[np.ndarray, int]:
    """Compute minimizer of equality constrained problem,
    by solving (16.4).

    Args:
        problem: Equality constrained problem.

    Returns:
        Minimizer x_star, which is the solution to (16.4) and iteration count (hardcoded to 1 here).
    """
    if len(problem.constraints) == 0:
        return min_no_constraint_qp(problem)

    if use_nullspace_method:
        x = solve_kkt_nullspace(problem, x)
    else:
        x_lambda = solve_kkt_schur(problem)
        x = x_lambda[:len(problem.G)]

    return x, 1


def min_no_constraint_qp(problem: QuadraticProblem) -> Tuple[np.ndarray, int]:
    """
    Computes minimizer for an unconstrained quadratic problem

    Args:
        problem: Unconstrained quadratic problem that we want to solve

    Returns:
        Minimizer x_star and iteration count (hardcoded to 1 here).
    """

    # our quadratic problems are ensured to be positive-definite
    x = np.linalg.solve(problem.G, -problem.c)

    return x, 1


def transform_working_set_to_eq_constraints(working_set: Sequence[LinearConstraint]) -> Sequence[LinearConstraint]:
    """Transforms the constraints in the working set to equality-constrained problems with b=0"""
    return [LinearConstraint(
        c=LinearCallable(a=c.c.a, b=0),
        equation_type=c.equation_type) for c in working_set]


def min_ineq_qp(problem: QuadraticProblem, use_nullspace_method=False) -> Tuple[np.ndarray, int]:
    """Compute minimizer of inequality constrained problem using the active-set method (algorithm 16.3).

    Args:
        problem: Equality constrained problem.

    Returns:
        Minimizer x_star, which is the solution to (16.4) and iteration count (hardcoded to 1 here).
    """
    x = problem.find_x0()

    # we start with an empty working set to ensure linear independence in the constraints from here on
    working_set = []

    c = problem.c
    G = problem.G

    for i in range(QP_MAX_ITER):
        # Solve subproblem.
        g = G @ x + c
        subproblem = QuadraticProblem(
            G=G, c=g,
            constraints=transform_working_set_to_eq_constraints(working_set),
            n=len(G), solution=None, x0=None
        )
        p, _ = min_eq_qp(subproblem, x, use_nullspace_method)

        if np.allclose(p, np.zeros_like(p)):

            # get lambda-vector
            if len(working_set) == 0:
                lambda_vec = np.empty(shape=0)
            else:
                A, _ = combine_linear([eq.c for eq in working_set])
                lambda_vec = np.linalg.lstsq(A.T, g, rcond=-1)[0]

                # in the working set we transformed all constraints into equalities, but we want to check for inequalities
                current_set = [constr for constr in problem.constraints if
                               np.any([constr.equal_callables(working_constr) for working_constr in working_set])]
                _is_ineq_constr = [eq.equation_type != EquationType.EQ for eq in current_set]
                lambda_vec = lambda_vec[_is_ineq_constr]

            if np.all(lambda_vec >= 0):
                return x, i + 1
            else:
                least_lambda_index = np.argmin(lambda_vec)
                del working_set[least_lambda_index]
        else:
            # compute next x
            non_working_set_constraints = [constr for constr in problem.constraints if not
                    np.any([constr.equal_callables(working_constr) for working_constr in working_set])]
            alpha, blocking_constraint = compute_alpha_and_blocking_constraints(non_working_set_constraints, p, x)
            x += alpha * p

            # if we have a blocking constraint, add it to the working set
            if blocking_constraint is not None:
                working_set.append(blocking_constraint)

    raise TimeoutError(f"Solution not found within {QP_MAX_ITER} steps; current x = {x}")


def compute_alpha_and_blocking_constraints(
        non_working_set_constraints: List[LinearConstraint],
        p: np.ndarray,
        x: np.ndarray) -> Tuple[float, Optional[LinearConstraint]]:
    """Compute alpha and blocking constraints for Algorithm 16.3 as described in equation (16.41).

    Args:
        non_working_set_constraints: List of constraints which are not in the working set.
        p: Solution to the subproblem (16.39)
        x: Current iterate.

    Returns:
           Alpha (float)
    """
    if not non_working_set_constraints:
        return 1, None

    non_working_set_constraints_as_callables = [constraint.c for constraint in non_working_set_constraints]
    A, b = combine_linear(non_working_set_constraints_as_callables)

    alpha = 1
    blocking_constraint = None
    for i, (b_i, a_i) in enumerate(zip(b, A)):
        ap = np.inner(a_i, p)
        if ap < 0:
            blocking_term = (b_i - np.inner(a_i, x)) / np.inner(a_i, p)
            if blocking_term < alpha:
                blocking_constraint = non_working_set_constraints[i]
                alpha = blocking_term

    return alpha, blocking_constraint


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


def solve_kkt_schur(problem: QuadraticProblem):
    """Computes solution to KKT-matrix equation using the Schur-complement method (page 455) using the inverse
    of the KKT-Matrix (16.16)

    Args:
        problem: QuadraticProblem to compute the inverse KKT-matrix

    Returns:
        np.ndarray: solution to Ax=b where A is the KKT-matrix and b the KKT-solution
    """

    kkt_solution = np.block([-problem.c, problem.b])
    kkt_inv = get_kkt_inv(problem)
    x_lambda = kkt_inv @ kkt_solution

    return x_lambda


def solve_kkt_nullspace(problem: QuadraticProblem, x: np.ndarray) -> np.ndarray:
    """Computes solution to KKT matrix equation using the null-space method (page 457)

    Args:
        problem: QuadraticProblem to compute solution for
        x: Current point x

    Returns:
        np.ndarray: solution to the quadratic problem
    """

    A = problem.A
    G = problem.G

    m, n = A.shape

    # perform QR-factorization to retrieve Y and Z matrices
    Q, _ = qr_factorization_householder(A.T)

    # split up Q into Q1 and Q2, which by equation (15.22) are Y and Z
    Y = Q[:, :m]
    Z = Q[:, m:]

    g = G @ x + problem.c
    h = A @ x - problem.b

    p_y = np.linalg.inv(A @ Y) @ (-h)  # equation (16.18)
    p_z = solve_positive_definite(Z.T @ G @ Z, -Z.T @ G @ Y @ p_y - Z.T @ g)    # equation (16.19)
    p = Y @ p_y + Z @ p_z   # equation (16.17)

    return p + x    # equation (16.6)
