import numpy as np
from typing import Tuple, Optional

from quadratic.base import minimize_quadratic_problem
from shared.minimization_problem import MinimizationProblem
from shared.constraints import Constraint, LinearCallable, LinearConstraint, EquationType, combine_linear
from quadratic.quadratic_problem import QuadraticProblem
from sqp.quasi_newton_approx import sr1
from sqp.sub_problem_structures import SqpIterateQuadraticProblem

# precision constant
_epsilon = np.sqrt(np.finfo(float).eps)

# defines the minimum step sizes in terms of the norm of the direction
# mainly to avoid extremely slow convergence for showcase purpose
# can also be replaced by _epsilon if slow convergence is not an issue
MIN_BACKTRACKING_STEP_NORM_SIZE = 1e-5


def minimize_nonlinear_problem(
        original_problem: MinimizationProblem,
        eta=0.8, tau=0.8, tolerance=1e-3, max_iter=100_000) -> Tuple[np.ndarray, int]:
    """
    Performs the Line Search SQP Algorithm (18.3) to solve a non-linear constrained problem.

    Args:
        original_problem: Problem that we want to minimize
        eta: As in 18.3 - used to dampen the influence of the directional gradient in the line search
        tau: As in 18.3 - multiplied with alpha in the line search when step-size condition is not yet fulfilled
        tolerance: Tolerance of how close our solution needs to be in terms of the KKT-conditions.
                That is, for conditions 12.34b-e, the KKT-conditions are checked elementwise to
                the granularity of the tolerance and for 12.34a, we check the L2-norm of the Lagrangian gradient
                to be smaller than the tolerance.
        max_iter: Maximum amount of iterations before a TimeoutError will be thrown

    Returns:
        Tuple of minimizer x and the number of iterations it took to find the minimizer
    """

    # ensure that our problem is in the format that we expect
    problem = standardize_problem(original_problem)

    # Define x0 and lambda0
    x = np.ones(shape=problem.n)
    lambda_ = np.zeros(shape=len(problem.constraints))
    if problem.x0 is not None:
        x = np.copy(problem.x0)

    # initialize to avoid errors when used for the first iteration
    B = None
    x_prev = None
    mu = None

    for i in range(max_iter):
        # Calculate all the terms needed for the stopping criterion check
        constraint_values = problem.calc_constraints_at(x)

        # check stopping-condition (KKT conditions fulfilled)
        if kkt_fulfilled(problem, x, lambda_, constraint_values, tolerance):
            return x, i

        # calculate all the other terms needed for this iteration
        f_x = problem.calc_f_at(x)
        f_grad = problem.calc_gradient_at(x)
        c_norm = np.linalg.norm(constraint_values, ord=1)
        A = problem.calc_constraints_jacobian_at(x)

        # second stopping criteria: x_k dose not change anymore
        if x_prev is not None:
            if np.allclose(x, x_prev, atol=1e-12, rtol=1e-10):
                return x, i

        # approximate the Hessian using SR1
        B = sr1(problem, B=B, x=x, x_old=x_prev, lambda_=lambda_)

        # Create and solve quadratic sub-problem
        quadr_problem = SqpIterateQuadraticProblem(G=B, c=f_grad,
                                                   original_constraints=problem.constraints,
                                                   original_constraint_values=constraint_values,
                                                   original_constraint_jacobian=A)
        p, l_hat = solve_quadratic_sub_problem(quadr_problem)
        p_lambda = l_hat - lambda_
        mu = find_mu(mu, f_grad, p, B, c_norm)

        # Find alpha for step-length calculation using a line search
        alpha = find_alpha_with_line_search(problem, x, f_x, f_grad, c_norm, mu, p, eta, tau)

        # second stopping criteria: if gradient of merit function is 0 we are in an extremum!
        if np.linalg.norm(l1_merit_directional_gradient(f_grad, p, mu, c_norm)) <= 10 ** -10:
            return x, i

        # Remember current x for SR1 and calculate next x and lambda vector
        x_prev = np.copy(x)
        x += alpha * p
        lambda_ += alpha * p_lambda

    raise TimeoutError(f"SQP ran into timeout with {max_iter} steps")


def find_alpha_with_line_search(
        problem: MinimizationProblem,
        x: np.ndarray,
        f_x: float,
        f_grad: np.ndarray,
        c_norm: float,
        mu: float,
        p: np.ndarray,
        eta: float,
        tau: float) -> float:
    """
    Performs the backtracking line search for algorithm 18.3.
    Note: We mock the second Wolfe condition by requiring step-sizes of at least 1e-5.
    This is to avoid ultra-slow convergence of some non-convex problems.

    Args:
        problem: Problem that we want to minimize
        x: Current approximated minimizer x
        f_x: Value of f(x)
        f_grad: Value of derivative of f(x), i.e., f'(x)
        c_norm: L1-norm of all c_i(x)
        mu: Mu as defined in 18.36
        p: Direction p found via QP
        eta: As in 18.3 - used to dampen the influence of the directional gradient in the line search
        tau: As in 18.3 - multiplied with alpha in the line search when step-size condition is not yet fulfilled

    Returns:
        Alpha that satisfies the conditions of the backtracking line search.
    """

    # Calculate the merit function and directional gradient of the merit function at x
    merit_at_x = l1_merit(f_x, mu, c_norm)
    dir_gradient_merit_at_x = l1_merit_directional_gradient(f_grad, p, mu, c_norm)

    alpha = 1
    while not merit_wolfe_condition_satisfied(problem, x, alpha, p, mu, merit_at_x, dir_gradient_merit_at_x, eta):
        alpha *= tau

        if alpha * np.linalg.norm(p) < MIN_BACKTRACKING_STEP_NORM_SIZE:
            # step must not become smaller than precision, early exit to ensure valid alpha
            # note: we use 1e-5 which is a lot bigger than our precision; this is to encourage faster convergence
            # for non-convex problems with low-tolerance needs (e.g. 1e-3)
            break

    return alpha


def merit_wolfe_condition_satisfied(
        problem: MinimizationProblem,
        x: np.ndarray,
        alpha: float,
        p: np.ndarray,
        mu: float,
        merit_at_x: float,
        dir_gradient_merit_at_x: float,
        eta: float) -> bool:
    """
    Validates, if the first Wolfe condition is satisfied in terms of the merit function.

    Args:
        problem: Problem that we want to minimize
        x: Current approximated minimizer x
        alpha: Current alpha-value for step-size calculation
        p: Direction p found via QP
        mu: Mu as defined in 18.36
        merit_at_x: Value of merit function at x
        dir_gradient_merit_at_x: Value of directional gradient of merit function at x
        eta: As in 18.3 - used to dampen the influence of the directional gradient in the backtracking line search

    Returns:
        True, if the first Wolfe condition is satisfied in terms of the merit function.
    """

    next_x_candidate = x + alpha * p

    # recalculate constraint values c_i(x) at the candidate x
    c = problem.calc_constraints_at(next_x_candidate)
    c_norm = np.linalg.norm(c, ord=1)

    # calculate the two terms on both sides for the first Wolfe condition inequality
    left_term_of_inequality = l1_merit(problem.calc_f_at(next_x_candidate), mu, c_norm)
    right_term_of_inequality = merit_at_x + eta * alpha * dir_gradient_merit_at_x

    return left_term_of_inequality <= right_term_of_inequality


def standardize_problem(original_problem: MinimizationProblem) -> MinimizationProblem:
    """

    Args:
        original_problem: Problem that we want to standardize

    Returns:
        Problem in standardized form as required by SQP (18.10)

    """

    # convert less-than inequalities to greater-than inequalities
    prepared_constraints = [c.as_ge_if_le() for c in original_problem.constraints]

    return MinimizationProblem(f=original_problem.f, n=original_problem.n, x0=original_problem.x0,
                               solution=original_problem.solution, constraints=prepared_constraints)


def solve_quadratic_sub_problem(problem: QuadraticProblem) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the quadratic sub-problem (18.11).

    Args:
        problem: quadratic sub-problem that we want to solve

    Returns:
        Tuple of minimizer x and the corresponding lambda vector (Lagrange multipliers)
    """
    x_solution, _ = minimize_quadratic_problem(problem)
    lambda_ = find_lambda_of_qp(problem, x_solution)

    return x_solution, lambda_


def find_mu(
        prev_mu: Optional[float],
        f_grad: np.ndarray,
        p: np.ndarray,
        B: np.ndarray,
        c_norm: float,
        rho=0.5) -> float:
    """
    Find mu as described in 18.36 with sigma=1 (as required by algorithm 18.3)

    Args:
        prev_mu: Mu as returned by this function from the previous iteration
        f_grad: Derivative of function at current approximated minimizer x, i.e, f'(x)
        p: Direction found with QP
        B: Langrange function Hessian or approximation of Hessian
        c_norm: L1-norm of constraints at current approximimated minimizer x, i.e, L1-norm of all c_i(x)
        rho: As in 18.36 - decides how much penalty we want to have

    Returns:
        Mu that fulfilles the inequality of 18.36
    """

    hessian_quadratic_p = p @ B @ p
    sigma = 1  # hardcoded according to algorithm

    inequality_18_36 = (f_grad @ p + (sigma / 2) * hessian_quadratic_p) / ((1 - rho) * c_norm)

    # if previous mu fulfills the inequality, use it
    if prev_mu is not None and prev_mu >= inequality_18_36:
        return prev_mu

    # use a new mu that fulfills the inequality
    return inequality_18_36 + 0.1


def l1_merit(
        f_x: float,
        mu: float,
        c_norm: float) -> float:
    """
    Calculates the merit function at x.

    Args:
        f_x: Function value at current iterate x
        mu: Mu that fulfills 18.36
        c_norm: L1-norm of constraint values at current iterate x

    Returns:
        Value of merit function
    """
    return f_x + mu * c_norm


def l1_merit_directional_gradient(
        f_grad: np.ndarray,
        p: np.ndarray,
        mu: float,
        c_norm: float) -> float:
    """
    Calculates the directional gradient of the merit function wrt p at x.

    Args:
        f_grad: Derivative of function at current iterate x
        p: Direction found with QP
        mu: Mu that fulfills 18.36
        c_norm: L1-norm of constraint values at current iterate x

    Returns:
        Directional gradient of merit function wrt p
    """
    return f_grad @ p - mu * c_norm


def find_lambda_of_qp(
        problem: QuadraticProblem,
        x_solution: np.ndarray) -> np.ndarray:
    """
    Computes the lambda vector (Lagrange multipliers) for a given solution x and a quadratic sub-problem

    Args:
        problem: Quadratic sub-problem
        x_solution: Solution x of KKT-system for quadratic sub-problem

    Returns:
        Lambda vector (Lagrange multipliers) for given solution x
    """

    # start with 0 for all constraints
    # s.t. all non-active constraints i already have the correct lambda_i=0
    lambda_ = np.zeros(len(problem.constraints))

    # find all active constraints at the current point
    active_constraint_mask = np.array([c.is_active(x_solution) for c in problem.constraints])
    if np.all(~active_constraint_mask):
        # no active constraints, early exit with lambda=0
        return lambda_

    # compute derivative of quadratic function
    g = problem.G @ x_solution + problem.c

    # calculate constraint jacobian for active constraints
    # note: slicing will work as constraints are always numpy arrays
    # noinspection PyTypeChecker
    A, _ = combine_linear([c.c for c in problem.active_set_at(x_solution, as_equalities=False)])

    # solve linear system Ax=g which effectively ensures KKT-condition 12.34a
    lambda_[active_constraint_mask] = np.linalg.lstsq(A.T, g, rcond=-1)[0]

    return lambda_


def transform_sqp_to_linear_constraint(
        constraint: Constraint,
        constraint_value: float,
        constraint_grad: np.ndarray) -> LinearConstraint:
    """
    Creates a linear constraint for a quadratic sub-problem based on a non-linear constraint (see 18.11).

    Args:
        constraint: Original non-linear constraint
        constraint_value: Value of constraint at x
        constraint_grad: Value of derivative of constraint at x

    Returns:
        Linear constraint for quadratic sub-problem
    """

    return LinearConstraint(
        c=LinearCallable(a=constraint_grad, b=-constraint_value),
        equation_type=constraint.equation_type)


def kkt_fulfilled(
        problem: MinimizationProblem,
        x: np.ndarray,
        lambda_: np.ndarray,
        constraint_values: np.ndarray,
        tolerance=1e-5) -> bool:
    """
    Checks all the KKT-conditions (12.34) and if they are sufficiently close to the tolerance,
    we return True.

    Args:
        problem: Problem that we want to minimize
        x: Current approximated minimizer x
        lambda_: Current approximated lambda vector (Lagrange multipliers)
        constraint_values: Constraint values at current iterate x
        tolerance: Tolerance of how close our solution needs to be in terms of the KKT-conditions

    Returns:
        True, if KKT-conditions are sufficiently close to being fulfilled
    """

    # 12.34d
    if np.any(lambda_ + tolerance < 0):
        return False

    for lambda_i, constraint_i, constraint_value_i in zip(lambda_, problem.constraints, constraint_values):
        # 12.34b
        if constraint_i.equation_type == EquationType.EQ:
            if np.any(abs(constraint_value_i) > tolerance):
                return False
        # 12.34c
        else:
            if np.any(constraint_value_i + tolerance < 0):
                return False

        # 12.34e
        if np.any(abs(lambda_i * constraint_value_i) > tolerance):
            return False

    l_gradient = problem.calc_lagrangian_gradient_at(x, lambda_)

    # 12.34a
    if np.any(abs(l_gradient) > tolerance):
        return False

    return True
