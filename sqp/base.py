
import numpy as np
from typing import Tuple, Sequence

from quadratic.base import minimize_quadratic_problem
from shared.minimization_problem import MinimizationProblem
from shared.constraints import Constraint, LinearCallable, LinearConstraint, EquationType, combine_linear
from quadratic.quadratic_problem import QuadraticProblem
from sqp.quasi_newton_approx import damped_bfgs_updating, sr1


# should be done
def minimize_nonlinear_problem(
        original_problem: MinimizationProblem,
        eta=0.8, tau=0.8, tolerance=1e-3, max_iter=100_000) -> Tuple[np.ndarray, int]:

    # we expect inequalities to be greater-than inequalities
    prepared_constraints = [c.as_ge_if_le() for c in original_problem.constraints]
    problem = MinimizationProblem(f=original_problem.f, n=original_problem.n, x0=original_problem.x0,
                               solution=original_problem.solution, constraints=prepared_constraints)

    x = problem.x0
    if x is None:
        x = np.ones(shape=problem.n)

    lambda_ = np.zeros(shape=len(problem.constraints))

    f_x = problem.calc_f_at(x)
    f_grad = problem.calc_gradient_at(x)
    c = problem.calc_constraints_at(x)
    c_norm = np.linalg.norm(c, ord=1)
    A = problem.calc_constraints_jacobian_at(x)
    mu = None

    #B = problem.calc_lagrangian_hessian_at(x, lambda_)
    B = sr1(problem, None, x, None, lambda_)

    i = 0
    for i in range(max_iter):
        quadr_problem = create_iterate_quadratic_problem(problem, f_x, f_grad, c, A, B)
        p, l_hat = solve_quadratic(quadr_problem)
        p_lambda = l_hat - lambda_
        mu = find_mu(mu, f_grad, p, B, c_norm)

        merit_at_x = l1_merit(f_x, mu, c_norm)
        dir_gradient_merit = l1_merit_directional_gradient(f_grad, p, mu, c_norm)

        alpha = 1
        first_wolfe_condition_term = merit_at_x + eta * alpha * dir_gradient_merit
        merit_term = l1_merit(problem.calc_f_at(x + alpha * p), mu, c_norm)
        while merit_term > first_wolfe_condition_term:
            alpha *= tau

            c = problem.calc_constraints_at(x + alpha * p)
            c_norm = np.linalg.norm(c, ord=1)

            merit_term=l1_merit(problem.calc_f_at(x + alpha * p), mu, c_norm)
            first_wolfe_condition_term = merit_at_x + eta * alpha * dir_gradient_merit

            if alpha * np.linalg.norm(p) < 1e-5:
                # step must not become smaller than precision, early exit to ensure valid a
                break

        x_prev = np.copy(x)
        x += alpha * p

        lambda_ += alpha * p_lambda

        f_x = problem.calc_f_at(x)
        f_grad = problem.calc_gradient_at(x)
        c = problem.calc_constraints_at(x)
        c_norm = np.linalg.norm(c, ord=1)
        A = problem.calc_constraints_jacobian_at(x)

        #B = problem.calc_lagrangian_hessian_at(x, lambda_)
        B = sr1(problem, B, x, x_prev, lambda_)

        if kkt_fulfilled(problem, x, lambda_, c, tolerance):
            return x, i + 1

    raise TimeoutError(f"SQP ran into timeout with {MAX_ITER_SQP} steps")


def solve_quadratic(problem: QuadraticProblem) -> Tuple[np.ndarray, np.ndarray]:
    x_solution, _ = minimize_quadratic_problem(problem)
    lambda_ = find_lambda_of_qp(problem, x_solution)

    return x_solution, lambda_

# should be done
def find_mu(
        prev_mu: float, 
        f_grad: np.ndarray, 
        p: np.ndarray, 
        L_hessian: np.ndarray, 
        c_norm: float, 
        rho=0.5):

    hessian_quadratic_p = p @ L_hessian @ p
    sigma = 1 # hardcoded according to algorithm

    inequality_18_36 = (f_grad @ p + (sigma/2) * hessian_quadratic_p) / ((1-rho) * c_norm)
    if prev_mu is not None and prev_mu >= inequality_18_36:
        return prev_mu
    
    return inequality_18_36 + 1

# should be done (phi_1 function in book)
def l1_merit(
        f_x: float, 
        mu: float, 
        c_norm: float):
    return f_x + mu * c_norm

# should be done (D_1(phi_1) function in book)
def l1_merit_directional_gradient(
        f_grad: np.ndarray, 
        p: np.ndarray, 
        mu: float, 
        c_norm: float):
    return f_grad @ p - mu * c_norm


def find_lambda_of_qp(
        problem: QuadraticProblem,
        x_solution: np.ndarray) -> np.ndarray:

    lambda_ = np.zeros(len(problem.constraints))

    g = problem.G @ x_solution + problem.c
    active_constraint_mask = np.array([c.as_equality().holds(x_solution) for c in problem.constraints])
    if np.all(~active_constraint_mask):
        # no active constraints, all lambda=0
        return lambda_

    A, _ = combine_linear([c.c for c in problem.constraints[active_constraint_mask]])
    lambda_[active_constraint_mask] = np.linalg.lstsq(A.T, g)[0]

    return lambda_

# should be done
def transform_sqp_to_linear_constraint(
        constraint: Constraint, 
        c_i: float, 
        c_i_grad: np.ndarray):

    return LinearConstraint(
        c=LinearCallable(a=c_i_grad, b=-c_i),
        equation_type=constraint.equation_type)

# should be done
def create_iterate_quadratic_problem(
        problem: MinimizationProblem, 
        f_x: float, 
        f_grad: np.ndarray, 
        c: np.ndarray, 
        A: np.ndarray, 
        L_hessian: np.ndarray) -> QuadraticProblem:

    constraints = np.array([transform_sqp_to_linear_constraint(constraint, c_i, c_i_grad) for constraint, c_i, c_i_grad in zip(problem.constraints, c, A)])
    return QuadraticProblem(G=L_hessian, c=f_grad, n=len(f_grad), bias=f_x, constraints=constraints, x0=None, solution=None)

# should be done if meant that way?
def kkt_fulfilled(
        problem: MinimizationProblem, 
        x: np.ndarray, 
        lambda_: np.ndarray, 
        c: np.ndarray, 
        tolerance=1e-5):

    # 12.34d
    if np.any(lambda_ + tolerance < 0):
        return False
    
    for l_i, constraint, c_i in zip(lambda_, problem.constraints, c):
        # 12.34b
        if constraint.equation_type == EquationType.EQ:
            if np.any(abs(c_i) > tolerance):
                return False
        # 12.34c
        else:
            if np.any(c_i + tolerance < 0):
                return False

        #12.34e
        if np.any(abs(l_i * c_i ) > tolerance):
            return False
        
    l_gradient = problem.calc_lagrangian_gradient_at(x, lambda_)

    # 12.34a
    if np.any(abs(l_gradient) > tolerance):
        return False

    return True
