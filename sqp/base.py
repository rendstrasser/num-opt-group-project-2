
import numpy as np

from shared.minimization_problem import MinimizationProblem
from shared.constraints import Constraint, LinearCallable, LinearConstraint
from quadratic.quadratic_problem import QuadraticProblem
from typing import Tuple

# should be done
def backtracking_line_search_sqp(
        problem: MinimizationProblem, 
        eta=0.3, tau=0.5, tolerance=1e-5) -> Tuple[np.ndarray, np.ndarray]:

    assert problem.x0 is not None

    x = problem.x0
    lambda_ = np.zeros(shape=len(problem.constraints))

    f_x = problem.calc_f_at(x)
    f_grad = problem.calc_gradient_at(x)
    c = problem.calc_constraints_at(x)
    c_norm = np.linalg.norm(c, ord=1)
    A = problem.calc_constraints_jacobian_at(x)

    # TODO quasi-newton approx 
    L_hessian = problem.calc_lagrangian_hessian_at(x, lambda_)

    while True:
        quadr_problem = create_iterate_quadratic_problem(problem, f_x, f_grad, c, A, L_hessian)
        p, l_hat = solve_quadratic(quadr_problem)
        p_lambda = l_hat - lambda_
        mu = find_mu()

        merit_at_x = l1_merit(x, mu, c_norm)
        dir_gradient_merit = l1_merit_directional_gradient(f_grad, p, mu, c_norm)
        first_wolfe_condition_term = merit_at_x + eta * alpha * dir_gradient_merit

        alpha = 1
        while l1_merit(x + alpha * p, mu, c_norm) > first_wolfe_condition_term:
            alpha *= tau
        
        x += alpha * p
        lambda_ += alpha * p_lambda

        f_x = problem.calc_f_at(x)
        f_grad = problem.calc_gradient_at(x)
        c = problem.calc_constraints_at(x)
        c_norm = np.linalg.norm(c, ord=1)
        A = problem.calc_constraints_jacobian_at(x)

        # TODO quasi-newton approx 
        L_hessian = problem.calc_lagrangian_hessian_at(x, lambda_)

        if kkt_fulfilled(problem, x, lambda_, c, tolerance):
            break

    return x, lambda_

# TODO implement quadratic minimization
def solve_quadratic(problem):
    p = (1,1,1,1,1)
    l = (0,0,0)
    return p, l

# should be done
def find_mu(
        prev_mu: float, 
        f_grad: np.ndarray, 
        p: np.ndarray, 
        L_hessian: np.ndarray, 
        c_norm: float, 
        rho=0.5):

    hessian_quadratic_p = p @ L_hessian @ p
    sigma = 1 if hessian_quadratic_p > 0 else 0

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

# should be done
def transform_sqp_to_linear_constraint(
        constraint: Constraint, 
        c_i: float, 
        c_i_grad: np.ndarray):

    return LinearConstraint(
        c=LinearCallable(a=c_i_grad, b=c_i), 
        equality=constraint.equality)

# should be done
def create_iterate_quadratic_problem(
        problem: MinimizationProblem, 
        f_x: float, 
        f_grad: np.ndarray, 
        c: np.ndarray, 
        A: np.ndarray, 
        L_hessian: np.ndarray):

    constraints = np.array([transform_sqp_to_linear_constraint(constraint, c_i, c_i_grad) for constraint, c_i, c_i_grad in zip(problem.constraints, c, A)])
    return QuadraticProblem(G=L_hessian, c=f_grad, bias=f_x, constraints=constraints, x0=None, solution=None)

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
        if constraint.equality:
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
