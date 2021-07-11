import numpy as np
from typing import Tuple

from simplex.linear_problem import LinearProblem
from shared.minimization_problem import LinearConstraintsProblem


def minimize_linear_problem(original_problem: LinearProblem, standardized=False) -> Tuple[np.ndarray, int]:
    """
    Minimizes the given linear problem. Supports not auto-standardizing when 'standardized=False', otherwise 
    expects a standardized problem input.
    """

    # standardize problem
    if standardized:
        problem = original_problem
        non_positive_constrained_indices = []
    else:
        problem, non_positive_constrained_indices = original_problem.to_standard_form()

    m = len(problem.b) # number of constraints -> size of basis
    
    # starting the simplex method
    x0 = find_x0(problem, True)

    # the basis consists of x0 elements which are not 0. all others should be 0
    x0_args_sorted = np.argsort(x0)
    basis = x0_args_sorted[-m:]
    basis = np.sort(basis)

    non_basis = x0_args_sorted[:-m]
    non_basis = np.sort(non_basis)
    
    # algorithm 13.1
    i = 0
    while True:
        B = problem.A[:, basis]
        N = problem.A[:, non_basis]

        c_b = problem.c[basis]
        c_n = problem.c[non_basis]

        B_inv = np.linalg.inv(B)

        if i == 0: # only necessary for initial run
            x_b = B_inv @ problem.b

        lambda_ = B_inv.T @ c_b
        s_n = c_n - N.T @ lambda_

        if np.all(s_n >= 0):
            # optimal point found
            x = np.zeros_like(x0)
            x[basis] = x_b
            x = destandardize_x(original_problem, x, non_positive_constrained_indices)
            return x, i+1

        q = non_basis[np.argmin(s_n)]
        A_q = problem.A[:, q].flatten()
        d = B_inv @ A_q

        if np.all(d <= 0):
            raise ValueError("Problem is unbounded")

        # find p
        xb_div_d = x_b[d > 0]/d[d > 0]
        min_div_idx = np.argmin(xb_div_d)
        basis_idx = np.argwhere(d > 0)[min_div_idx]
        p = basis[basis_idx]

        # update basis/non-basis
        remaining_basis_items_mask = basis != p
        basis = basis[remaining_basis_items_mask]
        basis = np.append(basis, q)
        basis = np.sort(basis)
        q_idx_in_basis = np.where(basis==q)[0]
        
        non_basis = non_basis[non_basis != q]
        non_basis = np.append(non_basis, p)
        non_basis = np.sort(non_basis)

        # update x_b according to x_q^plus and x_b^plus in procedure 13.1
        # to avoid a usage of an inverted B and matrix multiplication
        x_q = xb_div_d[min_div_idx]
        x_b -= d * x_q
        x_b = x_b[remaining_basis_items_mask]
        x_b = np.insert(x_b, q_idx_in_basis, x_q)

        i += 1

def destandardize_x(original_problem: LinearProblem, x: np.ndarray, non_positive_constrained_indices: np.ndarray):
    """
    Destandardizes x based on the original problem
    """
    n = original_problem.n

    x_plus = x[:n]
    x_neg = x[n:n + len(non_positive_constrained_indices)]

    x_plus[non_positive_constrained_indices] -= x_neg

    return x_plus


def find_x0(problem: LinearConstraintsProblem, standardized: bool):
    if problem.x0 is not None:
        return problem.x0

    # Phase I approach, if no x0 is given

    phase_I_problem, non_positive_constrained_indices, slack_var_count = LinearProblem.phase_I_problem_from(problem, standardized)

    n = problem.n + len(non_positive_constrained_indices) + slack_var_count # n of standardized constraints problem
    xz0, _ = minimize_linear_problem(phase_I_problem, standardized=True)

    x0 = xz0[:n]
    z0 = xz0[n:]

    if np.any(np.absolute(z0) > 1e-4):
        # no solution!
        raise ValueError("Problem has no solution!")

    x0 = destandardize_x(problem, x0, non_positive_constrained_indices)
    
    return x0
