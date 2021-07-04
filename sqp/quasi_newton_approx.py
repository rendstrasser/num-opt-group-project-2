import numpy as np

# TODO in sqp.ipynb code, such that we can use one of these functions:
# - need to store previous x
# - initialize B with np.eye() for the first iteration


def damped_bfgs_updating(problem, B, x, x_old, l):
    """
    updates hessian according to procedure 18.2
    possibly could behave bad on difficult problems
    """

    s = x - x_old  # we would need to store x_old
    L_grad_x = problem.calc_lagrangian_gradient_at(x, l)
    L_grad_xold = problem.calc_lagrangian_gradient_at(x_old, l)
    y = L_grad_x - L_grad_xold

    s_B_s = np.inner(np.inner(s, B), s)

    if np.inner(s, y) >= 0.2 * s_B_s:
        theta = 1
    else:
        theta = (0.8 * s_B_s) / (s_B_s - np.inner(s, y))

    r = theta * y + (1 - theta) * (B @ s)

    B_new = B - (np.outer(B @ s, s) @ B) / s_B_s + np.outer(r, r) * 1 / np.inner(s, r)

    return B_new


def sr1(problem, B, x, x_old, l, delta=0.1):
    """
    updates hessian according to SR1 (6.24)
    S1 alone doesn't guarantee positive definiteness: adds multiple of identity to hessian if this case is encountered
    """

    s = x - x_old  # we would need to store x_old
    L_grad_x = problem.calc_lagrangian_gradient_at(x, l)
    L_grad_xold = problem.calc_lagrangian_gradient_at(x_old, l)
    y = L_grad_x - L_grad_xold

    y_minus_B_s = y - B @ s
    B_new = B + np.outer(y_minus_B_s,y_minus_B_s) / np.inner((y_minus_B_s), s)

    # according to page 538, adding sufficiently large multiple of identity
    # while criterion might not be sufficient enough (but hopefully is)
    n = B.shape[0]
    while not np.all(np.linalg.eigvals(B_new) > 0):
        B_new += delta * np.eye(n)

    return B_new
