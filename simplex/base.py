import numpy as np

from simplex.linear_problem import LinearProblem


def minimize_linear_problem(problem: LinearProblem) -> np.ndarray:
    """
    A problem - assumed to be in standard form - is optimized.
    """

    x = find_x0(problem)
    m = len(problem.b) # number of constraints -> size of basis

    # the basis consists of x0 elements which are not 0. all others should be 0
    x0_args_sorted = np.argsort(x)
    basis = np.argsort(x)[-m:]
    
    # TODO implement algorithm 13.1
    while True:
        break

    return x


def find_x0(problem: LinearProblem):
    if problem.x0 is not None:
        return problem.x0

    # Phase I approach, if no x0 is given

    phase_I_problem = LinearProblem.phase_I_problem_from(problem)
    xz0 = minimize_linear_problem(phase_I_problem)

    x0 = xz0[:problem.n]
    z0 = xz0[problem.n:]

    if np.any(np.absolute(z0) > 1e-4):
        # no solution!
        raise ValueError("Problem has no solution!")
    
    return x0