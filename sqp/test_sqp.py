import pytest

import numpy as np

from shared.minimization_problem import MinimizationProblem
from shared.constraints import LinearConstraint, LinearCallable, EquationType
from sqp.base import minimize_nonlinear_problem
from quadratic.problems import create_exercise_example_16_1
from sqp.problems import create_example_18_3_problem, create_convex_in_f_problem


def test_qp_as_sqp_solved():
    # slow, activate on-demand
    # pytest.skip()

    problem = create_exercise_example_16_1()

    x, _ = minimize_nonlinear_problem(problem, max_iter=100_000)

    assert (np.allclose(x, problem.solution, atol=1e-1, rtol=1e-1))

def test_convex():
    problem = create_convex_in_f_problem()

    x, iter_count = minimize_nonlinear_problem(problem)

    assert (np.allclose(x, problem.solution, atol=1e-3, rtol=1e-3))

def test_example_18_3():
    # seems to be ill-conditioned for our purpose
    #pytest.skip()

    problem = create_example_18_3_problem()

    x, _ = minimize_nonlinear_problem(problem, max_iter=100)

    assert (np.allclose(x, problem.solution))