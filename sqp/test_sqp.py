import pytest

import numpy as np

from shared.minimization_problem import MinimizationProblem
from shared.constraints import LinearConstraint, LinearCallable, EquationType
from sqp.base import minimize_nonlinear_problem
from quadratic.problems import create_exercise_example_16_1
from sqp.problems import create_example_18_3_problem


def test_qp_as_sqp_solved():
    pytest.skip()

    problem = create_exercise_example_16_1()

    x, _ = minimize_nonlinear_problem(problem)

def test_example_18_3():
    pytest.skip()

    problem = create_example_18_3_problem()

    x, _ = minimize_nonlinear_problem(problem)