import numpy as np
import pytest

from quadratic.base import minimize_quadratic_problem
from quadratic.quadratic_problem import QuadraticProblem
from simplex.base import find_x0
from shared.constraints import EquationType


@pytest.fixture
def sample_qp_params():
    """Sample parameters of example 16.2 in the book."""
    G = np.array([
        [6, 2, 1],
        [2, 5, 2],
        [1, 2, 4]
    ])

    c = np.array([-8, -3, -3])

    A = np.array([
        [1, 0, 1],
        [0, 1, 1]
    ])

    b = np.array([
        3, 0
    ])

    solution = np.array([2, -1, 1])

    return G, c, A, b, solution


@pytest.fixture
def sample_qp(sample_qp_params) -> QuadraticProblem:
    """Fixture returning Example 16.2."""
    G, c, A, b, solution = sample_qp_params
    return QuadraticProblem.from_params(G, c, A, b, solution=solution)


@pytest.fixture
def sample_ineq_qp_params() -> tuple:
    G = np.eye(2)
    c = np.zeros(2)
    A = np.eye(2)
    b = np.array([1, 3])
    solution = b
    return G, c, A, b, solution


@pytest.fixture
def sample_ineq_qp(sample_ineq_qp_params) -> QuadraticProblem:
    G, c, A, b, solution = sample_ineq_qp_params
    equation_type_vec = np.repeat(EquationType.GE, 2)
    return QuadraticProblem.from_params(G, c, A, b, equation_type_vec, solution=solution)


def test_ineq_qp(sample_ineq_qp):
    x = minimize_quadratic_problem(sample_ineq_qp)
    assert  np.all(np.isclose(x, sample_ineq_qp.solution))


def test_combined_params(sample_qp, sample_qp_params):
    G, c, A, b, _ = sample_qp_params
    assert np.all(sample_qp.A == A)
    assert np.all(sample_qp.b == b)
    assert sample_qp.f(np.array([1, 1, 1])) == -1.5


def test_no_indefinite_G(sample_qp_params):
    G, c, A, b, _ = sample_qp_params
    G = np.diag([1, -1, 1])
    with pytest.raises(ValueError):
        QuadraticProblem.from_params(G, c, A, b)


def test_solve_equality_problem(sample_qp):
    x = minimize_quadratic_problem(sample_qp)
    assert np.all(np.isclose(x, sample_qp.solution))


def test_phase_1_works_on_qp(sample_qp):
    x = find_x0(sample_qp, standardized=False)
    assert all(constraint.holds(x) for constraint in sample_qp.constraints)

def test_phase_1_ineq(sample_ineq_qp):
    x = find_x0(sample_ineq_qp, standardized=False)
    assert all(constraint.holds(x) for constraint in sample_ineq_qp.constraints)
