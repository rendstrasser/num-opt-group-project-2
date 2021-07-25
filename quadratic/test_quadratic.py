import numpy as np
import pytest

from dataclasses import asdict

from quadratic.base import minimize_quadratic_problem
from quadratic.problems import create_exercise_example_16_1, create_another_example, create_example_16_4
from quadratic.quadratic_problem import QuadraticProblem
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
    A = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    b = np.array([1, 3, 0, 0])
    solution = np.array([1, 3])
    return G, c, A, b, solution


@pytest.fixture
def sample_ineq_qp(sample_ineq_qp_params) -> QuadraticProblem:
    G, c, A, b, solution = sample_ineq_qp_params
    equation_type_vec = np.repeat(EquationType.GE, 4)
    return QuadraticProblem.from_params(G, c, A, b, equation_type_vec, solution=solution)


@pytest.fixture(params=
[
    create_example_16_4,
    create_another_example,
    create_exercise_example_16_1
])
def other_problem(request):
    return request.param()


class TestPhase1:

    def test_phase_1_works_on_qp(self, sample_qp):
        x = sample_qp.find_x0()
        assert sample_qp.is_feasible(x)

    def test_phase_1_ineq(self, sample_ineq_qp):
        x = sample_ineq_qp.find_x0()
        assert all(constraint.holds(x) for constraint in sample_ineq_qp.constraints)

    @pytest.mark.parametrize('initial_guess', [
        [2, 2],  # Try out multiple starting points.
        [5, 5],
        [0, 0]
    ])
    def test_with_initial_guess(self, sample_ineq_qp, initial_guess):
        x = sample_ineq_qp.find_x0(original_initial_guess=np.array(initial_guess))
        assert sample_ineq_qp.is_feasible(x)

    def test_other_problems(self, other_problem):
        initial_guess = np.random.random(len(other_problem.solution))
        assert other_problem.is_feasible(other_problem.find_x0(initial_guess))


class TestSolving:

    def test_linearly_dependent_constraints(self, ):
        problem = create_another_example()

        x, _ = minimize_quadratic_problem(problem)

        assert np.all(np.isclose(x, problem.solution))

    def test_other_problems(self, other_problem):
        x, _ = minimize_quadratic_problem(other_problem)
        assert np.all(np.isclose(x, other_problem.solution))

    def test_solve_equality_problem(self, sample_qp):
        x, _ = minimize_quadratic_problem(sample_qp)
        assert np.all(np.isclose(x, sample_qp.solution))

    def test_ineq_qp(self, sample_ineq_qp):
        x, _ = minimize_quadratic_problem(sample_ineq_qp)
        assert np.all(np.isclose(x, sample_ineq_qp.solution))


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
