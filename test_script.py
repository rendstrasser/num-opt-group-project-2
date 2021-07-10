import pytest
import numpy as np

from shared import constraints
from shared.constraints import *
from shared.solve_linear_system import solve
from simplex.linear_problem import *
from quadratic.quadratic_problem import *
from quadratic.base import minimize_quadratic_problem


def test_combined_params_linear():
    A = np.array([
        [2, 3],
        [4, 5]
    ])

    b = np.array([2, 3])

    sp = LinearProblem(
        c=np.array([2, 5]),
        n=2,
        constraints=[
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), is_equality=True),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), is_equality=True),
        ],
        x0=None,
        solution=None
    )

    assert np.all(sp.A == A)
    assert np.all(sp.b == b)
    assert sp.f(np.array([1, 1])) == 7


def test_standard_form():
    A = np.array([
        [2, 3],
        [4, 5]
    ])

    b = np.array([2, 3])

    lp = LinearProblem(
        c=np.array([2, 5]),
        n=2,
        constraints=[
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), is_equality=True),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), is_equality=True),
        ],
        x0=None,
        solution=None
    )

    standard_lp = lp.to_standard_form()

    assert standard_lp.n == 6
    assert standard_lp.calc_f_at(np.array((1, 2, 2, 1, 1, 2))) == 3
    assert (standard_lp.calc_constraints_at(np.array((1, 2, 2, 1, 1, 2))) == np.array((0, 0))).all()


class TestQuadratic:

    @pytest.fixture(scope='class')
    def sample_qp_params(self):
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

    @pytest.fixture(scope='class')
    def sample_qp(self, sample_qp_params) -> QuadraticProblem:
        """Fixture returning Example 16.2."""
        G, c, A, b, solution = sample_qp_params
        return QuadraticProblem.from_params(G, c, A, b, solution)

    def test_combined_params(self, sample_qp, sample_qp_params):
        G, c, A, b, _ = sample_qp_params
        assert np.all(sample_qp.A == A)
        assert np.all(sample_qp.b == b)
        assert sample_qp.f(np.array([1, 1, 1])) == -1.5

    def test_no_indefinite_G(self, sample_qp_params):
        G, c, A, b, _ = sample_qp_params
        G = np.diag([1, -1, 1])
        with pytest.raises(ValueError):
            QuadraticProblem.from_params(G, c, A, b)

    def test_solve_equality_problem(self, sample_qp):
        pytest.skip()

        x = minimize_quadratic_problem(sample_qp)
        assert np.isclose(x, sample_qp.solution)

    def test_find_nullspace_matrix(self, sample_qp_params):
        G, c, A, b, _ = sample_qp_params
        print(solve(A, np.zeros(len(A[0]))))

