import pytest

from simplex.linear_problem import *
from quadratic.quadratic_problem import *
from quadratic.base import minimize_quadratic_problem

from simplex.base import find_x0

def test_standard_form_positive_constraints_assumed():
    A = np.array([
        [2, 3],
        [4, 5]
    ])

    b = np.array([2, 3])

    lp = LinearProblem.from_positive_constrained_params(
        c=np.array([2, 5]),
        n=2,
        constraints=[
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), equality_type=InequalitySign.LESS_THAN_OR_EQUAL),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), equality_type=InequalitySign.LESS_THAN_OR_EQUAL),
        ]
    )

    standard_lp, _ = lp.to_standard_form()

    assert standard_lp.n == 4
    assert standard_lp.calc_f_at(np.array((1, 2, 1, 2))) == 12
    assert (standard_lp.calc_constraints_at(np.array((1, 2, 1, 2))) == np.array((7, 13))).all()

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
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), equality_type=InequalitySign.EQUAL),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), equality_type=InequalitySign.EQUAL),
        ],
        x0=None,
        solution=None
    )

    assert np.all(sp.A == A)
    assert np.all(sp.b == b)
    assert sp.f(np.array([1, 1])) == 7
    
def test_as_equality():
    c = LinearCallable(a=np.array([1, 2]), b=5)
    constraint = LinearConstraint(c, equality_type=InequalitySign.LESS_THAN_OR_EQUAL)
    assert constraint.as_equality().equality_type is InequalitySign.EQUAL

def test_standard_form_no_positive_constraints():
    A = np.array([
        [2, 3],
        [4, 5]
    ])

    b = np.array([2, 3])

    lp = LinearProblem(
        c=np.array([2, 5]),
        n=2,
        constraints=[
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), equality_type=InequalitySign.EQUAL),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), equality_type=InequalitySign.EQUAL),
        ],
        x0=None,
        solution=None
    )

    standard_lp, non_positive_constrained_indices = lp.to_standard_form()

    assert standard_lp.n == 4
    assert (non_positive_constrained_indices == np.array([0, 1])).all()
    assert standard_lp.calc_f_at(np.array((1, 2, 2, 1))) == 3
    assert (standard_lp.calc_constraints_at(np.array((1, 2, 1, 2))) == np.array((-2, -3))).all()


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
        return QuadraticProblem.from_params(G, c, A, b, solution=solution)

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
        x = minimize_quadratic_problem(sample_qp)
        assert np.all(np.isclose(x, sample_qp.solution))

    def test_phase_1_works_on_qp(self, sample_qp):
        x = find_x0(sample_qp, False)
        assert all(constraint.holds(x) for constraint in sample_qp.constraints)
