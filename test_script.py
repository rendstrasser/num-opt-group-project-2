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
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), equation_type=EquationType.LE),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), equation_type=EquationType.LE),
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
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), equation_type=EquationType.EQ),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), equation_type=EquationType.EQ),
        ],
        x0=None,
        solution=None
    )

    assert np.all(sp.A == A)
    assert np.all(sp.b == b)
    assert sp.f(np.array([1, 1])) == 7


def test_as_equality():
    c = LinearCallable(a=np.array([1, 2]), b=5)
    constraint = LinearConstraint(c, equation_type=EquationType.LE)
    assert constraint.as_equality().equation_type is EquationType.EQ


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
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), equation_type=EquationType.EQ),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), equation_type=EquationType.EQ),
        ],
        x0=None,
        solution=None
    )

    standard_lp, standardizing_meta_info = lp.to_standard_form()

    assert standard_lp.n == 4
    assert (standardizing_meta_info.indices_of_non_positive_constrained_vars == np.array([0, 1])).all()
    assert standard_lp.calc_f_at(np.array((1, 2, 2, 1))) == 3
    assert (standard_lp.calc_constraints_at(np.array((1, 2, 1, 2))) == np.array((-2, -3))).all()

