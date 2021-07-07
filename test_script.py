import pytest
import numpy as np

from shared import constraints
from shared.constraints import *
from simplex.linear_problem import *
from quadratic.quadratic_problem import *


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


def test_combined_params_quadratic():
    A = np.array([
        [2, 3],
        [4, 5]
    ])

    b = np.array([2, 3])

    qp = QuadraticProblem(
        G=np.array([[2, 5], [5, 2]]),
        c=np.array([1, 2]),
        n=2,
        constraints=[
            LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), is_equality=True),
            LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), is_equality=True),
        ],
        x0=None,
        solution=None
    )

    assert np.all(qp.A == A)
    assert np.all(qp.b == b)
    assert qp.f(np.array([1, 1])) == 10


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
