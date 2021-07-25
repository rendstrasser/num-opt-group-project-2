import pytest

import numpy as np
from shared.constraints import *
from shared.factorizations import qr_factorization_householder


def test_positivity_constraint_success_GE():
    constraint = LinearConstraint(LinearCallable(a=np.array([0, 1]), b=0), EquationType.GE)
    assert constraint.positivity_constraint_idx() == 1


def test_positivity_constraint_failure_wrong_b():
    constraint = LinearConstraint(LinearCallable(a=np.array([0, 1]), b=1), EquationType.GE)
    assert constraint.positivity_constraint_idx() is None


def test_positivity_constraint_failure_equality():
    constraint = LinearConstraint(LinearCallable(a=np.array([0, 1]), b=0), EquationType.EQ)
    assert constraint.positivity_constraint_idx() is None


def test_positivity_constraint_failure_GE_wrong_a_sign():
    constraint = LinearConstraint(LinearCallable(a=np.array([0, -1]), b=0), EquationType.GE)
    assert constraint.positivity_constraint_idx() is None


def test_positivity_constraint_success_LE():
    constraint = LinearConstraint(LinearCallable(a=np.array([0, -1]), b=0), EquationType.LE)
    assert constraint.positivity_constraint_idx() == 1


def test_positivity_constraint_failure_LE_wrong_a_sign():
    constraint = LinearConstraint(LinearCallable(a=np.array([0, 1]), b=0), EquationType.LE)
    assert constraint.positivity_constraint_idx() is None


def test_qr_factorization():
    np.random.seed(42)

    for _ in range(10):
        shape = np.random.randint(low=3, high=7, size=2)
        A = np.random.uniform(low=2, high=16, size=shape)
        Q, R = qr_factorization_householder(A)

        # ensure that QR factorization works
        assert np.allclose(A, Q @ R)
