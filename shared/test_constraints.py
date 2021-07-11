import pytest

from shared.constraints import *

def test_positivity_constraint_success_GE():
    constraint =  LinearConstraint(LinearCallable(a=np.array([0,1]), b=0), EquationType.GE)
    assert constraint.positivity_constraint_idx() == 1

def test_positivity_constraint_failure_wrong_b():
    constraint =  LinearConstraint(LinearCallable(a=np.array([0,1]), b=1), EquationType.GE)
    assert constraint.positivity_constraint_idx() is None

def test_positivity_constraint_failure_equality():
    constraint =  LinearConstraint(LinearCallable(a=np.array([0,1]), b=0), EquationType.EQ)
    assert constraint.positivity_constraint_idx() is None

def test_positivity_constraint_failure_GE_wrong_a_sign():
    constraint =  LinearConstraint(LinearCallable(a=np.array([0,-1]), b=0), EquationType.GE)
    assert constraint.positivity_constraint_idx() is None

def test_positivity_constraint_success_LE():
    constraint =  LinearConstraint(LinearCallable(a=np.array([0,-1]), b=0), EquationType.LE)
    assert constraint.positivity_constraint_idx() == 1

def test_positivity_constraint_failure_LE_wrong_a_sign():
    constraint =  LinearConstraint(LinearCallable(a=np.array([0,1]), b=0), EquationType.LE)
    assert constraint.positivity_constraint_idx() is None
