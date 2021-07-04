import pytest
from shared import constraints
from shared.constraints import *
from simplex.simplex_problem import *
import numpy as np

def test_combined_params():
    
    A = np.array([
        [2, 3],
        [4, 5]
    ])

    b = np.array([2, 3])

    sp = SimplexProblem(constraints=[
        LinearConstraint(c=LinearCallable(a=A[0], b=b[0]), is_equality=True),
        LinearConstraint(c=LinearCallable(a=A[1], b=b[1]), is_equality=True),
    ],
    c = np.array([2, 5]),
    x0='whatever',
    solutions=[]
    )

    assert np.all(sp.A == A)
    assert np.all(sp.b == b)
    assert sp.f(np.array([1, 1])) == 7

    

