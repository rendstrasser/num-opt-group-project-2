import numpy as np

from shared.minimization_problem import LinearConstraintsProblem
from shared.constraints import LinearConstraint, LinearCallable
from simplex.linear_problem import LinearProblem

def create_phase_I_problem(problem: LinearConstraintsProblem):
    n = problem.n
    m = len(problem.constraints)

    e_x = np.zeros(shape=n)
    e_z = np.ones(shape=m)
    e = np.concatenate((e_x, e_z))

    x0 = np.zeros(n)
    z0 = np.abs(problem.b)
    xz0 = np.concatenate((x0, z0))

    constraints = []
    for i, constraint in enumerate(problem.constraints):
        E_i = np.eye(m)[i]
        if problem.b[i] < 0:
            E_i = -E_i

        A_i = problem.a
        a = np.concatenate(A_i, E_i)

        constraints.append(LinearConstraint(c=LinearCallable(a=a, b=constraint.b), is_equality=constraint.is_equality))

    return LinearProblem(c=e, constraints=constraints, x0=xz0)