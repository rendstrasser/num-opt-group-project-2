"""
File with functions and routines to construct MinimizationProblems.
"""

import numpy as np

from shared.constraints import Constraint, EquationType
from shared.minimization_problem import MinimizationProblem


def create_example_18_3_problem():
    """
    Example 18.3 from the book
    """

    def f(x):
        return np.e ** (np.prod(x)) - 1 / 2 * (x[0] ** 3 + x[1] ** 3 + 1) ** 2

    def c1(x):
        return np.sum(x ** 2) - 10

    def c2(x):
        return x[1] * x[2] - 5 * x[3] * x[4]

    def c3(x):
        return x[0] ** 3 + x[1] ** 3 + 1

    x0 = np.array((-1.71, 1.59, 1.82, -0.763, -0.763))
    solution = np.array((-1.8, 1.7, 1.9, -0.8, -0.8))
    constraints = np.array((Constraint(c1, equation_type=EquationType.EQ),
                            Constraint(c2, equation_type=EquationType.EQ),
                            Constraint(c3, equation_type=EquationType.EQ)))

    return MinimizationProblem(f=f, n=len(solution), constraints=constraints, solution=solution, x0=x0)


def create_convex_in_f_problem() -> MinimizationProblem:
    """
    Custom example for a problem that is non-linear in the constraints but convex in the objective.
    """

    def f(x):
        return x[0]**2 + x[1]**2

    def c1(x):
        return x[0]/(1+x[1]**2)

    def c2(x):
        return (x[0]+x[1])**2

    x0 = None
    solution = np.array((0, 0))
    constraints = np.array((Constraint(c1, equation_type=EquationType.LE),
                            Constraint(c2, equation_type=EquationType.EQ)))

    return MinimizationProblem(f=f, n=len(solution), constraints=constraints, solution=solution, x0=x0)

def create_made_up_problem_1():

    def f(x):
        return np.e ** (-0.5*(x[0]**2+x[1]**2))+x[2]**2

    def c1(x):
        return x[0]+x[1]-0.5
    def c2(x):
        return x[2]**3+x[0]-10

    x0 = np.array((11.,10.,9.))
    solution = np.array((10.,-9.5,0))
    constraints = np.array((Constraint(c1, equation_type=EquationType.EQ),
                            Constraint(c2, equation_type=EquationType.EQ)))

    return MinimizationProblem(f=f, n=len(x0), constraints=constraints, solution=solution, x0=x0)

def create_made_up_problem_2():
    """
    takes a little bit longer to compute, but is solved by the algorithm!
    shows that we have good local convergence even for non convex problems
    """
    def f(x):
        return x[0]**1.1*(x[1])**2

    def c1(x):
        return np.e**x[0]-5
    def c2(x):
        return (x[0]-x[1])*x[0]-2




    x0 = np.array((1.7,0.5))
    solution = np.array((1.6096558 , 0.36709701))
    constraints = np.array((Constraint(c1, equation_type=EquationType.EQ),
                            Constraint(c2, equation_type=EquationType.EQ)))

    return MinimizationProblem(f=f, n=len(x0), constraints=constraints, solution=solution, x0=x0)


def create_made_up_problem_3():

    def f(x):
        return (x[0]-2)**2+(x[1]-4)**4+(x[2]+3.3)**6


    def c1(x):
        return x[1]+x[2]-3
    def c2(x):
        return x[0]**2-3




    x0 = np.array((1.,1.,1.))
    solution = np.array((2, 5.247203969423337074659622658, -2.247203969423337074659622658))
    constraints = np.array((Constraint(c1, equation_type=EquationType.EQ),
                            Constraint(c2, equation_type=EquationType.GE)))

    return MinimizationProblem(f=f, n=len(x0), constraints=constraints, solution=solution, x0=x0)

