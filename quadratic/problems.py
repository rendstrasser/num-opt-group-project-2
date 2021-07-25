import numpy as np

from shared.constraints import LinearConstraint, LinearCallable, EquationType
from quadratic.quadratic_problem import QuadraticProblem

def create_exercise_example_16_1():
    """
    Exercise example 16.1 from the book.
    """
    G = np.array([[8, 2], [2, 2]])
    c = np.array([2, 3])

    a1 = np.array([1, -1], dtype=np.float64)
    a2 = np.array([1, 1], dtype=np.float64)
    a3 = np.array([1, 0], dtype=np.float64)
    constraints = np.array([
        LinearConstraint(LinearCallable(a=a1, b=0), equation_type=EquationType.GE),
        LinearConstraint(LinearCallable(a=a2, b=4), equation_type=EquationType.LE),
        LinearConstraint(LinearCallable(a=a3, b=3), equation_type=EquationType.LE)
    ])

    solution = np.array([1/6, -10/6], dtype=np.float64)

    return QuadraticProblem(G=G, c=c, n=2, constraints=constraints, x0=None, solution=solution)

def create_example_16_4():
    """
    Example 16.4 from the book.
    """
    G = np.array([[2, 0], [0, 2]])
    c = np.array([-2, -5])

    a1 = np.array([1, -2], dtype=np.float64)
    a2 = np.array([-1, -2], dtype=np.float64)
    a3 = np.array([-1, 2], dtype=np.float64)
    a4 = np.array([1, 0], dtype=np.float64)
    a5 = np.array([0, 1], dtype=np.float64)
    constraints = np.array([
        LinearConstraint(LinearCallable(a=a1, b=-2), equation_type=EquationType.GE),
        LinearConstraint(LinearCallable(a=a2, b=-6), equation_type=EquationType.GE),
        LinearConstraint(LinearCallable(a=a3, b=-2), equation_type=EquationType.GE),
        LinearConstraint(LinearCallable(a=a4, b=0), equation_type=EquationType.GE),
        LinearConstraint(LinearCallable(a=a5, b=0), equation_type=EquationType.GE),
    ])

    solution = np.array([1.4, 1.7], dtype=np.float64)

    return QuadraticProblem(G=G, c=c, n=2, constraints=constraints, x0=None, solution=solution)

def create_another_example():
    """
    Custom example for a quadratic problem.
    """
    G = np.array([[2, 0], [0, 2]])
    c = np.array([-4, -4])

    a1 = np.array([1, 1], dtype=np.float64)
    a2 = np.array([1, -2], dtype=np.float64)
    a3 = np.array([-1, -1], dtype=np.float64)
    a4 = np.array([-2, 1], dtype=np.float64)
    constraints = np.array([
        LinearConstraint(LinearCallable(a=a1, b=2), equation_type=EquationType.LE),
        LinearConstraint(LinearCallable(a=a2, b=2), equation_type=EquationType.LE),
        LinearConstraint(LinearCallable(a=a3, b=1), equation_type=EquationType.LE),
        LinearConstraint(LinearCallable(a=a4, b=2), equation_type=EquationType.LE),
    ])

    solution = np.array([1, 1], dtype=np.float64)

    return QuadraticProblem(G=G, c=c, n=2, constraints=constraints, x0=None, solution=solution)