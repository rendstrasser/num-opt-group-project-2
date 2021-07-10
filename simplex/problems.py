import numpy as np

from simplex.linear_problem import LinearProblem, LinearCallable, LinearConstraint

def create_example_13_1_problem():
    c = np.array([-4, -2, 0, 0], dtype=np.float64)

    a1 = np.array([1, 1, 1, 0], dtype=np.float64)
    a2 = np.array([2, 1/2, 0, 1], dtype=np.float64)
    constraints = np.array([
        LinearConstraint(LinearCallable(a=a1, b=5), is_equality=True),
        LinearConstraint(LinearCallable(a=a2, b=8), is_equality=True),
    ])

    x0 = np.array([0, 0, 5, 8], dtype=np.float64)
    solution = np.array([11/3, 4/3, 0, 0], dtype=np.float64)

    return LinearProblem(c=c, n=4, constraints=constraints, x0=x0, solution=solution)

def create_another_example_1():
    """
    Example 1 from https://realpython.com/linear-programming-python/
    """ 

    c = np.array([-1, -2], dtype=np.float64)

    a1 = np.array([2, 1], dtype=np.float64)
    a2 = np.array([-4, 5], dtype=np.float64)
    a3 = np.array([1, -2], dtype=np.float64)
    a4 = np.array([-1, 5], dtype=np.float64)
    constraints = np.array([
        LinearConstraint(LinearCallable(a=a1, b=20), is_equality=False),
        LinearConstraint(LinearCallable(a=a2, b=10), is_equality=False),
        LinearConstraint(LinearCallable(a=a3, b=2), is_equality=False),
        LinearConstraint(LinearCallable(a=a4, b=15), is_equality=True),
    ])

    x0 = None
    solution = np.array([85/11, 50/11], dtype=np.float64)

    return LinearProblem(c=c, n=2, constraints=constraints, x0=x0, solution=solution)

def create_another_example_2():
    """
    Example 2 from https://realpython.com/linear-programming-python/
    """ 

    c = np.array([-20, -12, -40, -25], dtype=np.float64)

    a1 = np.array([1, 1, 1, 1], dtype=np.float64)
    a2 = np.array([3, 2, 1, 0], dtype=np.float64)
    a3 = np.array([0, 1, 2, 3], dtype=np.float64)
    constraints = np.array([
        LinearConstraint(LinearCallable(a=a1, b=50), is_equality=False),
        LinearConstraint(LinearCallable(a=a2, b=100), is_equality=False),
        LinearConstraint(LinearCallable(a=a3, b=90), is_equality=False)
    ])

    x0 = None
    solution = np.array([5, 0, 45, 0], dtype=np.float64)

    return LinearProblem(c=c, n=4, constraints=constraints, x0=x0, solution=solution)