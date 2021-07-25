import numpy as np
from shared.factorizations import cholesky_factorization, LU_factorization


def _forward_(L, b):
    """Performs forward-step for solving a linear system"""
    y = []
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i] = y[i] - (L[i,j]*y[j])
        y[i] = y[i]/L[i,i]

    return y


def _backward_(Lt, y):
    """Performs backward-step for solving a linear system"""
    x = np.zeros_like(y)
    for i in range(x.shape[0] , 0, -1):
        if Lt[i-1, i-1] == 0:
            raise np.linalg.LinAlgError("Matrix is singular")

        x[i - 1] = (y[i -1] - np.dot(Lt[i-1, i:],x[i:]))/Lt[i-1, i-1]

    return x


def solve_positive_definite(A, b) -> np.ndarray:
    """
    Solves Ax = b without computing the inverse of A but uses forward/backward substitution.
    The specified substitution uses the Cholesky factorization and hence can only be used for positive-definite matrices.

    Args:
        A: square, positive-definite matrix
        b: solution to linear system

    Returns:
        x (np.array with shape n x 1)
    """
    L = cholesky_factorization(A)
    y = _forward_(L, b)
    solution = _backward_(L.T, y)
    return solution


def solve(A, b) -> np.ndarray:
    """
    Solves Ax = b without computing the inverse of A but uses forward/backward substitution.
    The specified substitution uses the PLU factorization and hence can be used for arbitrary square matrices.

    Args:
        A: square matrix
        b: solution to linear system

    Returns:
        x (np.array with shape n x 1)
    """
    L, U, P = LU_factorization(A)
    y = _forward_(L, P @ b)
    solution = _backward_(U, y)

    return solution
