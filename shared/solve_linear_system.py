import numpy as np

def cholesky_factorization(A):
    """
    factorizes input matrix in lower triangular matrix
    :param A: has to me a square matrix
    :return: lower triangular matrix
    """
    assert A.shape[0] == A.shape[1]

    N = A.shape[1]

    L = np.zeros_like(A, dtype=np.float64)

    for i in range(N):
        for j in range(i+1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(np.max(A[i, i] - s, 0))
            else:
                L[i, j] = (1 / L[j, j]) * (A[i, j] - s)


    return L


def forward(L, b):
    y = []
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i] = y[i] - (L[i,j]*y[j])
        y[i] = y[i]/L[i,i]

    return y


def backward(Lt, y):
    x = np.zeros_like(y)
    for i in range(x.shape[0] , 0, -1):
        x[i - 1] = (y[i -1] - np.dot(Lt[i-1, i:],x[i:]))/Lt[i-1, i-1]

    return x


def solve(A, b):
    """
    solves Ax = b without computing the inverse of A but uses forward/backward substitution and cholesky factorization
    :param A: square matrix
    :param b: solution to linear system
    :return: x (np.array with shape n x 1)
    """
    L = cholesky_factorization(A)
    y = forward(L, b)
    solution = backward(L.T, y)

    return solution