import numpy as np

def cholesky_factorization(H):
    """
    factorizes input matrix in lower triangular matrix
    :param H: has to me a square matrix
    :return: lower triangular matrix
    """
    assert H.shape[0] == H.shape[1]

    N = H.shape[1]

    L = np.zeros_like(H, dtype=np.float64)

    for i in range(N):
        for j in range(i+1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(np.max(H[i, i] - s, 0))
            else:
                L[i, j] = (1 / L[j, j]) * (H[i, j] - s)


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


def solve(H, b):
    """
    solves Hx = b without computing the inverse of H but uses forward/backward substitution and cholesky factorization
    :param H: square matrix
    :param b: solution to linear system
    :return: x (np.array with shape n x 1)
    """
    L = cholesky_factorization(H)
    y = forward(L, b)
    solution = backward(L.T, y)

    return solution