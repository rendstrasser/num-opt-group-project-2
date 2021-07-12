import numpy as np

def cholesky_factorization(A):
    """
    factorizes input matrix in lower triangular matrix, requires the matrix to be symmetric and positive definite
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


def LU_factorization(A: np.ndarray):
    """
    factorizes a square matrix, matrix does not have to be symmetric or positive definite
    use as A = LU
    :param A: matrix to decompose
    :return: lower matrix L and upper matrix U
    """
    assert A.shape[0] == A.shape[1]  # requires square matrix

    n = A.shape[0]

    # pivot matrix
    p = np.eye(n)
    for j in range(n):
        max_row = j + np.argmax(np.abs(A[j:n, j]))
        if j != max_row:  # swap rows if off-diagonal element is bigger in j-th column
            p[[max_row, j]] = p[[j, max_row]]

    L = np.zeros_like(A, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)
    PA = np.matmul(p, A)

    for i in range(n):
        # calculate U
        for j in range(i,n):
            s = 0
            for k in range(i):
                s += L[i, k] * U[k, j]
            U[i, j] = PA[i, j] - s

        # calculate L
        for j in range(i, n):
            if j == i:
                L[i, j] = 1
            else:
                s = 0
                for k in range(i):
                    s += L[j, k] * U[k, i]
                L[j, i] = (PA[j, i] - s) / U[i, i]

    return L, U, p


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


def solve(A, b, cholesky = False):
    """
    solves Ax = b without computing the inverse of A but uses forward/backward substitution and the specified substitution
    by default uses PLU factorization and hence can be used for arbitrary square matrices
    (! attention: might fail if a column has only negative and 0 entries !, if needed @Franzi modifies it)
    :param A: square matrix
    :param b: solution to linear system
    :param cholesky: if set to false, PLU factorization is used. use cholesky only for positive definite matrices, else LU
    :return: x (np.array with shape n x 1)
    """
    if cholesky:
        L = cholesky_factorization(A)
        y = forward(L, b)
        solution = backward(L.T, y)

    else:
        L, U, P = LU_factorization(A)
        y = forward(L, P @ b)
        solution = backward(U, y)

    return solution
