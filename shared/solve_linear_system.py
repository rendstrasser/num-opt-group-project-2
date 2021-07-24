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


def LU_factorization(A: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    factorizes a square matrix, matrix does not have to be symmetric or positive definite
    use as A = p @ L @ U
    :param A: matrix to decompose
    :return: lower matrix L, upper matrix U, permutation matrix p
    """
    assert A.shape[0] == A.shape[1]  # requires square matrix

    n = A.shape[0]

    U = A.copy()
    L = np.eye(n, dtype=np.double)
    p = np.eye(n, dtype=np.double)

    # Loop over rows
    for i in range(n):

        # adjust permutation matrix
        if np.isclose(U[i, i], 0.0):
            max_idxs = A[i:n, i].argsort()[::-1]

            # loop through indices sorted according to descending values, take first index that is non-zero
            j = 0
            while A[i:n, i][max_idxs[j]] == 0:
                if j == (len(max_idxs) - 1):
                    raise ValueError("PLU factorization isn't possible as there is a column with only 0 entries.")
                j += 1

            max_row = i + max_idxs[j]
            if i != max_row:  # swap rows if off-diagonal element is bigger in j-th column
                p[[max_row, i]] = p[[i, max_row]]
                U[[max_row, i]] = U[[i, max_row]]

        # perform operations
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    return L, U, p


def _forward_(L, b):
    y = []
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i] = y[i] - (L[i,j]*y[j])
        y[i] = y[i]/L[i,i]

    return y


def _backward_(Lt, y):
    x = np.zeros_like(y)
    for i in range(x.shape[0] , 0, -1):
        x[i - 1] = (y[i -1] - np.dot(Lt[i-1, i:],x[i:]))/Lt[i-1, i-1]

    return x


def solve(A, b, cholesky = False) -> np.ndarray:
    """
    solves Ax = b without computing the inverse of A but uses forward/backward substitution and the specified substitution
    by default uses PLU factorization and hence can be used for arbitrary square matrices
    (! attention: tell @Franzi if it fails, she modifies it)
    :param A: square matrix
    :param b: solution to linear system
    :param cholesky: if set to false, PLU factorization is used. use cholesky only for positive definite matrices, else LU
    :return: x (np.array with shape n x 1)
    """
    if cholesky:
        L = cholesky_factorization(A)
        y = _forward_(L, b)
        solution = _backward_(L.T, y)

    else:
        L, U, P = LU_factorization(A)
        y = _forward_(L, P @ b)
        solution = _backward_(U, y)

    return solution
