import numpy as np
from numpy.linalg import norm, LinAlgError


def qr_factorization_householder(A):
    """
    Performs the QR factorization via the Householder transformation in O(n^3)
    see e.g. https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections

    Args:
        A: a (m x n) matrix, that should be factorized (there are no further requirements)

    Returns: The solution matrices Q and R, where
                Q is an orthogonal (m x m) matrix and
                R is (m x n) upper triangular
    """
    assert norm(A) > 0, "QR factorization of zero-matrix not possible"
    m, n = A.shape

    I = np.identity(m)
    Q = I.copy()

    for k in range(min(m - 1, n)):
        A_small = A[k:, k:]
        x = A_small[:, 0]  # select column vector of A
        alpha = np.abs(norm(x))
        u = x - alpha * np.identity(m - k)[0, :]

        u_norm = norm(u)
        v = u / u_norm if u_norm > 0 else u

        Q_small = np.identity(m - k) - 2 * np.outer(v, v)

        Q_k = I.copy()
        Q_k[k:, k:] = Q_small

        Q = Q @ Q_k.T  # update Q
        A = Q_k @ A  # update A, this will at termination become the upper-triangular matrix R

    return Q, A


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
        for j in range(i + 1):
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
                    raise LinAlgError("PLU factorization isn't possible as there is a column with only 0 entries.")
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


# TODO: change to actual unit test
if __name__ == "__main__":
    for _ in range(10):
        shape = np.random.randint(low=3, high=7, size=2)
        A = np.random.uniform(low=2, high=16, size=shape)
        Q, R = qr_factorization_householder(A)

        # ensure that QR factorization works
        assert np.allclose(A, Q @ R)
