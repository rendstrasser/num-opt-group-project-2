import numpy as np
from numpy.linalg import norm


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
        v = u / norm(u)

        Q_small = np.identity(m - k) - 2 * np.outer(v, v)

        Q_k = I.copy()
        Q_k[k:, k:] = Q_small

        Q = Q @ Q_k.T  # update Q
        A = Q_k @ A  # update A, this will at termination become the upper-triangular matrix R

    return Q, A


# TODO: change to actual unit test
if __name__ == "__main__":
    for _ in range(10):
        shape = np.random.randint(low=3, high=7, size=2)
        A = np.random.uniform(low=2, high=16, size=shape)
        Q, R = qr_factorization_householder(A)

        # ensure that QR factorization works
        assert np.all(np.isclose(A, Q @ R))

