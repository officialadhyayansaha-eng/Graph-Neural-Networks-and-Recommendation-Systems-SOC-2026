import numpy as np


def gram_schmidt(A):
    A = np.array(A, dtype=float)

    m, n = A.shape
    Q = np.zeros((m, n))

    for k in range(n):
        v = A[:, k].copy()

        for j in range(k):
            proj = np.dot(v, Q[:, j]) * Q[:, j]
            v = v - proj

        norm = np.linalg.norm(v)

        if norm < 1e-12:
            raise ValueError("Vectors are linearly dependent")

        Q[:, k] = v / norm

    return Q


if __name__ == "__main__":

    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])

    Q = gram_schmidt(A)

    print("Q =")
    print(Q)

    print("\nQᵀQ =")
    print(Q.T @ Q)

    print(
        "\nOrthogonality Check:",
        np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)
    )
