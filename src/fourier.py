import numpy as np
from typing import Tuple


def signal1(size: int = 1000):
    f = np.random.random(size)
    return np.array(f, dtype=np.complex128)


def signal2(shape: Tuple[int, int] = (100, 100)):
    f = np.random.random(shape)
    return np.array(f, dtype=np.complex128)


def signal3(shape: Tuple[int, int, int] = (10, 10, 10)):
    f = np.random.random(shape)
    return np.array(f, dtype=np.complex128)


def dft_sum1(f: np.ndarray):
    M = len(f)
    F = np.zeros(M, dtype=np.complex128)
    for p in range(0, M):
        for m in range(0, M):
            F[p] += f[m] * np.exp(-1j * 2 * np.pi * p * m / M)
    return F


def dft_sum2(f: np.ndarray):
    M, N = np.shape(f)
    F = np.zeros([M, N], dtype=np.complex128)
    for p in range(0, M):
        for q in range(0, N):
            for m in range(0, M):
                for n in range(0, N):
                    F[p, q] += f[m, n] * np.exp(-1j * 2 * np.pi * (p * m / M + q * n / N))
    return F


def dft_sum3(f: np.ndarray):
    M, N, O = np.shape(f)
    F = np.zeros([M, N, O], dtype=np.complex128)
    for p in range(0, M):
        for q in range(0, N):
            for r in range(0, O):
                for m in range(0, M):
                    for n in range(0, N):
                        for o in range(0, O):
                            F[p, q, r] += f[m, n, o] * np.exp(
                                -1j * 2 * np.pi * (p * m / M + q * n / N + r * o / O)
                            )
    return F


def twiddle_factor(k: int, n: int, N: int):
    return np.exp(-1j * 2 * np.pi * k * n / N)


def dft_matrix(N: int):
    A = np.zeros([N, N], dtype=np.complex128)

    for k in range(0, N):
        for n in range(0, N):
            A[k, n] = twiddle_factor(k, n, N)

    #return np.array([[twiddle_factor(k, n, N) for n in range(N)] for k in range(N)])
    return A


def dft_mat1(f: np.ndarray):
    N = len(f)
    A = dft_matrix(N)

    return A.dot(f)


def dft_mat2(f: np.ndarray):
    M, N = f.shape
    A_M = dft_matrix(M)  # DFT matrix for the rows
    A_N = dft_matrix(N)  # DFT matrix for the columns

    # Apply DFT along rows
    F_rowwise = np.dot(A_M, f)

    # Apply DFT along columns
    F = np.dot(F_rowwise, A_N.T)  # Transpose A_N to align for column-wise operation

    return F


def dft_mat3(f: np.ndarray):
    M, N, O = f.shape
    A_M = dft_matrix(M)  # DFT matrix for the rows
    A_N = dft_matrix(N)  # DFT matrix for the columns
    A_O = dft_matrix(O)  # DFT matrix for the depths

    # Apply DFT along the first dimension (M)
    F_first = np.zeros_like(f, dtype=np.complex128)
    for n in range(N):
        for o in range(O):
            F_first[:, n, o] = A_M.dot(f[:, n, o])

    # Apply DFT along the second dimension (N)
    F_second = np.zeros_like(f, dtype=np.complex128)
    for m in range(M):
        for o in range(O):
            F_second[m, :, o] = A_N.dot(F_first[m, :, o])

    # Apply DFT along the third dimension (O)
    F = np.zeros_like(f, dtype=np.complex128)
    for m in range(M):
        for n in range(N):
            F[m, n, :] = A_O.dot(F_second[m, n, :])

    return F



