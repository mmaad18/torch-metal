import numpy as np
from typing import Tuple


def signal1(size: int = 1000):
    f = np.random.random(size)
    return np.array(f, dtype=np.complex128)


def signal2(size: Tuple[int, int] = (100, 100)):
    f = np.random.random(size)
    return np.array(f, dtype=np.complex128)


def signal3(size: Tuple[int, int, int] = (10, 10, 10)):
    f = np.random.random(size)
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


def dft_mat1(f: np.ndarray):
    N = len(f)
    A = np.zeros([N, N], dtype=np.complex128)
    A[0, :] = np.ones(N, dtype=np.complex128)
    A[:, 0] = np.ones(N, dtype=np.complex128)

    for k in range(1, N):
        for n in range(1, N):
            A[k, n] = twiddle_factor(k, n, N)

    return A.dot(f)


def dft_mat2(f: np.ndarray):
    M, N = np.shape(f)
    A = np.zeros([M, N], dtype=np.complex128)
    A[0, :] = np.ones(N, dtype=np.complex128)
    A[:, 0] = np.ones(M, dtype=np.complex128)

    for p in range(1, M):
        for q in range(1, N):
            for m in range(1, M):
                for n in range(1, N):
                    A[p, q] += twiddle_factor(p, m, M) * twiddle_factor(q, n, N)

    return A.dot(f)




