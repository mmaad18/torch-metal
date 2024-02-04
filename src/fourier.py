import numpy as np
from typing import Tuple


'''
1-D signal
'''
def signal1(size: int = 1000):
    f = np.random.random(size)
    return np.array(f, dtype=np.complex128)


'''
N-point wrapped 1-D signal
'''
def signal_wrap1(f: np.ndarray, N: int):
    L = len(f)
    P = int(np.ceil(L / N))

    # Pad f with zeros
    f_pad = np.zeros(P * N, dtype=np.complex128)
    f_pad[:L] = f

    # Reshape f_pad into a matrix
    f_tilde = f_pad.reshape([P, N])

    return f_tilde.sum(axis=0)


'''
2-D signal
'''
def signal2(shape: Tuple[int, int] = (100, 100)):
    f = np.random.random(shape)
    return np.array(f, dtype=np.complex128)


'''
3-D signal
'''
def signal3(shape: Tuple[int, int, int] = (10, 10, 10)):
    f = np.random.random(shape)
    return np.array(f, dtype=np.complex128)


'''
Calculate the DFT of a 1-D signal using sums
'''
def dft_sum1(f: np.ndarray):
    M = len(f)
    F = np.zeros(M, dtype=np.complex128)
    for p in range(0, M):
        for m in range(0, M):
            F[p] += f[m] * np.exp(-1j * 2 * np.pi * p * m / M)
    return F


'''
Calculate the DFT of a 2-D signal using sums
'''
def dft_sum2(f: np.ndarray):
    M, N = np.shape(f)
    F = np.zeros([M, N], dtype=np.complex128)
    for p in range(0, M):
        for q in range(0, N):
            for m in range(0, M):
                for n in range(0, N):
                    F[p, q] += f[m, n] * np.exp(-1j * 2 * np.pi * (p * m / M + q * n / N))
    return F


'''
Calculate the DFT of a 3-D signal using sums
'''
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


'''
Twiddle factor
'''
def twiddle_factor(k: int, n: int, N: int):
    return np.exp(-1j * 2 * np.pi * k * n / N)


'''
DFT matrix
'''
def dft_matrix(M: int, N: int):
    A = np.zeros([M, N], dtype=np.complex128)

    for k in range(0, M):
        for n in range(0, N):
            A[k, n] = twiddle_factor(k, n, M)

    return A


'''
Symmetric DFT matrix
'''
def dft_matrix_sym(N: int):
    return dft_matrix(N, N)


'''
DFT matrix using the wrapped method
'''
def dft_matrix_wrap(M: int, N: int):
    P = int(np.ceil(N / M))
    A_t = dft_matrix_sym(M)

    return np.tile(A_t, (1, P))


'''
Calculate the DFT of a 1-D signal using DFT matrix
'''
def dft_mat1(f: np.ndarray):
    N = len(f)
    A = dft_matrix_sym(N)

    return A.dot(f)


'''
Calculate the DFT of a 2-D signal using 2 DFT matrix
'''
def dft_mat2(f: np.ndarray):
    M, N = f.shape
    A_M = dft_matrix_sym(M)  # DFT matrix for the rows
    A_N = dft_matrix_sym(N)  # DFT matrix for the columns

    return A_M.dot(f).dot(A_N)


'''
Calculate the DFT of a 3-D signal using 3 DFT matrix
'''
def dft_mat3(f: np.ndarray):
    M, N, O = f.shape
    A_M = dft_matrix_sym(M)  # DFT matrix for the rows
    A_N = dft_matrix_sym(N)  # DFT matrix for the columns
    A_O = dft_matrix_sym(O)  # DFT matrix for the depths

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


'''
Calculate the IDFT of a 1-D signal using sums
'''
def idft_sum1(F: np.ndarray, M: int):
    P = len(F)
    f = np.zeros(M, dtype=np.complex128)
    for m in range(0, M):
        for p in range(0, P):
            f[m] += F[p] * np.exp(1j * 2 * np.pi * p * m / M)
    return f / M


'''
Calculate the IDFT of a 1-D signal using DFT matrix
'''
def idft_wrap1(A_t: np.ndarray, F: np.ndarray, N: int):
    return A_t.conj().dot(F) / N


'''
Calculate the IDFT of a 1-D signal using DFT matrix.

When N > L and x_t is Real => Final conjuction is not needed.
'''
def idft_wrap_comp1(A_t: np.ndarray, F: np.ndarray, N: int):
    return A_t.dot(F.conj()) / N


def fft_mat1(f: np.ndarray):
    L = len(f)
    M = int(np.ceil(L / 2))

    G = np.zeros(M, dtype=np.complex128)
    H = np.zeros(M, dtype=np.complex128)
    W = np.zeros(M, dtype=np.complex128)

    for m in range(0, M, 2):
        for n in range(0, M, 2):
            G[m] += f[n] * twiddle_factor(m, n, M)

    for m in range(1, M, 2):
        for n in range(1, M, 2):
            H[m] += f[n] * twiddle_factor(m, n, M)
            W[m] = twiddle_factor(m, 1, M)

    X1 = np.zeros(M, dtype=np.complex128)
    X2 = np.zeros(M, dtype=np.complex128)

    X1 = G + H * W
    X2 = G - H * W

    return np.concatenate([X1, X2])






