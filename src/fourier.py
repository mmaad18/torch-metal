import numpy as np
from typing import Tuple

from src.operations import fip

'''
1-D signal
'''
def signal1(size: int = 1000):
    return np.random.random(size)


'''
N-point wrapped 1-D signal
'''
def signal_wrap1(f: np.ndarray, N: int):
    L = len(f)
    P = int(np.ceil(L / N))

    # Pad f with zeros
    f_pad = np.zeros(P * N, dtype=np.float64)
    f_pad[:L] = f

    # Reshape f_pad into a matrix
    f_tilde = f_pad.reshape([P, N])

    return f_tilde.sum(axis=0)


'''
2-D signal
'''
def signal2(shape: Tuple[int, int] = (100, 100)):
    return np.random.random(shape)


'''
3-D signal
'''
def signal3(shape: Tuple[int, int, int] = (10, 10, 10)):
    return np.random.random(shape)


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
    return np.exp(-2j * np.pi * k * n / N)


'''
Twiddle factor where n = 1
'''
def twiddle_factor1(k: int, N: int):
    return np.exp(-2j * np.pi * k / N)


'''
Vector of twiddle factors
'''
def twiddle_vector(N: int):
    return np.exp(-2j * np.pi * np.arange(N // 2) / N)


'''
Matrix of twiddle factors
'''
def twiddle_matrix(k1: int, k2: int, M: int):
    x_vec = np.exp(-2j * np.pi * k1 * np.arange(M) / M)
    y_vec = np.exp(-2j * np.pi * k2 * np.arange(M) / M)

    return np.outer(y_vec, x_vec)


def twiddle_matrix_manual(k1: int, k2: int, M: int):
    W = np.zeros((M, M), dtype=np.complex128)

    for m1 in range(M):
        for m2 in range(M):
            W[m1, m2] = np.exp(-2j * np.pi * (k1 * m1 + k2 * m2) / M)

    return W

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
def dft_matrix_symm(N: int):
    return dft_matrix(N, N)


'''
DFT matrix using the wrapped method
'''
def dft_matrix_wrap(M: int, N: int):
    P = int(np.ceil(N / M))
    A_t = dft_matrix_symm(M)

    return np.tile(A_t, (1, P))


'''
Calculate the DFT of a 1-D signal using DFT matrix
'''
def dft_mat1(f: np.ndarray):
    N = len(f)
    A = dft_matrix_symm(N)

    return A.dot(f)


'''
Calculate the DFT of a 2-D signal using 2 DFT matrix
'''
def dft_mat2(f: np.ndarray):
    M, N = f.shape
    A_M = dft_matrix_symm(M)  # DFT matrix for the rows
    A_N = dft_matrix_symm(N)  # DFT matrix for the columns

    return A_M.dot(f).dot(A_N)


'''
Calculate the DFT of a 3-D signal using 3 DFT matrix
'''
def dft_mat3(f: np.ndarray):
    M, N, O = f.shape
    A_M = dft_matrix_symm(M)  # DFT matrix for the rows
    A_N = dft_matrix_symm(N)  # DFT matrix for the columns
    A_O = dft_matrix_symm(O)  # DFT matrix for the depths

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


'''
Bit reversal

int(bin(i)[2:] => convert i to binary and remove the '0b' prefix
.zfill(bits) => pad the binary number with zeros to the left
[::-1] => reverse the binary number
int(, 2) => convert the binary number back to decimal
'''
def bit_rev_raw(N: int):
    bits = int(np.log2(N))

    rev_indices = np.zeros(N, dtype=int)

    for i in range(N):
        rev_indices[i] = int(bin(i)[2:].zfill(bits)[::-1], 2)

    return rev_indices


def bit_rev(N: int):
    bits = int(np.log2(N))
    rev_indices = np.arange(N, dtype=int)
    reversed_bits = np.zeros_like(rev_indices)

    for i in range(bits):
        # Shift rev_indices right by i, isolate the bit, shift it to its new position, and accumulate
        reversed_bits |= ((rev_indices >> i) & 1) << (bits - 1 - i)

    return reversed_bits


def bit_rev_signal2(f: np.ndarray):
    N, M = f.shape

    brN = bit_rev(N)
    brM = bit_rev(M)

    f_reordered = np.zeros_like(f)
    for i in range(N):
        for j in range(M):
            f_reordered[i, j] = f[brN[i], brM[j]]

    return f_reordered


def construct_stages1(N: int):
    stages = []
    n = 2

    while n <= N:
        m = n // 2
        x = np.zeros((N // m, m), dtype=np.complex128)
        stages.append(x)
        n *= 2

    stages.append(np.zeros((1, N), dtype=np.complex128))

    return stages


def construct_stages_symm2(N):
    stages = []
    n = 2

    while n <= N:
        m = n // 2
        x = np.zeros((N // m, N // m, m, m), dtype=np.complex128)
        stages.append(x)
        n *= 2

    stages.append(np.zeros((1, 1, N, N), dtype=np.complex128))

    return stages


'''
FFT of a 1-D signal 
'''
def fft_mat1(f: np.ndarray):
    L = len(f)

    indices = bit_rev(L)
    f = f[indices]

    stages = construct_stages1(L)
    stages[0] = f

    n = 2
    for i in range(1, len(stages)):
        W = twiddle_vector(n)
        prev_stage = stages[i-1]

        for j in range(0, len(prev_stage), 2):
            psW = prev_stage[j + 1] * W

            stages[i][j//2] = np.concatenate((prev_stage[j] + psW, prev_stage[j] - psW))

        n *= 2

    return stages[-1]


'''
FFT of a 2-D signal
'''
def fft_mat2(f: np.ndarray):
    N0 = f.shape[0]
    f = bit_rev_signal2(f)

    stages = construct_stages_symm2(N0)

    stages[0] = f.reshape(N0, N0, 1, 1)

    CMS = np.array([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1]
    ])

    for i in range(1, len(stages)):
        stage = stages[i]
        size = stage.shape[0]
        N = stage.shape[2]

        prev_stage = stages[i - 1]
        prev_size = prev_stage.shape[0]
        M = prev_stage.shape[2]

        print(f"Stage {i}: size={size}, prev_size={prev_size}, dft_size={N}, prev_dft_size={M}")

        for j in range(0, prev_size, 2):
            for k in range(0, prev_size, 2):
                X_in = prev_stage[j:j + 2, k:k + 2]

                X_out = np.zeros((N, N), dtype=np.complex128)

                for k1 in range(M):
                    for k2 in range(M):
                        W_M = twiddle_matrix_manual(k1, k2, M)

                        #print(f"(j, k): ({j}, {k}), (k1, k2): ({k1}, {k2}), W_M: {W_M.shape}, X_in: {X_in.shape}")

                        S_00 = fip(X_in[0, 0], W_M)
                        S_01 = fip(X_in[0, 1], W_M)
                        S_10 = fip(X_in[1, 0], W_M)
                        S_11 = fip(X_in[1, 1], W_M)

                        S_4 = np.array([
                            S_00,
                            twiddle_factor1(k2, N) * S_01,
                            twiddle_factor1(k1, N) * S_10,
                            twiddle_factor1(k1 + k2, N) * S_11
                        ])

                        X_4 = CMS @ S_4

                        X_out[k1, k2] = X_4[0]
                        X_out[k1, k2 + M] = X_4[1]
                        X_out[k1 + M, k2] = X_4[2]
                        X_out[k1 + M, k2 + M] = X_4[3]


                stages[i][j//2, k//2] = X_out

    return stages[-1].reshape(N0, N0)


