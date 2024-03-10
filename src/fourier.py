import numpy as np
from typing import Tuple

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
def bit_rev(N):
    bits = int(np.log2(N))

    rev_indices = np.zeros(N, dtype=int)

    for i in range(N):
        rev_indices[i] = int(bin(i)[2:].zfill(bits)[::-1], 2)

    return rev_indices


def bit_rev_optimized(N):
    bits = int(np.log2(N))
    rev_indices = np.arange(N, dtype=int)
    reversed_bits = np.zeros_like(rev_indices)

    for i in range(bits):
        # Shift rev_indices right by i, isolate the bit, shift it to its new position, and accumulate
        reversed_bits |= ((rev_indices >> i) & 1) << (bits - 1 - i)

    return reversed_bits



def construct_stages(N):
    stages = []
    n = 2

    while n <= N:
        m = n // 2
        x = np.zeros((N // m, m), dtype=np.complex128)
        stages.append(x)
        n *= 2

    stages.append(np.zeros((1, N), dtype=np.complex128))

    return stages


'''
FFT of a 1-D signal 
'''
def fft_mat1(f: np.ndarray):
    L = len(f)

    indices = bit_rev_optimized(L)
    f = f[indices]

    stages = construct_stages(L)
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
    M, N = f.shape
    F = np.zeros([M, N], dtype=np.complex128)

    for m in range(M):
        F[m, :] = fft_mat1(f[m, :])

    for n in range(N):
        F[:, n] = fft_mat1(F[:, n])

    return F


