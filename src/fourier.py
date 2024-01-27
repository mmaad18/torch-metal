import numpy as np


def signal(size: int = 1000):
    f = np.random.random(size)
    return np.array(f, dtype=np.complex128)

def dft_sum1(f: np.ndarray):
    M = len(f)
    F = np.zeros(M, dtype=np.complex128)
    for p in range(0, M):
        for m in range(0, M):
            F[p] += f[m] * np.exp(-1j * 2 * np.pi * p * m / M)
    return F



