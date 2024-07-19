import numpy as np


"""
Calculate the Frobenius inner product of two matrices A and B.
"""
def fip(A: np.ndarray, B: np.ndarray):
    return np.sum(A * B)


