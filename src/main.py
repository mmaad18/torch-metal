import numpy as np
import torch

from fourier import *
from utils import *
from typing import Tuple


def main():
    #dft_sum1_test()
    #dft_sum2_test()
    #dft_sum3_test()
    #dft_mat1_test()
    #dft_mat2_test()
    dft_mat3_test()


def dft_sum1_test():
    M = 2000
    f = signal1(M)
    ft = torch.from_numpy(f)

    F = time_function(dft_sum1, f)
    Ft = time_function(torch.fft.fft, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / M
    print("diff:\n", diff)
    print("f:\n", signal2(3))


def dft_sum2_test():
    M, N = 50, 50
    f = signal2((M, N))
    ft = torch.from_numpy(f)

    F = time_function(dft_sum2, f)
    Ft = time_function(torch.fft.fft2, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / (M*N)
    print("diff:\n", diff)


def dft_sum3_test():
    M, N, O = 15, 15, 15
    f = signal3((M, N, O))
    ft = torch.from_numpy(f)

    F = time_function(dft_sum3, f)
    Ft = time_function(torch.fft.fftn, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / (M*N*O)
    print("diff:\n", diff)


def dft_mat1_test():
    M = 2000
    f = signal1(M)
    ft = torch.from_numpy(f)

    F = time_function(dft_mat1, f)
    Ft = time_function(torch.fft.fft, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / M
    print("diff:\n", diff)


def dft_mat2_test():
    M, N = 1000, 1000
    f = signal2((M, N))
    ft = torch.from_numpy(f)

    F = time_function(dft_mat2, f)
    Ft = time_function(torch.fft.fft2, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / (M*N)
    print("diff:\n", diff)


def dft_mat3_test():
    M, N, O = 100, 100, 100
    f = signal3((M, N, O))
    ft = torch.from_numpy(f)

    F = time_function(dft_mat3, f)
    Ft = time_function(torch.fft.fftn, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / (M*N*O)
    print("diff:\n", diff)


main()
