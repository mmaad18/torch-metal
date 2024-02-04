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
    #dft_mat3_test()
    #signal_wrap1_test()
    #dft_matrix_wrap_test()
    #dft_matrix_wrap_test2()
    #idft_wrap1_test()
    fft_mat1_test()


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


def signal_wrap1_test():
    M = 20
    f = signal1(M)

    fw = time_function(signal_wrap1, f, 10)
    F = time_function(dft_mat1, f)

    print("fw:\n", fw)
    print("F:\n", F)


def dft_matrix_wrap_test1():
    np.set_printoptions(precision=2, suppress=True)

    N = 4
    L = 8

    A = dft_matrix(N, L)
    Aw = dft_matrix_wrap(L, N)

    print("Aw:\n", Aw)
    print("A:\n", A)

    B = A - Aw
    print("B:\n", B)

    diff = np.sum(np.abs(A - Aw)) / (N*L)
    print("diff:\n", diff)


def dft_matrix_wrap_test2():
    np.set_printoptions(precision=2, suppress=True)

    N = 800
    L = 800

    f = signal1(L)

    start = time.perf_counter()
    A = dft_matrix(N, L)
    F = A.dot(f)
    end = time.perf_counter()
    time1 = end - start

    print(f"F took {time1} seconds")

    start = time.perf_counter()
    fw = signal_wrap1(f, N)
    Aw = dft_matrix_sym(N)
    Fw = Aw.dot(fw)
    end = time.perf_counter()
    time2 = end - start

    print(f"Fw took {time2} seconds")

    time_factor = time1 / time2
    print(f"time_factor: {time_factor}")


def idft_wrap1_test():
    np.set_printoptions(precision=3, suppress=True)

    M = 2000
    f = signal1(M)

    F = dft_mat1(f)

    fs = time_function(idft_sum1, F, M)

    A = time_function(dft_matrix_sym, M)

    ft = time_function(idft_wrap1, A, F, M)
    ftc = time_function(idft_wrap_comp1, A, F, M)

    diff1 = np.sum(np.abs(fs - ft)) / M
    print("diff1:\n", diff1)

    diff2 = np.sum(np.abs(fs - ftc)) / M
    print("diff2:\n", diff2)


def fft_mat1_test():
    M = 2000
    f = signal1(M)
    ft = torch.from_numpy(f)

    F = time_function(fft_mat1, f)
    Ft = time_function(torch.fft.fft, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / M
    print("diff:\n", diff)


main()
