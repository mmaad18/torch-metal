import numpy as np
import torch

from fourier import *
from src.symbolic import *
from utils import *

'''
Verifies that dft_sum1() is working correctly. 
Torch fft is orders of magnitude faster than dft_sum1.
'''
def dft_sum1_test():
    title_print("dft_sum1_test")

    M = 2000
    f = signal1(M)
    ft = torch.from_numpy(f)

    F, t1 = time_function_out(dft_sum1, f)
    Ft, t2 = time_function_out(torch.fft.fft, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / M
    print("diff:\n", diff)

    time_factor = t1 / t2
    print(f"time_factor: {time_factor}")

    print("F:\n", F)


'''
Verifies that dft_sum2() is working correctly. 
Torch fft is orders of magnitude faster than dft_sum2.
'''
def dft_sum2_test():
    title_print("dft_sum2_test")

    M, N = 50, 50
    f = signal2((M, N))
    ft = torch.from_numpy(f)

    F, t1 = time_function_out(dft_sum2, f)
    Ft, t2 = time_function_out(torch.fft.fft2, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / (M*N)
    print("diff:\n", diff)

    time_factor = t1 / t2
    print(f"time_factor: {time_factor}")


'''
Verifies that dft_sum3() is working correctly. 
Torch fft is orders of magnitude faster than dft_sum3.
'''
def dft_sum3_test():
    title_print("dft_sum3_test")

    M, N, O = 15, 15, 15
    f = signal3((M, N, O))
    ft = torch.from_numpy(f)

    F, t1 = time_function_out(dft_sum3, f)
    Ft, t2 = time_function_out(torch.fft.fftn, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / (M*N*O)
    print("diff:\n", diff)

    time_factor = t1 / t2
    print(f"time_factor: {time_factor}")


'''
Verifies that dft_mat1() is working correctly. 
Torch fft is orders of magnitude faster than dft_mat1.
dft_mat1 is faster than dft_sum1.
'''
def dft_mat1_test():
    title_print("dft_mat1_test")

    M = 2000
    f = signal1(M)
    ft = torch.from_numpy(f)

    F, t1 = time_function_out(dft_mat1, f)
    Ft, t2 = time_function_out(torch.fft.fft, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / M
    print("diff:\n", diff)

    time_factor = t1 / t2
    print(f"time_factor1: {time_factor}")


'''
Verifies that dft_mat2() is working correctly. 
Torch fft is orders of magnitude faster than dft_mat2.
dft_mat2 is orders of magnitude faster than dft_sum2.
'''
def dft_mat2_test():
    title_print("dft_mat2_test")

    M, N = 1000, 1000
    f = signal2((M, N))
    ft = torch.from_numpy(f)

    F, t1 = time_function_out(dft_mat2, f)
    Ft, t2 = time_function_out(torch.fft.fft2, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / (M*N)
    print("diff:\n", diff)

    time_factor = t1 / t2
    print(f"time_factor1: {time_factor}")


'''
Verifies that dft_mat3() is working correctly. 
Torch fft is orders of magnitude faster than dft_mat3.
dft_mat3 is orders of magnitude faster than dft_sum3.
'''
def dft_mat3_test():
    title_print("dft_mat3_test")

    M, N, O = 300, 300, 300
    f = signal3((M, N, O))
    ft = torch.from_numpy(f)

    F, t1 = time_function_out(dft_mat3, f)
    Ft, t2 = time_function_out(torch.fft.fftn, ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / (M*N*O)
    print("diff:\n", diff)

    time_factor = t1 / t2
    print(f"time_factor1: {time_factor}")


'''
Verifies that output from signal_wrap1() looks sensible.
'''
def signal_wrap1_test():
    title_print("signal_wrap1_test")

    M = 20
    f = signal1(M)

    fw = time_function(signal_wrap1, f, 10)

    print("fw:\n", fw)
    print("fw length: ", len(fw))


'''
Verifies that dft_matrix_wrap() is working correctly.
'''
def dft_matrix_wrap_test1():
    title_print("dft_matrix_wrap_test1")

    np.set_printoptions(precision=2, suppress=True)

    N = 4
    L = 8

    A = dft_matrix(N, L)
    Aw = dft_matrix_wrap(N, L)

    print("Aw:\n", Aw)
    print("A:\n", A)

    diff = np.sum(np.abs(A - Aw)) / (N*L)
    print("diff:\n", diff)


'''
Proves that:
- When N < L => Computing DFT is faster using the wrapped method.
- When N > L => Computing DFT is faster using the normal method.
'''
def dft_matrix_wrap_test2():
    title_print("dft_matrix_wrap_test2")

    np.set_printoptions(precision=2, suppress=True)

    N = 2000
    L = 2000

    f = signal1(L)

    # Normal DFT
    start = time.perf_counter()
    A = dft_matrix(N, L)
    F = A.dot(f)
    end = time.perf_counter()
    time1 = end - start

    print(f"F took {time1} seconds")

    # Wrapped DFT
    start = time.perf_counter()
    fw = signal_wrap1(f, N)
    Aw = dft_matrix_symm(N)
    Fw = Aw.dot(fw)
    end = time.perf_counter()
    time2 = end - start

    print(f"Fw took {time2} seconds")

    time_factor = time1 / time2
    print(f"time_factor: {time_factor}")


'''
Proves that:
- idft_wrap1 is much faster than idft_sum1.
- idft_wrap_comp1 is much faster than idft_wrap1.
- Having to conjugate only the signal is much faster than having to conjugate the DFT matrix.
'''
def idft_wrap1_test():
    title_print("idft_wrap1_test")

    np.set_printoptions(precision=3, suppress=True)

    M = 2000
    f = signal1(M)
    A = dft_matrix_symm(M)
    F = dft_mat1(f)

    fs, t1 = time_function_out(idft_sum1, F, M)
    ft, t2 = time_function_out(idft_wrap1, A, F, M)
    ftc, t3 = time_function_out(idft_wrap_comp1, A, F, M)

    diff1 = np.sum(np.abs(fs - ft)) / M
    print("diff1:\n", diff1)

    diff2 = np.sum(np.abs(fs - ftc)) / M
    print("diff2:\n", diff2)

    time_factor1 = t1 / t2
    print(f"time_factor1: {time_factor1}")

    time_factor2 = t2 / t3
    print(f"time_factor2: {time_factor2}")


'''
Verifies that fft_mat1() is working correctly. 
Proves that:
- fft_mat1 is much slower than np.fft 
- bit_rev takes some time to run 
- construct_stages uses a small fraction of the total time of fft_mat1 
'''
def fft_mat1_test():
    title_print("fft_mat1_test")

    M = 2**3
    f = signal1(M)
    ft = torch.from_numpy(f)

    # time_function(bit_rev, M)
    # time_function(construct_stages, M)

    F1 = time_function(fft_mat1, f)
    Ft = time_function(torch.fft.fft, ft)
    Fn = time_function(np.fft.fft, f)

    diff1 = np.sum(np.abs(F1 - Fn)) / M
    print("diff1:\n", diff1)

    diff2 = np.sum(np.abs(Fn - Ft.numpy())) / M
    print("diff2:\n", diff2)


'''
Verifies that fft_mat2() works correctly.
'''
def fft_mat2_test():
    title_print("fft_mat2_test")

    M, N = 2**3, 2**3
    f = signal2((M, N))
    ft = torch.from_numpy(f)

    F1 = time_function(fft_mat2, f)
    Ft = time_function(torch.fft.fft2, ft)
    Fn = time_function(np.fft.fft2, f)
    Fm = time_function(dft_mat2, f)

    diff1 = np.sum(np.abs(F1 - Fn)) / (M*N)
    print("diff1:\n", diff1)

    diff2 = np.sum(np.abs(Fn - Ft.numpy())) / (M*N)
    print("diff2:\n", diff2)

    diff3 = np.sum(np.abs(Fm - Fn)) / (M*N)
    print("diff3:\n", diff3)


'''
Example of a 2D bit reversal 
'''
def bit_rev2_test():
    title_print("2D bit_rev test")

    N = 4
    M = 4
    f = np.zeros((N, M), dtype=int)

    start = time.perf_counter()

    F = time_function(bit_rev_signal2, f)

    end = time.perf_counter()
    print(f"bit_rev2 took {end - start} seconds")

    print("F:\n", F)


def sym_mat_test():
    title_print("sym_mat_test")

    A = symbolic_mat_symm(4, 'a')
    B = symbolic_mat_symm(4, 'b')

    M1 = symbolic_sum(A, B)
    M2 = sum(A.multiply_elementwise(B))

    diff = M1 - M2

    symbolic_print(M1)
    print("Diff:")
    symbolic_print(diff)


def main():
    #fft_mat1_test()
    fft_mat2_test()



main()
