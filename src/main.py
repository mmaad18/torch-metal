import numpy as np
import torch

from fourier import *
from utils import *

def main():
    M = 1000
    f = signal(M)
    ft = torch.from_numpy(f)

    F = time_function(dft_sum1, f)
    Ft = time_function(torch.fft.fft, ft)
    print("F:\n", F)
    print("Ft:\n", Ft)

    diff = np.sum(np.abs(F - Ft.numpy())) / M
    print("diff:\n", diff)




main()
