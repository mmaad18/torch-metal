import sympy as sp


def symbolic_print(M):
    sp.pprint(M, use_unicode=False)


def symbolic_print_latex(M):
    print(sp.latex(M))


def symbolic_mat(N, M, symbol: str = 'a'):
    return sp.Matrix([[sp.symbols(f'{symbol}{i}{j}') for j in range(1, N + 1)] for i in range(1, M + 1)])


def symbolic_mat_symm(N, symbol: str = 'a'):
    return sp.Matrix([[sp.symbols(f'{symbol}{i}{j}') for j in range(1, N + 1)] for i in range(1, N + 1)])


def symbolic_sum(A, B):
    N = A.shape[0]
    out = 0

    for i in range(N):
        for j in range(N):
            out += A[i, j] * B[i, j]

    return out




