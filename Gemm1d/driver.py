import numpy as np
from mpi4py import MPI
from util import MATRIX_DTYPE, MINI_MATRIX_A, MINI_MATRIX_B, MINI_MATRIX_C
from gemm1d import allgather_A_col


def matrices_equal(A, B):
    return np.isclose(A, B).all()

    # in the future command line args will be used to pick between a small large and 
    # whatever size matrix you want and whatever size matrix with rules for easy computation

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == 4, "Number of processors must be 4."

    standard_multiply = np.matmul(MINI_MATRIX_A, MINI_MATRIX_B) + MINI_MATRIX_C

    m, k = MINI_MATRIX_A.shape 
    n = MINI_MATRIX_B.shape[1]

    A_I = MINI_MATRIX_A[:, rank * (k // size) : (rank + 1) * (k // size)]
    B_I = MINI_MATRIX_B[:, rank * (n // size) : (rank + 1) * (n // size)]
    C_I = MINI_MATRIX_C[:, rank * (n // size) : (rank + 1) * (n // size)]

    out = np.empty((n,m), dtype=MATRIX_DTYPE)

    allgather_A_col(A_I, B_I, C_I, out)
    
    if (rank == 0):
        print(out.T)
        print(f"Correct output?: {matrices_equal(out.T, standard_multiply)}")
        