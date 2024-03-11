import numpy as np
np.set_printoptions(linewidth=np.inf)
from mpi4py import MPI
import logging
import sys
import argparse
from util import (MATRIX_DTYPE, MINI_MATRIX_A, MINI_MATRIX_B, MINI_MATRIX_C, generate_matrix, matrix_multiply, matrices_equal, calculate_throughput)
from gemm1d import allgather_A_col, allgather_A_row, allgather_B_col, allgather_B_row, reducescatter_C_col, reducescatter_C_row

logging.basicConfig(level=logging.DEBUG) # nothing is 51

DEFAULT_STRATEGY = allgather_A_col

def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Array Dimensions. A (m x k), B (k x n), C (m x n)')
    parser.add_argument('-m', dest="m", type=int, help='M dimension', default=None)
    parser.add_argument('-k', dest="k", type=int, help='K dimension', default=None)
    parser.add_argument('-n', dest="n", type=int, help='N dimension', default=None)
    parser.add_argument('-s', '--strategy', dest='strategy', type=str, help='Specify strategy', default=None)
    args = parser.parse_args()

    m = args.m
    k = args.k
    n = args.n
    strategy = args.strategy

    if all(elem is None for elem in [m,k,n]):
        # all elements are none so set to default value
        m, k, n = (16,8,4)
        print(f"No dimension arguments provided using default of (m,k,n) = ({m},{k},{n})")

    if strategy is None:
        if n < k and k < m:
            strategy = allgather_A_col
        elif k < n and n < m:
            strategy = allgather_A_row
        elif k < m and m < n:
            strategy = allgather_B_col
        elif m < m and n < k:
            strategy = allgather_B_row
        elif m < n and n < k:
            strategy = reducescatter_C_col
        elif n < m and m < k:
            strategy = reducescatter_C_row
        else:
            strategy = DEFAULT_STRATEGY
        print(f"No strategy provided using best option based on dimensions which is {strategy.__name__}")
    else:
        strategies = [allgather_A_col, allgather_A_row, allgather_B_col, allgather_B_row, reducescatter_C_col, reducescatter_C_row]
        strategy = [strat for strat in strategies if strategy == strat.__name__]
        if len(strategy) == 0:
            parser.error(f"Provided strategy is not one of the presets of: \n {[strat.__name__ for strat in strategies]}")
        strategy = strategy[0]

    if not all([m,k,n]):
        parser.error("If you specify any dimension, you must specify all of them.")
    
    return (m,k,n,strategy)

def main():
    strategies = [allgather_A_col, allgather_A_row, allgather_B_col, allgather_B_row, reducescatter_C_col, reducescatter_C_row]
    m, k, n, strategy = parse_command_line_args()

    print(f"(m,k,n) = ({m},{k},{n})")
    print(f"strategy = {strategy.__name__}")


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    assert size > 1 and all(dimension % size == 0 for dimension in [m, k, n]), "All dimensions must be divisible by the number of processors"

    MATRIX_A = generate_matrix(m, k, -10, 10)
    MATRIX_B = generate_matrix(k, n, -10, 10)
    MATRIX_C = np.zeros((m, n), dtype=MATRIX_DTYPE)


    standard_multiply = matrix_multiply(MATRIX_A, MATRIX_B, MATRIX_C)

    if rank == 0:
        print(f"A:\n{MATRIX_A}\nB:\n{MATRIX_B}\nC:\n{MATRIX_C}\nExpected:\n{standard_multiply}")

    A_I, B_I, C_I = [None] * 3

    if strategy.__name__ == "allgather_A_col":
        A_I = MATRIX_A[:, rank * (k // size) : (rank + 1) * (k // size)]
        B_I = MATRIX_B[:, rank * (n // size) : (rank + 1) * (n // size)]
        C_I = MATRIX_C[:, rank * (n // size) : (rank + 1) * (n // size)]
    elif strategy.__name__ == "allgather_A_row":
        A_I = MATRIX_A[rank * (m // size) : (rank + 1) * (m // size),:]
        B_I = MATRIX_B[:, rank * (n // size) : (rank + 1) * (n // size)]
        C_I = MATRIX_C[:, rank * (n // size) : (rank + 1) * (n // size)]


    # only rank 0 has the full out matrix
    start_time = MPI.Wtime()
    out = strategy(A_I, B_I, C_I)
    elapsed_time = MPI.Wtime() - start_time


    if (rank == 0):
        print(f"Output:\n{out}")
        print(f"Correct output?: {matrices_equal(out, standard_multiply)}, Throughput GF/s: {calculate_throughput(elapsed_time, m, k, n)}, Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main()