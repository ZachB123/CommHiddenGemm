import numpy as np
np.set_printoptions(linewidth=np.inf)
from mpi4py import MPI
import logging
import sys
import os
import csv
import argparse
from util import (MATRIX_DTYPE, BENCHMARK_FILE, MINI_MATRIX_A, MINI_MATRIX_B, MINI_MATRIX_C, generate_matrix, matrix_multiply, matrices_equal, 
                  calculate_throughput, split_matrix)
from gemm1d import (allgather_A_col, allgather_A_row, allgather_B_col, allgather_B_row, reducescatter_C_col, reducescatter_C_row,
                    algorithm1, algorithm2, throughput_test)
from gemm1d_no_compute import (allgather_A_col_no_compute, allgather_A_row_no_compute, allgather_B_col_no_compute, allgather_B_row_no_compute, 
                               reducescatter_C_col_no_compute, reducescatter_C_row_no_compute, algorithm1_no_compute, algorithm2_no_compute)
logging.basicConfig(level=logging.DEBUG) # nothing is 51

DEFAULT_STRATEGY = allgather_A_col
STRATEGIES = [(allgather_A_col, allgather_A_col_no_compute),
                (allgather_A_row, allgather_A_row_no_compute),
                (allgather_B_col, allgather_B_col_no_compute),
                (allgather_B_row, allgather_B_row_no_compute),
                (reducescatter_C_col, reducescatter_C_col_no_compute),
                (reducescatter_C_row, reducescatter_C_row_no_compute), 
                (algorithm1, algorithm1_no_compute),
                (algorithm2, algorithm2_no_compute),
                (throughput_test,)
                ]

EXPLODED_STRATEGIES = [item for sublist in STRATEGIES for item in sublist]
NUM_REPEATS = 10

# disregard all stdout
sys.stdout = open(os.devnull, 'w')

def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Array Dimensions. A (m x k), B (k x n), C (m x n)')
    parser.add_argument('-m', dest="m", type=int, help='M dimension', default=None)
    parser.add_argument('-k', dest="k", type=int, help='K dimension', default=None)
    parser.add_argument('-n', dest="n", type=int, help='N dimension', default=None)
    parser.add_argument('-s', '--strategy', dest='strategy', type=str, help='Specify strategy', default=None)
    parser.add_argument('-nc', '--no-compute', dest='no_compute', action="store_true")
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
        local_index = 1 if args.no_compute and strategy != "throughput_test" else 0
        strategy = [strat[local_index] for strat in STRATEGIES if strategy == strat[0].__name__]
        if len(strategy) == 0:
            parser.error(f"Provided strategy is not one of the presets of: \n {[strat.__name__ for strat in EXPLODED_STRATEGIES]}")
        strategy = strategy[0]

    if not all([m,k,n]):
        parser.error("If you specify any dimension, you must specify all of them.")
    
    return (m,k,n,strategy)

def driver(manual_args):
    if manual_args is None:
        m, k, n, strategy = parse_command_line_args()
    else:
        m = manual_args["m"]
        k = manual_args["k"]
        n = manual_args["n"]
        strategy = manual_args["strategy"]


    print(f"(m,k,n) = ({m},{k},{n})")
    print(f"strategy = {strategy.__name__}")


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    assert all(dimension % size == 0 for dimension in [m, k, n]), "All dimensions must be divisible by the number of processors"

    MATRIX_A = generate_matrix(m, k, -10, 10)
    MATRIX_B = generate_matrix(k, n, -10, 10)
    MATRIX_C = np.zeros((m, n), dtype=MATRIX_DTYPE)


    standard_multiply = matrix_multiply(MATRIX_A, MATRIX_B, MATRIX_C)

    if rank == 0:
        print(f"A:\n{MATRIX_A}\nB:\n{MATRIX_B}\nC:\n{MATRIX_C}\nExpected:\n{standard_multiply}")

    A_I, B_I, C_I = [None] * 3

    if strategy.__name__ in ["allgather_A_col", "allgather_A_col_no_compute"]:
        A_I = split_matrix(MATRIX_A, "c", rank, size)
        B_I = split_matrix(MATRIX_B, "c", rank, size)
        C_I = split_matrix(MATRIX_C, "c", rank, size)
    elif strategy.__name__ in ["allgather_A_row", "allgather_A_row_no_compute"]:
        A_I = split_matrix(MATRIX_A, "r", rank, size)
        B_I = split_matrix(MATRIX_B, "c", rank, size)
        C_I = split_matrix(MATRIX_C, "c", rank, size)
    elif strategy.__name__ in ["allgather_B_col", "allgather_B_col_no_compute"]:
        A_I = split_matrix(MATRIX_A, "r", rank, size)
        B_I = split_matrix(MATRIX_B, "c", rank, size)
        C_I = split_matrix(MATRIX_C, "r", rank, size)
    elif strategy.__name__ in ["allgather_B_row", "allgather_B_row_no_compute", "algorithm1", "algorithm1_no_compute", "algorithm2", "algorithm2_no_compute"]:
        A_I = split_matrix(MATRIX_A, "r", rank, size)
        B_I = split_matrix(MATRIX_B, "r", rank, size)
        C_I = split_matrix(MATRIX_C, "r", rank, size)
    elif strategy.__name__ in ["reducescatter_C_col", "reducescatter_C_col_no_compute"]:
        A_I = split_matrix(MATRIX_A, "c", rank, size)
        B_I = split_matrix(MATRIX_B, "r", rank, size)
        C_I = split_matrix(MATRIX_C, "c", rank, size)
    elif strategy.__name__ in ["reducescatter_C_row", "reducescatter_C_row_no_compute"]:
        A_I = split_matrix(MATRIX_A, "c", rank, size)
        B_I = split_matrix(MATRIX_B, "r", rank, size)
        C_I = split_matrix(MATRIX_C, "r", rank, size)
    elif strategy.__name__ in ["throughput_test"]:
        A_I = MATRIX_A
        B_I = MATRIX_B
        C_I = MATRIX_C
    
    # if rank == 0:
    #     print(f"A_I\n{A_I}\nB_I\n{B_I}\nC_I\n{C_I}\n")

    # only rank 0 has the full out matrix
    start_time = MPI.Wtime()
    out = strategy(A_I, B_I, C_I)
    elapsed_time = MPI.Wtime() - start_time

    if (rank == 0):
        if manual_args is None:
            print(f"Output:\n{out}")
            print(f"Correct output?: {matrices_equal(standard_multiply, out)}, Throughput GF/s: {calculate_throughput(elapsed_time, m, k, n)}, Elapsed time: {elapsed_time}")
        else:
            with open(f"n{size}-{BENCHMARK_FILE}", mode="a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([strategy.__name__, size, m, n, k, calculate_throughput(elapsed_time, m, k, n), elapsed_time, matrices_equal(standard_multiply, out)])


def main():

    # driver(None)
    # I am going to try and run on max 48 cpus? maybe more later
    # with 48 divisors are 1,2,4,6,8,12,16,24,48
    # dimensions = [48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672] #, 720, 768, 816, 864, 912, 960, 1008, 1440] #, 1920, 2400, 2880, 3360, 3840, 4320, 4800, 5760, 7680, 8640, 9600, 12000, 14400, 16800, 19200, 21600, 24000, 31200, 48000, 60000] #, 72000, 84000, 96000, 120000]
    dimensions = [48, 144, 240, 480, 720, 960, 2400, 4800, 9600, 12000, 14400, 16800, 19200, 24000]
    # dimensions = [48, 144, 240, 480, 720]
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size != 1:
        EXPLODED_STRATEGIES.remove(throughput_test) # throughput is just testing one processor
    for algo in EXPLODED_STRATEGIES:
        for m in dimensions:
            for k in dimensions:
                for n in dimensions:
                    for _ in range(NUM_REPEATS):
                        driver({"strategy": algo, "m": m, "k": k, "n":n})
    # driver({"strategy": algorithm2, "m": 4, "k": 4, "n": 4})

if __name__ == "__main__":
    main()