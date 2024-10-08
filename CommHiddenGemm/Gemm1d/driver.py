import numpy as np
from mpi4py import MPI
import logging
import sys
import os
import csv
import argparse
import gc  # garbage collection

from GemmUtil.constants import (
    MATRIX_DTYPE,
    BENCHMARK_FILE,
)

from GemmUtil.helper_general import (
    generate_matrix,
    matrix_multiply,
    matrices_equal,
    calculate_throughput,
)

from GemmUtil.helper_1d import (
    split_matrix,
    dump_unequal_matrices,
    generate_local_matrix,
)

from gemm1d import (
    allgather_A_col,
    allgather_A_row,
    allgather_B_col,
    allgather_B_row,
    reducescatter_C_col,
    reducescatter_C_row,
    broadcast_based,
    broadcast_based_with_overlap,
    throughput_test,
)
from gemm1d_no_compute import (
    allgather_A_col_no_compute,
    allgather_A_row_no_compute,
    allgather_B_col_no_compute,
    allgather_B_row_no_compute,
    reducescatter_C_col_no_compute,
    reducescatter_C_row_no_compute,
    broadcast_based_no_compute,
    broadcast_based_with_overlap_no_compute,
)

logging.basicConfig(level=logging.DEBUG)  # nothing is 51

STOP_OUTPUT = False
BENCHMARK_FOLDER = "benchmarks"
DEFAULT_STRATEGY = allgather_A_col

COMPUTE_STRATEGIES = [
    allgather_A_col,
    allgather_A_row,
    allgather_B_col,
    allgather_B_row,
    reducescatter_C_col,
    reducescatter_C_row,
    broadcast_based,
    broadcast_based_with_overlap,
    throughput_test,
]

NO_COMPUTE_STRATEGIES = [
    allgather_A_col_no_compute,
    allgather_A_row_no_compute,
    allgather_B_col_no_compute,
    allgather_B_row_no_compute,
    reducescatter_C_col_no_compute,
    reducescatter_C_row_no_compute,
    broadcast_based_no_compute,
    broadcast_based_with_overlap_no_compute,
]

EXPLODED_STRATEGIES = COMPUTE_STRATEGIES + NO_COMPUTE_STRATEGIES
NUM_REPEATS = 7


if not os.path.exists(BENCHMARK_FOLDER):
    os.makedirs(BENCHMARK_FOLDER)


def parse_command_line_args():
    """
    Parse command line arguments for array dimensions and strategy selection.

    Returns:
        tuple: A tuple containing:
            - m (int): M dimension of the matrix.
            - k (int): K dimension of the matrix.
            - n (int): N dimension of the matrix.
            - strategy (function): The chosen strategy function.
            - ntpn (int): Number of tasks per node.
    """
    parser = argparse.ArgumentParser(
        description="Array Dimensions. A (m x k), B (k x n), C (m x n)"
    )
    parser.add_argument("-m", dest="m", type=int, help="M dimension", default=None)
    parser.add_argument("-k", dest="k", type=int, help="K dimension", default=None)
    parser.add_argument("-n", dest="n", type=int, help="N dimension", default=None)
    parser.add_argument(
        "-s",
        "--strategy",
        dest="strategy",
        type=str,
        help="Specify strategy",
        default=None,
    )
    parser.add_argument("-nc", "--no-compute", dest="no_compute", action="store_true")
    parser.add_argument(
        "-ntasks-per-node", dest="ntpn", default=-1, type=int, action="store_true"
    )
    args = parser.parse_args()

    m = args.m
    k = args.k
    n = args.n
    strategy = args.strategy
    ntpn = args.ntpn

    if all(elem is None for elem in [m, k, n]):
        # all elements are none so set to default value
        m, k, n = (16, 8, 4)
        print(
            f"No dimension arguments provided using default of (m,k,n) = ({m},{k},{n})"
        )

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
        print(
            f"No strategy provided using best option based on dimensions which is {strategy.__name__}"
        )
    else:
        local_index = 1 if args.no_compute and strategy != "throughput_test" else 0
        strategy = [
            # EXPLODED_STRATEGIES used to be STRATEGIES idk what this does come back here if it crashes
            strat[local_index]
            for strat in EXPLODED_STRATEGIES
            if strategy == strat[0].__name__
        ]
        if len(strategy) == 0:
            parser.error(
                f"Provided strategy is not one of the presets of: \n {[strat.__name__ for strat in EXPLODED_STRATEGIES]}"
            )
        strategy = strategy[0]

    if not all([m, k, n]):
        parser.error("If you specify any dimension, you must specify all of them.")

    return (m, k, n, strategy, ntpn)


def driver(manual_args):
    if manual_args is None:
        m, k, n, strategy = parse_command_line_args()
    else:
        m = manual_args["m"]
        k = manual_args["k"]
        n = manual_args["n"]
        strategy = manual_args["strategy"]
        ntpm = manual_args["ntpm"]

    # print(f"(m,k,n) = ({m},{k},{n})")
    # print(f"strategy = {strategy.__name__}")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert all(
        dimension % size == 0 for dimension in [m, k, n]
    ), "All dimensions must be divisible by the number of processors"

    A_I, B_I, C_I = [None] * 3
    # this does not accoutn for before the algorithm where all 3 matrices exist or when it finishes and the final matrix is gathered to a process
    expected_max_memory_per_proc_GB = None  # some will have 2 for the point when like you doing to computation but one is like arriving/leaving

    if strategy.__name__ in ["allgather_A_col", "allgather_A_col_no_compute"]:
        A_I = generate_local_matrix(m, k, "c", size)
        B_I = generate_local_matrix(k, n, "c", size)
        C_I = generate_local_matrix(m, n, "c", size, zeros=True)
        expected_max_memory_per_proc_GB = (
            2 * sys.getsizeof(A_I) + sys.getsizeof(B_I) + sys.getsizeof(C_I)
        ) / (1024**3)
    elif strategy.__name__ in ["allgather_A_row", "allgather_A_row_no_compute"]:
        A_I = generate_local_matrix(m, k, "r", size)
        B_I = generate_local_matrix(k, n, "c", size)
        C_I = generate_local_matrix(m, n, "c", size, zeros=True)
        expected_max_memory_per_proc_GB = (
            2 * sys.getsizeof(A_I) + sys.getsizeof(B_I) + sys.getsizeof(C_I)
        ) / (1024**3)
    elif strategy.__name__ in ["allgather_B_col", "allgather_B_col_no_compute"]:
        A_I = generate_local_matrix(m, k, "r", size)
        B_I = generate_local_matrix(k, n, "c", size)
        C_I = generate_local_matrix(m, n, "r", size, zeros=True)
        expected_max_memory_per_proc_GB = (
            sys.getsizeof(A_I) + 2 * sys.getsizeof(B_I) + sys.getsizeof(C_I)
        ) / (1024**3)
    elif strategy.__name__ in [
        "allgather_B_row",
        "allgather_B_row_no_compute",
        "broadcast_based",
        "broadcast_based_no_compute",
        "broadcast_based_with_overlap",
        "broadcast_based_with_overlap_no_compute",
    ]:
        A_I = generate_local_matrix(m, k, "r", size)
        B_I = generate_local_matrix(k, n, "r", size)
        C_I = generate_local_matrix(m, n, "r", size, zeros=True)
        expected_max_memory_per_proc_GB = (
            sys.getsizeof(A_I) + 2 * sys.getsizeof(B_I) + sys.getsizeof(C_I)
        ) / (1024**3)
    elif strategy.__name__ in ["reducescatter_C_col", "reducescatter_C_col_no_compute"]:
        A_I = generate_local_matrix(m, k, "c", size)
        B_I = generate_local_matrix(k, n, "r", size)
        C_I = generate_local_matrix(m, n, "c", size, zeros=True)
        expected_max_memory_per_proc_GB = (
            sys.getsizeof(A_I) + sys.getsizeof(B_I) + 2 * sys.getsizeof(C_I)
        ) / (1024**3)
    elif strategy.__name__ in ["reducescatter_C_row", "reducescatter_C_row_no_compute"]:
        A_I = generate_local_matrix(m, k, "c", size)
        B_I = generate_local_matrix(k, n, "r", size)
        C_I = generate_local_matrix(m, n, "r", size, zeros=True)
        expected_max_memory_per_proc_GB = (
            sys.getsizeof(A_I) + sys.getsizeof(B_I) + 2 * sys.getsizeof(C_I)
        ) / (1024**3)
    elif strategy.__name__ in ["throughput_test"]:
        A_I = generate_matrix(m, k, -10, 10)
        B_I = generate_matrix(k, n, -10, 10)
        C_I = generate_matrix(m, n, -10, 10)
        expected_max_memory_per_proc_GB = (
            (sys.getsizeof(A_I) + sys.getsizeof(B_I) + sys.getsizeof(C_I))
        ) / (1024**3)

    MATRIX_A = None
    MATRIX_B = None
    MATRIX_C = None
    gc.collect()  # ensure that we do not bog down the memory of the system

    # only rank 0 has the full out matrix
    start_time = MPI.Wtime()
    out = strategy(A_I, B_I, C_I)
    elapsed_time = MPI.Wtime() - start_time

    if rank == 0:
        if manual_args is None:
            print(f"Output:\n{out}")
            print(
                f"Correct output?: N/A, Throughput GF/s: {calculate_throughput(elapsed_time, m, k, n)}, Elapsed time: {elapsed_time}"
            )
        else:
            with open(
                f"{BENCHMARK_FOLDER}/N{size}-n{ntpm}-{BENCHMARK_FILE}",
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                # equal = matrices_equal(standard_multiply, out)
                writer.writerow(
                    [
                        strategy.__name__,
                        size,
                        m,
                        n,
                        k,
                        calculate_throughput(elapsed_time, m, k, n),
                        elapsed_time,
                        expected_max_memory_per_proc_GB,
                    ]
                )

                if not True and strategy not in NO_COMPUTE_STRATEGIES:
                    dump_unequal_matrices(
                        f"{BENCHMARK_FOLDER}/failure.txt",
                        MATRIX_A,
                        MATRIX_B,
                        MATRIX_C,
                        standard_multiply,
                        out,
                        other_info=f"(m,k,n)=({m},{k},{n})",
                    )


def main():
    if len(sys.argv) > 2:
        driver(None)
        return
    ntasks_per_node = -1
    if len(sys.argv) == 2:
        ntasks_per_node = int(sys.argv[1])
    # I am going to try and run on max 48 cpus? maybe more later
    # with 48 divisors are 1,2,4,6,8,12,16,24,48
    # dimensions = [48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672] #, 720, 768, 816, 864, 912, 960, 1008, 1440] #, 1920, 2400, 2880, 3360, 3840, 4320, 4800, 5760, 7680, 8640, 9600, 12000, 14400, 16800, 19200, 21600, 24000, 31200, 48000, 60000] #, 72000, 84000, 96000, 120000]
    # dimensions = [480, 1200, 2400, 4800, 9600, 12000, 14400, 16800, 24000, 36000]
    # dimensions = [1200, 2400, 4800, 9600, 12000, 18000, 24000]
    dimensions = [1440, 2880, 4320, 7200, 10080, 14400, 18720, 24480]
    # dimensions = [48, 240]#, 720]
    # dimensions = [4, 8, 12, 16]
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    EXPLODED_STRATEGIES.remove(throughput_test)
    if size == 1:
        EXPLODED_STRATEGIES.clear()
        EXPLODED_STRATEGIES.append(
            throughput_test
        )  # throughput is just testing one processor
    # n2 starts at allgather_B_col_no_compute - adjust time on the sbatch
    # n4 starts at reducescatter_C_row
    # n6 starts at broadcast_based_with_overlap
    # 5 10 14
    if rank == 0:
        for val in EXPLODED_STRATEGIES:
            print(f"{val.__name__}")
    for algo in EXPLODED_STRATEGIES:
        if rank == 0:
            print(f"algorithm is {algo.__name__}", flush=True)
        for m in dimensions:
            if rank == 0:
                print(f"m is {m}", flush=True)
            for k in dimensions:
                for n in dimensions:
                    for _ in range(NUM_REPEATS):
                        np.random.seed(
                            42
                        )  # reset the seed each time so we can go back and debug if needed
                        driver(
                            {
                                "strategy": algo,
                                "m": m,
                                "k": k,
                                "n": n,
                                "ntpm": ntasks_per_node,
                            }
                        )
                gc.collect()


if __name__ == "__main__":
    main()
