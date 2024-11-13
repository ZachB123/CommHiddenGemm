import numpy as np
from mpi4py import MPI
import gc
import os
import csv

from GemmUtil.constants import MATRIX_DTYPE, DEBUG_RANK

from GemmUtil.helper_general import (
    generate_matrix,
    parallel_print,
    rank_print,
    set_numpy_seed,
    get_date_string,
    calculate_throughput,
    matrix_multiply,
)

from GemmUtil.helper_2d import (
    block_cyclic,
    col_block_comm,
    col_comm,
    col_major_distribution,
    pad_amount,
    pure_column_distribution,
    pure_row_distribution,
    remove_padding,
    pad_matrix_with_zeros,
    get_step_indices,
    get_subtile2,
    row_block_comm,
    row_comm,
    row_major_distribution,
)

from .comm_hidden_2d import (
    AG_A_COL_X_AG_B_COL,
    AG_A_COL_X_AG_B_ROW,
    AG_A_ROW_X_AG_B_ROW,
    AG_A_ROW_X_RS_C_COL,
)


def one_processor_test():
    if not os.path.exists("./Gemm2d/TempBenchmarks"):
        os.makedirs("./Gemm2d/TempBenchmarks")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    prow = 4
    pcol = 4

    row_comm = comm.Split(rank // pcol, rank)
    col_comm = comm.Split(rank % pcol, rank)

    J = row_comm.Get_rank()
    I = col_comm.Get_rank()

    num_iterations = 100
    m = 16000
    k = 16000
    n = 16000

    # now we want the subile sizes cuz this is what is actually multiplied locally

    a_subtile_row = m // prow
    a_subtile_col = k // (prow * pcol)

    b_subtile_row = k // (prow * pcol)
    b_subtile_col = n // pcol

    c_subtile_row = m // prow
    c_subtile_col = n // pcol

    for i in range(num_iterations):
        if rank == 0:
            print(i, flush=True)

        A_I = generate_matrix(a_subtile_row, a_subtile_col, -10, 10)
        B_I = generate_matrix(b_subtile_row, b_subtile_col, -10, 10)
        C_I = generate_matrix(c_subtile_row, c_subtile_col, -10, 10)

        start_time = MPI.Wtime()
        for i in range(prow * pcol):
            matrix_multiply(A_I, B_I, C_I)
        elapsed_time = MPI.Wtime() - start_time

        gc.collect()

        if rank == 0:
            with open(
                f"./Gemm2d/TempBenchmarks/N16-n1-LocalMultiplication-{get_date_string()}.csv",
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                # equal = matrices_equal(standard_multiply, out)
                writer.writerow(
                    [
                        "LocalMultiplication",
                        size,
                        m,
                        n,
                        k,
                        elapsed_time,
                    ]
                )


def initial_benchmark():
    if not os.path.exists("./Gemm2d/TempBenchmarks"):
        os.makedirs("./Gemm2d/TempBenchmarks")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    prow = 4
    pcol = 4

    row_comm = comm.Split(rank // pcol, rank)
    col_comm = comm.Split(rank % pcol, rank)

    J = row_comm.Get_rank()
    I = col_comm.Get_rank()

    num_iterations = 100
    m = 160
    k = 160
    n = 160

    a_tile_row = m // prow
    a_tile_col = k // pcol

    b_tile_row = k // prow
    b_tile_col = n // pcol

    c_tile_row = m // prow
    c_tile_col = n // pcol

    for i in range(num_iterations):
        if rank == 0:
            print(i, flush=True)

        A_I = generate_matrix(a_tile_row, a_tile_col, -10, 10)
        B_I = generate_matrix(b_tile_row, b_tile_col, -10, 10)
        C_I = generate_matrix(c_tile_row, c_tile_col, -10, 10)

        start_time = MPI.Wtime()
        AG_A_COL_X_AG_B_ROW(A_I, B_I, C_I, row_comm, col_comm, m, k, n, 4, 4, I, J)
        elapsed_time = MPI.Wtime() - start_time

        gc.collect()

        if rank == 0:
            with open(
                f"./Gemm2d/TempBenchmarks/N16-n1-Gemm2dInitial-{get_date_string()}.csv",
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                # equal = matrices_equal(standard_multiply, out)
                writer.writerow(
                    [
                        "AG_AG_COL_X_AG_B_ROW",
                        size,
                        m,
                        n,
                        k,
                        calculate_throughput(elapsed_time, m, k, n),
                        elapsed_time,
                    ]
                )


def AG_A_COL_B_ROW_driver(m, k, n, prow, pcol):
    # only works for ag a col ag b row as of now
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert (
        prow * pcol == size
    ), f"The number of processors must be turned into a {prow}x{pcol} grid."

    # dont need these with the padding
    # assert k % prow == 0 and k % pcol == 0, "processors do not split k well"
    # assert k % (pcol * prow) == 0, "unable to properly subtile the matrices"
    # assert m % prow == 0, "processors do not split m well"
    # assert n % pcol == 0, "processors do not split n well"

    A_comm = col_block_comm(comm, pcol, rank)  # comm.Split(rank // pcol, rank)
    B_comm = col_comm(comm, pcol, rank)  # comm.Split(rank % pcol, rank)

    # the local row and column processor indices
    # these are flipped cuz its like a communicator for an entire row or column
    # J = A_comm.Get_rank()
    # I = B_comm.Get_rank()

    # even though this is done on each processor we get the same matrix cuz the seed is the same
    A = generate_matrix(m, k, -10, 10)
    B = generate_matrix(k, n, -10, 10)
    C = np.zeros((m, n), dtype=MATRIX_DTYPE)

    A_row_pad = pad_amount(m, prow)
    K_pad = pad_amount(k, pcol * prow)
    B_col_pad = pad_amount(n, pcol)

    A = pad_matrix_with_zeros(A, A_row_pad, K_pad)
    B = pad_matrix_with_zeros(B, K_pad, B_col_pad)
    C = pad_matrix_with_zeros(C, A_row_pad, B_col_pad)

    # rank_print(f"A:\n{A}\nB:\n{B}")

    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]

    A_width = k // (pcol * prow)
    # parallel_print(A_width)
    # A_I = A[
    #     I * (m // prow) : (I + 1) * (m // prow),
    #     get_step_indices(J * A_width, A.shape[1], pcol, A_width),
    # ].copy()
    A_I = block_cyclic(A, prow, pcol, B_comm.Get_rank(), A_comm.Get_rank())
    # B_I = B[
    #     I * (k // prow) : (I + 1) * (k // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    # ].copy()
    # C_I = C[
    #     I * (m // prow) : (I + 1) * (m // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    # ].copy()
    B_I = row_major_distribution(B, prow, pcol, rank)
    C_I = row_major_distribution(C, prow, pcol, rank)

    # parallel_print(f"AIJ:\n{AIJ}\nBIJ\n{BIJ}")
    AG_A_COL_X_AG_B_ROW(A_I, B_I, C_I, A_comm, B_comm, m, k, n, prow, pcol)

    comm.Barrier()

    row_gather = A_comm.gather(C_I, root=0)
    if A_comm.Get_rank() == 0:
        row_gather = np.hstack(row_gather)
        result = B_comm.gather(row_gather, root=0)
        if rank == 0:
            actual = remove_padding(np.vstack(result), A_row_pad, B_col_pad)
            expected = remove_padding(np.matmul(A, B) + C, A_row_pad, B_col_pad)
            # parallel_print(f"Expected:\n{expected}")
            # parallel_print(f"Actual:\n{actual}")
            parallel_print(f"Equal: {np.all(np.isclose(expected, actual))}")
    # if not os.path.exists("./Gemm2d/TempBenchmarks"):
    #     os.makedirs("./Gemm2d/TempBenchmarks")


def AG_A_COL_B_ROW_driver_2(m, k, n, prow, pcol):
    pass


def AG_A_COL_B_COL_driver(m, k, n, prow, pcol):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert n % size == 0, "n must be divisible by the number of processors"
    assert k % pcol == 0, "pcol must split k"
    assert m % prow == 0, "prow must split m"

    # A_comm = comm.Split(rank % prow, rank)
    # magic formula idk why it works
    # B_comm = comm.Split(rank // (size // pcol), rank)
    A_comm = row_comm(comm, prow, rank)
    B_comm = row_block_comm(comm, prow, rank)

    # parallel_print(f"{rank} - {A_comm.Get_rank()}")

    A = generate_matrix(m, k, -10, 10)
    B = generate_matrix(k, n, -10, 10)
    C = np.zeros((m, n))

    # rank_print(f"A:\n{A}")
    # rank_print(f"B\n{B}")

    # A_I = get_subtile2(A, prow, pcol, rank % prow, rank // prow).copy()
    # B_I = get_subtile2(B, 1, size, 0, rank).copy()
    # C_I = get_subtile2(C, prow, pcol, rank % prow, rank // prow).copy()

    A_I = col_major_distribution(A, prow, pcol, rank)
    B_I = pure_column_distribution(B, size, rank)
    C_I = col_major_distribution(C, prow, pcol, rank)

    AG_A_COL_X_AG_B_COL(A_I, B_I, C_I, A_comm, B_comm, m, k, n, prow, pcol)

    comm.Barrier()

    col_comm = comm.Split(A_comm.Get_rank(), rank)

    A_gather = A_comm.gather(C_I, root=0)
    if A_comm.Get_rank() == 0:
        A_row_gather = np.hstack(A_gather)
        result = col_comm.gather(A_row_gather, 0)
        if rank == 0:
            actual = np.vstack(result)
            expected = np.matmul(A, B) + C
            # parallel_print(f"Expected:\n{expected}")
            # parallel_print(f"Actual:\n{actual}")
            parallel_print(f"Equal: {np.all(np.isclose(expected, actual))}")


def AG_A_ROW_AG_B_ROW_driver(m, k, n, prow, pcol):
    # 7
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert m % size == 0, "m must be divisible by the number of processors"
    assert k % prow == 0, "prow must split k"
    assert n % pcol == 0, "pcol must split n"

    # A_comm = comm.Split(rank // pcol, rank)
    # B_comm = comm.Split(rank % pcol, rank)
    A_comm = col_block_comm(comm, pcol, rank)
    B_comm = col_comm(comm, pcol, rank)

    A = generate_matrix(m, k, -10, 10)
    B = generate_matrix(k, n, -10, 10)
    C = np.zeros((m, n))

    # rank_print(f"A:\n{A}")
    # rank_print(f"B\n{B}")

    # A_I = get_subtile2(A, size, 1, rank, 0).copy()
    # B_I = get_subtile2(B, prow, pcol, rank // pcol, rank % pcol).copy()
    # C_I = get_subtile2(C, prow, pcol, rank // pcol, rank % pcol).copy()
    A_I = pure_row_distribution(A, size, rank)
    B_I = row_major_distribution(B, prow, pcol, rank)
    C_I = row_major_distribution(C, prow, pcol, rank)

    # rank_print(f"A_I:\n{A_I}\nB_I:\n{B_I}")

    AG_A_ROW_X_AG_B_ROW(A_I, B_I, C_I, A_comm, B_comm, m, k, n, prow, pcol)

    comm.Barrier()

    row_comm = comm.Split(B_comm.Get_rank(), rank)

    B_gather = B_comm.gather(C_I, 0)
    if B_comm.Get_rank() == 0:
        B_row_gather = np.vstack(B_gather)
        result = row_comm.gather(B_row_gather, 0)
        if rank == 0:
            actual = np.hstack(result)
            expected = np.matmul(A, B) + C
            # parallel_print(f"Expected:\n{expected}")
            # parallel_print(f"Actual:\n{actual}")
            parallel_print(f"Equal: {np.all(np.isclose(expected, actual))}")

    # rank_print(f"Expected:\n{np.matmul(A, B) + C}")


def AG_A_ROW_RS_C_COL(m, k, n, prow, pcol):
    # 8
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert m % prow == 0, "prow must split m"
    assert k % pcol == 0, "pcol must split k"
    assert n % size == 0, "n must be divisible by the number of processors"

    A_comm = col_comm(comm, pcol, rank)  # comm.Split(rank % pcol, rank)
    C_comm = col_block_comm(comm, pcol, rank)  # comm.Split(rank // pcol, rank)

    A = generate_matrix(m, k, -10, 10)
    B = generate_matrix(k, n, -10, 10)
    C = np.zeros((m, n))

    # rank_print(f"A:\n{A}")
    # rank_print(f"B\n{B}")

    # A_I = get_subtile2(A, prow, pcol, rank // pcol, rank % pcol).copy()
    # B_I = get_subtile2(B, pcol, prow, rank % pcol, rank // pcol).copy()
    # C_I = get_subtile2(C, 1, size, 0, rank).copy()
    A_I = row_major_distribution(A, prow, pcol, rank)
    B_I = col_major_distribution(
        B, pcol, prow, rank
    )  # THE PROW AND PCOL ARE FLIPPED THIS IS IMPORTANT
    C_I = pure_column_distribution(C, size, rank)

    # rank_print(f"A_I:\n{A_I}\nB_I:\n{B_I}")

    AG_A_ROW_X_RS_C_COL(A_I, B_I, C_I, A_comm, C_comm, m, k, n, prow, pcol)

    comm.Barrier()
    gather = comm.gather(C_I, 0)
    if rank == 0:
        actual = np.hstack(gather)
        expected = np.matmul(A, B) + C
        # parallel_print(f"Expected:\n{expected}")
        # parallel_print(f"Actual:\n{actual}")
        parallel_print(f"Equal: {np.all(np.isclose(expected, actual))}")

    # rank_print(f"Expected:\n{np.matmul(A, B) + C}")


def main():
    set_numpy_seed(5)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    AG_A_COL_B_COL_FLAG = True
    AG_A_COL_B_ROW_FLAG = True
    AG_A_ROW_B_ROW_FLAG = True
    AG_A_ROW_C_COL_FLAG = True

    # Different test suites
    if size == 2:
        if AG_A_COL_B_COL_FLAG:
            AG_A_COL_B_COL_driver(6, 6, 6, 1, 2)
            AG_A_COL_B_COL_driver(6, 6, 6, 2, 1)
        if AG_A_COL_B_ROW_FLAG:
            AG_A_COL_B_ROW_driver(3, 6, 4, 1, 2)
            AG_A_COL_B_ROW_driver(120, 6, 4, 2, 1)
        if AG_A_ROW_B_ROW_FLAG:
            AG_A_ROW_AG_B_ROW_driver(8, 8, 8, 1, 2)
            AG_A_ROW_AG_B_ROW_driver(8, 8, 8, 2, 1)
        if AG_A_ROW_C_COL_FLAG:
            AG_A_ROW_RS_C_COL(4, 4, 4, 1, 2)
            AG_A_ROW_RS_C_COL(4, 4, 4, 2, 1)
            AG_A_ROW_RS_C_COL(100, 100, 100, 1, 2)
            AG_A_ROW_RS_C_COL(100, 100, 100, 2, 1)

    if size == 6:
        if AG_A_COL_B_COL_FLAG:
            AG_A_COL_B_COL_driver(3, 2, 6, 3, 2)
            AG_A_COL_B_COL_driver(2, 3, 6, 2, 3)
        if AG_A_COL_B_ROW_FLAG:
            AG_A_COL_B_ROW_driver(3, 6, 4, 3, 2)
            AG_A_COL_B_ROW_driver(3 * 500, 6 * 3, 4, 3, 2)
            AG_A_COL_B_ROW_driver(2, 2 * 3, 3, 2, 3)
            AG_A_COL_B_ROW_driver(3, 10, 4, 2, 3)  # padding needed
            AG_A_COL_B_ROW_driver(4, 12, 6, 2, 3)
        if AG_A_ROW_B_ROW_FLAG:
            AG_A_ROW_AG_B_ROW_driver(6, 3, 4, 3, 2)
            AG_A_ROW_AG_B_ROW_driver(6, 4, 3, 2, 3)
        if AG_A_ROW_C_COL_FLAG:
            AG_A_ROW_RS_C_COL(3, 2, 6, 3, 2)
            AG_A_ROW_RS_C_COL(2, 3, 6, 2, 3)

    if size == 9:
        if AG_A_COL_B_COL_FLAG:
            AG_A_COL_B_COL_driver(3, 3, 9, 3, 3)
            AG_A_COL_B_COL_driver(27, 27, 27, 3, 3)
        if AG_A_COL_B_ROW_FLAG:
            AG_A_COL_B_ROW_driver(9, 9, 9, 3, 3)
        if AG_A_ROW_B_ROW_FLAG:
            AG_A_ROW_AG_B_ROW_driver(9, 9, 9, 3, 3)
            AG_A_ROW_AG_B_ROW_driver(27, 90, 3 * 29, 3, 3)
            AG_A_ROW_AG_B_ROW_driver(18, 9, 9, 3, 3)
        if AG_A_ROW_C_COL_FLAG:
            AG_A_ROW_RS_C_COL(3, 3, 9, 3, 3)
            AG_A_ROW_RS_C_COL(3 * 11, 3 * 23, 9 * 17, 3, 3)

    if size == 12:
        if AG_A_COL_B_COL_FLAG:
            AG_A_COL_B_COL_driver(4, 3, 12, 4, 3)
            AG_A_COL_B_COL_driver(3, 4, 12, 3, 4)
            AG_A_COL_B_COL_driver(3 * 13, 4 * 17, 12 * 150, 3, 4)
        if AG_A_COL_B_ROW_FLAG:
            AG_A_COL_B_ROW_driver(4 * 3, 12 * 7, 3 * 2, 4, 3)
            AG_A_COL_B_ROW_driver(5, 15, 5, 4, 3)  # padding needed
            AG_A_COL_B_ROW_driver(3 * 16, 3 * 4 * 7, 4 * 17, 3, 4)
        if AG_A_ROW_B_ROW_FLAG:
            AG_A_ROW_AG_B_ROW_driver(12 * 7, 4 * 11, 3 * 17, 4, 3)
            AG_A_ROW_AG_B_ROW_driver(12 * 13, 3 * 23, 4 * 29, 3, 4)
        if AG_A_ROW_C_COL_FLAG:
            AG_A_ROW_RS_C_COL(3, 4, 12, 3, 4)
            AG_A_ROW_RS_C_COL(4, 3, 12, 4, 3)
            AG_A_ROW_RS_C_COL(3 * 29, 4 * 11, 12 * 17, 3, 4)
            AG_A_ROW_RS_C_COL(4 * 7, 3 * 29, 12 * 17, 4, 3)

    if size == 16:
        if AG_A_COL_B_COL_FLAG:
            AG_A_COL_B_COL_driver(1000, 1000, 1600, 4, 4)
        if AG_A_COL_B_ROW_FLAG:
            AG_A_COL_B_ROW_driver(1600, 1600, 1600, 4, 4)
        if AG_A_ROW_C_COL_FLAG:
            AG_A_ROW_RS_C_COL(4, 4, 16, 4, 4)
            AG_A_ROW_RS_C_COL(4 * 31, 4 * 29, 16 * 11, 4, 4)

    if size == 25:
        if AG_A_COL_B_COL_FLAG:
            AG_A_COL_B_COL_driver(1000, 1000, 1600, 5, 5)
        if AG_A_COL_B_ROW_FLAG:
            AG_A_COL_B_ROW_driver(5, 5 * 5, 5, 5, 5)
            AG_A_COL_B_ROW_driver(14, 37, 17, 5, 5)
        if AG_A_ROW_C_COL_FLAG:
            AG_A_ROW_RS_C_COL(5, 5, 25, 5, 5)
            AG_A_ROW_RS_C_COL(5 * 37, 5 * 23, 25 * 7, 5, 5)


if __name__ == "__main__":
    # initial_benchmark()
    # one_processor_test()
    main()
