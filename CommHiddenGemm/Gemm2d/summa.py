import numpy as np
from mpi4py import MPI

from GemmUtil.constants import (
    MATRIX_DTYPE,
    MATRIX_A_9_9 as MATRIX_A,
    MATRIX_B_9_9 as MATRIX_B,
    MATRIX_C_9_9 as MATRIX_C,
)

from GemmUtil.helper_general import matrices_equal

from GemmUtil.helper_2d import get_local_block


def main():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    m, k, n = MATRIX_A.shape[0], MATRIX_A.shape[1], MATRIX_C.shape[1]
    expected = MATRIX_C + np.matmul(MATRIX_A, MATRIX_B)

    # how the processors are split
    num_rows = 3
    num_cols = 3

    assert (
        num_rows * num_cols == size
    ), f"The number of processors must be turned into a {num_rows}x{num_cols} grid."

    row_comm = comm.Split(rank // num_cols, rank)
    col_comm = comm.Split(rank % num_cols, rank)

    # these can be used as like coordinates in the processor grid
    row_rank = row_comm.Get_rank()
    col_rank = col_comm.Get_rank()

    # print(f"Rank is: {rank}, Row Rank: {row_rank}, Col Rank: {col_rank}")

    # how should this be determined
    strip_width = 1
    assert (
        k % strip_width == 0
    ), "The strip dimension must divide along the K dimension."

    A_I = get_local_block(MATRIX_A, row_rank, col_rank, m // num_rows, k // num_cols)
    B_I = get_local_block(MATRIX_B, row_rank, col_rank, k // num_rows, n // num_cols)
    C_I = get_local_block(MATRIX_C, row_rank, col_rank, m // num_rows, n // num_cols)

    for strip_index in range(0, k, strip_width):
        # communicate the column section from A_I
        row_sending_rank = strip_index // num_cols

        if row_sending_rank == row_rank:
            A_data = A_I[
                :, strip_index % num_cols : (strip_index % num_cols) + 1
            ].copy()
        else:
            A_data = np.empty((m // num_rows, strip_width), dtype=MATRIX_DTYPE)

        row_comm.Bcast(A_data, root=row_sending_rank)

        col_sending_rank = strip_index // num_rows

        if col_sending_rank == col_rank:
            B_data = B_I[
                strip_index % num_rows : (strip_index % num_rows) + 1, :
            ].copy()
        else:
            B_data = np.empty((strip_width, n // num_cols), dtype=MATRIX_DTYPE)

        col_comm.Bcast(B_data, root=col_sending_rank)

        C_I = C_I + np.matmul(A_data, B_data)

    if rank == 0:
        print(f"expected:\n{expected}")

    row_gather = row_comm.gather(C_I, root=0)
    if row_rank == 0:
        full_rows = np.concatenate(row_gather, axis=1)
        full_gather = col_comm.gather(full_rows, root=0)
        if rank == 0:
            actual = np.concatenate(full_gather, axis=0)
            print(f"Actual:\n{actual}")
            print(f"Equal: {matrices_equal(actual, expected)}")


if __name__ == "__main__":
    main()
