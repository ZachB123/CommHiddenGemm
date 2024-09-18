import numpy as np
from mpi4py import MPI

from CommHiddenGemm.Util.util import (
    MATRIX_DTYPE,
    matrices_equal,
)

np.random.seed(420)

MATRIX_A = np.array(
    [
        [-4.0, 9.0, 4.0, 0.0, -3.0, -4.0, 8.0, 0.0, 0.0],
        [-7.0, -3.0, -8.0, -9.0, 1.0, -5.0, -9.0, -10.0, 1.0],
        [1.0, 6.0, -1.0, 5.0, 4.0, 4.0, 8.0, 1.0, 9.0],
        [-8.0, -6.0, 8.0, -4.0, -2.0, -4.0, 7.0, -7.0, 3.0],
        [7.0, -2.0, -9.0, 9.0, 4.0, -4.0, 1.0, -3.0, 4.0],
        [-8.0, 3.0, 6.0, -7.0, 7.0, -3.0, -7.0, -9.0, -5.0],
        [-1.0, -7.0, 7.0, 1.0, -9.0, -1.0, -7.0, 3.0, 5.0],
        [4.0, -3.0, 3.0, -3.0, 5.0, 2.0, 7.0, 4.0, 2.0],
        [-2.0, 4.0, 2.0, -10.0, -4.0, -2.0, -10.0, 1.0, -3.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_B = np.array(
    [
        [7.0, -2.0, -4.0, 9.0, 4.0, 0.0, -2.0, 4.0, -4.0],
        [-7.0, -6.0, -10.0, 3.0, -5.0, -5.0, 5.0, 2.0, 1.0],
        [-2.0, -8.0, -5.0, -5.0, 3.0, 5.0, 5.0, -2.0, -1.0],
        [1.0, -1.0, -3.0, -1.0, -10.0, -6.0, 4.0, 7.0, 5.0],
        [8.0, -8.0, -5.0, 9.0, -6.0, 5.0, -7.0, 4.0, 4.0],
        [-9.0, 6.0, 5.0, 2.0, -8.0, 3.0, -7.0, -1.0, 8.0],
        [-2.0, 1.0, 1.0, 5.0, 0.0, -6.0, -1.0, 5.0, 6.0],
        [0.0, -4.0, -6.0, -2.0, -5.0, -2.0, 1.0, 8.0, 3.0],
        [-10.0, 5.0, -7.0, 3.0, 0.0, 3.0, -2.0, 8.0, -1.0],
    ],
    dtype=MATRIX_DTYPE,
)


MATRIX_C = np.zeros((9, 9), dtype=MATRIX_DTYPE)


def generate_matrix(row, col, min, max):
    # [min, max)
    return np.random.randint(min, max, size=(row, col)).astype(MATRIX_DTYPE, copy=False)


def get_local_block(matrix, local_i, local_j, row_block_size, col_block_size):
    return matrix[
        local_j * col_block_size : (local_j + 1) * col_block_size,
        local_i * row_block_size : (local_i + 1) * row_block_size,
    ].copy()


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
