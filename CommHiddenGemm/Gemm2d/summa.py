import numpy as np
from mpi4py import MPI

np.random.seed(420)
MATRIX_DTYPE = np.float64

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


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    m, k, n = MATRIX_A.shape[0], MATRIX_A.shape[1], MATRIX_C.shape[1]
    expected = MATRIX_C + np.matmul(MATRIX_A, MATRIX_B)

    num_rows = 3
    num_cols = 3
    assert (
        num_rows * num_cols == size
    ), f"The number of processors must be turned into a {num_rows}x{num_cols} grid."

    row_groups = []
    col_groups = []

    for i in range(num_rows):
        row_groups.append(
            comm.Create(comm.group.Incl(range(i * num_cols, (i + 1) * num_cols)))
        )
    for i in range(num_cols):
        col_groups.append(comm.Create(comm.group.Incl(range(i, size, num_rows))))

    for i in range(len(row_groups)):
        if row_groups[i] != MPI.COMM_NULL:
            print(f"Rank is {rank}. row group {i} rank is {row_groups[i].Get_rank()}")
    for i in range(len(col_groups)):
        if col_groups[i] != MPI.COMM_NULL:
            print(f"Rank is {rank}. col group {i} rank is {col_groups[i].Get_rank()}")

    # how should this be determined
    strip_width = 1
    assert (
        k % strip_width == 0
    ), "The strip dimension must divide along the K dimension."

    col_block = rank % num_cols
    row_block = rank // num_cols

    A_I = MATRIX_A[
        row_block * num_cols : (row_block + 1) * num_cols,
        col_block * num_rows : (col_block + 1) * num_rows,
    ].copy()
    B_I = MATRIX_A[
        row_block * num_cols : (row_block + 1) * num_cols,
        col_block * num_rows : (col_block + 1) * num_rows,
    ].copy()
    C_I = MATRIX_A[
        row_block * num_cols : (row_block + 1) * num_cols,
        col_block * num_rows : (col_block + 1) * num_rows,
    ].copy()

    strip_location = 0

    for i in range(k // strip_width):
        # check if the strip is in A_I
        Atmp = None
        if (
            strip_location >= row_block * num_cols
            and strip_location < (row_block + 1) * num_cols
        ):
            row_block[row_block].bcast()

        Btmp = None
        # check for B_I
        if (
            strip_location >= col_block * num_rows
            and strip_location < (col_block + 1) * num_rows
        ):
            pass

    print(f"Rank is {rank} and A_I is \n{A_I}\n\n")

    return


if __name__ == "__main__":
    main()

