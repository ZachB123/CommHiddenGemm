import numpy as np
from mpi4py import MPI

from CommHiddenGemm.Util.util import (
    MATRIX_DTYPE,
    matrices_equal,
    get_step_indices,
    processor_rank_from_IJ,
)


# Function to print with rank and color
def parallel_print(message, flush=False):
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    def get_color_code(rank, num_colors):
        return f"\033[38;5;{rank % num_colors}m"
    
    color_code = get_color_code(rank, size)

    print(f"{color_code}[{rank}/{size - 1}]\n{message}\033[0m", flush=True)

def proc_2d_indices_to_proc(I, J, prow, pcol):
    return I * pcol + J

# first test set is on 3x2 grid dimensions 3,6,4
MATRIX_A = np.array(
    [
        [1.0, -4.0, -9.0, -2.0, 0.0, -5.0],
        [-2.0, -10.0, 4.0, 9.0, -7.0, -6.0],
        [3.0, -10.0, -2.0, -4.0, -5.0, 3.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_B = np.array(
    [
        [5.0, 3.0, -7.0, 6.0],
        [-6.0, -2.0, -1.0, 1.0],
        [-4.0, 9.0, 7.0, 4.0],
        [4.0, -1.0, -9.0, 1.0],
        [2.0, 6.0, -7.0, -5.0],
        [4.0, -7.0, 0.0, -8.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_C = np.zeros((3, 4), MATRIX_DTYPE)

# second is for 4x3 grid of processors dimensions 4,12,3
MATRIX_A2 = np.array(
    [
        [ -5. ,  1. ,  2. , -2. , -1. ,  1. , -5. ,  5., -10.,  6. , -9. ,  2. ],
        [ -3. ,  3. , -4. ,  8. , -5. ,  8. ,  1. ,  0.,   4. ,  8. , -6. , -1. ],
        [  7. ,-10. ,  3. , -1. , -1. , -3. , -9.,-10.,   7. , -2. ,  3. ,  9. ],
        [  5. ,  0. , -2. , -3. , -7. , -4. ,  7., -7.,  -6. ,  7. ,  1. ,  2. ],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_B2 = np.array(
    [
        [ -5.,   1.,   2.],
        [ -2.,  -1.,   1.],
        [ -5.,   5., -10.],
        [  6.,  -9.,   2.],
        [ -3.,   3.,  -4.],
        [  8.,  -5.,   8.],
        [  1.,   0.,   4.],
        [  8.,  -6.,  -1.],
        [  7., -10.,   3.],
        [ -1.,  -1.,  -3.],
        [ -9., -10.,   7.],
        [ -2.,   3.,   9.]
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_C2 = np.zeros((4,3), MATRIX_DTYPE)

# 3 is for 3x2 grid of processors dimensions 6,6,4
MATRIX_A3 = np.array(
    [
        [ -5.0,   1.0,   2.0,  -2.0,  -1.0,   1.0],
        [ -5.0,   5.0, -10.0,   6.0,  -9.0,   2.0],
        [ -3.0,   3.0,  -4.0,   8.0,  -5.0,   8.0],
        [  1.0,   0.0,   4.0,   8.0,  -6.0,  -1.0],
        [  7.0, -10.0,   3.0,  -1.0,  -1.0,  -3.0],
        [ -9.0, -10.0,   7.0,  -2.0,   3.0,   9.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_B3 = np.array(
    [
        [  5.0,   0.0,  -2.0,  -3.0],
        [ -7.0,  -4.0,   7.0,  -7.0],
        [ -6.0,   7.0,   1.0,   2.0],
        [  6.0,   3.0,   9.0,  -1.0],
        [  8.0,   5.0, -10.0,  -6.0],
        [  5.0,  -8.0,  -3.0,  -2.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_C3 = np.zeros((6,4), dtype=MATRIX_DTYPE)

# 4 is for 3x2 grid of processors dimensionts 3,12,4
MATRIX_A4 = np.array(
    [
        [ -5.0,   1.0,   2.0,  -2.0,  -1.0,   1.0,  -5.0,   5.0, -10.0,   6.0,  -9.0,   2.0],
        [ -3.0,   3.0,  -4.0,   8.0,  -5.0,   8.0,   1.0,   0.0,   4.0,   8.0,  -6.0,  -1.0],
        [  7.0, -10.0,   3.0,  -1.0,  -1.0,  -3.0,  -9.0, -10.0,   7.0,  -2.0,   3.0,   9.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_B4 = np.array(
    [
        [  5.0,   0.0,  -2.0,  -3.0],
        [ -7.0,  -4.0,   7.0,  -7.0],
        [ -6.0,   7.0,   1.0,   2.0],
        [  6.0,   3.0,   9.0,  -1.0],
        [  8.0,   5.0, -10.0,  -6.0],
        [  5.0,  -8.0,  -3.0,  -2.0],
        [ -1.0,  -7.0,  -3.0,  -6.0],
        [ -5.0,   9.0,  -4.0,  -2.0],
        [-10.0,  -8.0,   0.0,   5.0],
        [  5.0,  -3.0,   9.0,   0.0],
        [  4.0, -10.0,  -9.0,   7.0],
        [  3.0,  -7.0, -10.0,   3.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_C4 = np.zeros((3,4), dtype=MATRIX_DTYPE)

# 5 is for 3x3 grid of processors with dimensions 3,3,3
MATRIX_A5 = np.array(
    [
        [ -5.0,   1.0,   2.0],
        [ -2.0,  -1.0,   1.0],
        [ -5.0,   5.0, -10.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_B5 = np.array(
    [
        [  6.0,  -9.0,   2.0],
        [ -3.0,   3.0,  -4.0],
        [  8.0,  -5.0,   8.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_C3 = np.zeros((3,3), dtype=MATRIX_DTYPE)

# processor grid 2x3 dimensions 2,6,3
MATRIX_A6 = np.array(
    [
        [ -5.0,   1.0,   2.0,  -2.0,  -1.0,   1.0],
        [ -5.0,   5.0, -10.0,   6.0,  -9.0,   2.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_B6 = np.array(
    [
        [ -3.0,   3.0,  -4.0],
        [  8.0,  -5.0,   8.0],
        [  1.0,   0.0,   4.0],
        [  8.0,  -6.0,  -1.0],
        [  7.0, -10.0,   3.0],
        [ -1.0,  -1.0,  -3.0],
    ],
    dtype=MATRIX_DTYPE,
)

MATRIX_C6 = np.zeros((2,3), dtype=MATRIX_DTYPE)


def get_subtile(tile, slice, n_slices, direction):
    # n_slices is like how many individual slices we will eventually make, slice is the index for n_slices
    # direction is a string for rows or columns and value will be r or c
    # if r we slice across the rows, so all columns preserved and vice versa
    tile = np.atleast_2d(tile)
    if direction == "r":
        rows = tile.shape[0]
        assert rows % n_slices == 0, f"Tiles rows {rows} are not divisible by slices {n_slices} requested"
        width = rows // n_slices
        return tile[width * slice : width * (slice + 1) , :]
    else:
        cols = tile.shape[1]
        assert cols % n_slices == 0, f"Tiles columns {cols} are not divisible by slices {n_slices} requested"
        width = cols // n_slices
        return tile[:, width * slice : width * (slice + 1)]
    
def set_subtile(tile, slice, n_slices, direction, block):
    tile = np.atleast_2d(tile)
    if direction == "r":
        rows = tile.shape[0]
        assert rows % n_slices == 0, f"Tiles rows {rows} are not divisible by slices {n_slices} requested"
        width = rows // n_slices
        tile[width * slice : width * (slice + 1) , :] = block
    else:
        cols = tile.shape[1]
        assert cols % n_slices == 0, f"Tiles columns {cols} are not divisible by slices {n_slices} requested"
        width = cols // n_slices
        tile[:, width * slice : width * (slice + 1)] = block


def AG_A_COL_X_AG_B_ROW(A, B, C, row_comm, col_comm, m, k, n, prow, pcol, I, J):
    row_rank = row_comm.Get_rank()
    col_rank = col_comm.Get_rank()

    # which chunk of the A matrix we are using
    outer_index = I

    # loop shuffles B around across the rows (allgather B row)
    for i in range(prow):

        # this is what subtile index of the B matrix we are using
        inner_index = J

        # this loop shuffles A across the columns (allgather A col)
        for j in range(pcol):

            A_curr = get_subtile(A, outer_index, prow, "c")
            B_curr = get_subtile(B, inner_index, pcol, "r")

            # parallel_print(f"Step ({i},{j})\nA_curr:\n{A_curr}\nB_curr\n{B_curr}\n", flush=True)

            # allgather B row
            if i != prow - 1:
                # we only send the part of B that we have used up and won't need for this iteration anymore
                B_send_request = col_comm.Isend(np.ascontiguousarray(B_curr), (col_rank - 1) % prow)
                B_next = np.empty(B_curr.shape)
                B_receive_request = col_comm.Irecv(B_next, (col_rank + 1) % prow)

            # allgather A col
            if j != pcol - 1:
                # sending subtile A
                A_send_request = row_comm.Isend(np.ascontiguousarray(A_curr), (row_rank - 1) % pcol)
                A_next = np.empty(A_curr.shape)
                A_receive_request = row_comm.Irecv(A_next, (row_rank + 1) % pcol)

            C += np.matmul(A_curr, B_curr)

            # allgather A col
            if j != pcol - 1:
                MPI.Request.Waitall([A_send_request, A_receive_request])
                set_subtile(A, outer_index, prow, "c", A_next)

            # allgather B row
            if i != prow - 1:
                # see if this can be moved to the outerloop since we don't need it until then
                MPI.Request.Waitall([B_send_request, B_receive_request])
                set_subtile(B, inner_index, pcol, "r", B_next)

            # updates which subtile of the B matrix we use since we now have a different chunk of A after communication
            inner_index = (inner_index + 1) % pcol

        # change the subtile of A since the entire B tile has been shuffled
        outer_index = (outer_index + 1) % prow



def test_matrix_multiply(algorithm, m, k, n, prow, pcol):


    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert (
        prow * pcol == size
    ), f"The number of processors must be turned into a {prow}x{pcol} grid."

    assert k % prow == 0 and k % pcol == 0, "processors do not split k well"
    assert k % (pcol * prow) == 0, "unable to properly subtile the matrices"
    assert m % prow == 0, "processors do not split m well"
    assert n % pcol == 0, "processors do not split n well"

    row_comm = comm.Split(rank // pcol, rank)
    col_comm = comm.Split(rank % pcol, rank)

    # the local row and column processor indices
    # these are flipped cuz its like a communicator for an entire row or column
    J = row_comm.Get_rank()
    I = col_comm.Get_rank()

    # even though this is done on each processor we get the same matrix cuz the seed is the same
    A = generate_matrix(m, k, -10, 10)
    B = generate_matrix(k, n, -10, 10)
    C = np.zeros((m, n), dtype=MATRIX_DTYPE)
    # A = MATRIX_A
    # B = MATRIX_B
    # C = MATRIX_C
    # A = MATRIX_A2
    # B = MATRIX_B2
    # C = MATRIX_C2
    # A = MATRIX_A3
    # B = MATRIX_B3
    # C = MATRIX_C3
    # A = MATRIX_A4
    # B = MATRIX_B4
    # C = MATRIX_C4
    # A = MATRIX_A5
    # B = MATRIX_B5
    # C = MATRIX_C5
    # A = MATRIX_A6
    # B = MATRIX_B6
    # C = MATRIX_C6

    A_width = k // (pcol * prow)
    # parallel_print(A_width)
    AIJ = A[
        I * (m // prow) : (I + 1) * (m // prow),
        get_step_indices(J * A_width, A.shape[1], pcol, A_width)
    ].copy()
    BIJ = B[
        I * (k // prow) : (I + 1) * (k // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()
    CIJ = C[
        I * (m // prow) : (I + 1) * (m // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()

    # parallel_print(f"AIJ:\n{AIJ}\nBIJ\n{BIJ}")
    algorithm(AIJ, BIJ, CIJ, row_comm, col_comm, m, k, n, prow, pcol, I, J)
    
    expected = np.matmul(A, B) + C
    comm.barrier()
    if rank == 0:
        print(f"Expected:\n{expected}")

    row_gather = row_comm.gather(CIJ, root=0)
    if row_comm.Get_rank() == 0:
        row_gather = np.hstack(row_gather)
        result = col_comm.gather(row_gather, root=0)
        if rank == 0:
            actual = np.vstack(result)
            parallel_print(f"Actual:\n{actual}")
            parallel_print(f"Equal: {np.all(np.isclose(expected, actual))}")


def main():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # different suites
    if size == 6:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 3, 6, 4, 3, 2)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 3 * 500, 6 * 3, 4, 3, 2)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 2, 2 * 3, 3, 2, 3)
    if size == 12:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 4 * 3, 12 * 7, 3 * 2, 4, 3)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 3 * 16, 3 * 4 * 7, 4 * 17, 3, 4)
    if size == 9:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 9, 9, 9, 3, 3)
    if size == 25:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 5, 5 * 5, 5, 5, 5)

    # print(generate_matrix(2, 6, -10, 10))
    # print(generate_matrix(6, 3, -10, 10))



    return
    # print(np.matmul(MATRIX_A2, MATRIX_B2))
    # print(get_step_indices(1, 12, 3, 2))
    # AIJ = MATRIX_A2[
    # 0, #* (m // prow) : (I + 1) * (m // prow),
    # # get_step_indices(J * (ktile // pcol), MATRIX_A.shape[1], (k // prow) // 2),
    # get_step_indices(1, MATRIX_A2.shape[1], 3, 1)
    # ].copy()
    # print(AIJ)
    # return

    # processor distribution
    prow = 3
    pcol = 2
    # matrix dimensions
    m = 3#4
    k = 6#12
    n = 4#3

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert (
        prow * pcol == size
    ), f"The number of processors must be turned into a {prow}x{pcol} grid."

    assert k % prow == 0 and k % pcol == 0, "processors do not split k well"
    assert m % prow == 0, "processors do not split m well"
    assert n % pcol == 0, "processors do not split n well"

    ktile = k // prow

    row_comm = comm.Split(rank // pcol, rank)
    col_comm = comm.Split(rank % pcol, rank)

    # the local row and column processor indices
    # these are flipped cuz its like a communicator for an entire row or column
    J = row_comm.Get_rank()
    I = col_comm.Get_rank()

    # print(f"Rank: {rank}, I: {I}, J: {J}")


    AIJ = MATRIX_A[
        I, #* (m // prow) : (I + 1) * (m // prow),
        # get_step_indices(J * (ktile // pcol), MATRIX_A.shape[1], (k // prow) // 2),
        get_step_indices(J, MATRIX_A.shape[1], pcol, 1)
    ].copy()
    BIJ = MATRIX_B[
        I * (k // prow) : (I + 1) * (k // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()
    CIJ = MATRIX_C[
        I * (m // prow) : (I + 1) * (m // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()

    AG_A_COL_X_AG_B_ROW(AIJ, BIJ, CIJ, row_comm, col_comm, m, k, n, prow, pcol, I, J)
    
    comm.barrier()
    if rank == 0:
        print(f"Expected:\n{np.matmul(MATRIX_A,MATRIX_B) + MATRIX_C}")

    row_gather = row_comm.gather(CIJ, root=0)
    if row_comm.Get_rank() == 0:
        row_gather = np.hstack(row_gather)
        result = col_comm.gather(row_gather, root=0)
        if rank == 0:
            parallel_print(f"Actual:\n{np.vstack(result)}")


    # print(f"Rank: {rank}\n{CIJ}")

    # comm.Barrier()

    # if rank == 1:
    # print(f"YUH\n{MATRIX_A[I * (m // prow) : (I + 1) * (m // prow), get_step_indices(J * (ktile // pcol), MATRIX_A.shape[1], (k // prow) // 2)]}")


def generate_matrix(row, col, min, max):
    # [min, max)
    return np.random.randint(min, max, size=(row, col)).astype(np.float64, copy=False)


if __name__ == "__main__":
    main()
