import numpy as np
from mpi4py import MPI

from GemmUtil.constants import MATRIX_DTYPE

from GemmUtil.helper_general import generate_matrix, parallel_print

from GemmUtil.helper_2d import (
    pad_amount,
    remove_padding,
    pad_matrix_with_zeros,
    get_step_indices,
    get_subtile,
    set_subtile,
)

from .comm_hidden_2d import AG_A_COL_X_AG_B_COL, AG_A_COL_X_AG_B_ROW


def mult(A, B, C, row_comm, col_comm, m, k, n, prow, pcol, I, J):
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

            # allgather A col
            if j != pcol - 1:
                # sending subtile A
                A_send_request = row_comm.Isend(
                    np.ascontiguousarray(A_curr, dtype=MATRIX_DTYPE),
                    (row_rank - 1) % pcol,
                )
                A_next = np.empty(A_curr.shape)
                A_receive_request = row_comm.Irecv(A_next, (row_rank + 1) % pcol)

            B_curr = get_subtile(B, inner_index, pcol, "r")

            # allgather B row
            if i != prow - 1:
                # we only send the part of B that we have used up and won't need for this iteration anymore
                B_send_request = col_comm.Isend(
                    np.ascontiguousarray(B_curr, dtype=MATRIX_DTYPE),
                    (col_rank - 1) % prow,
                )
                B_next = np.empty(B_curr.shape)
                B_receive_request = col_comm.Irecv(B_next, (col_rank + 1) % prow)

            # parallel_print(f"Step ({i},{j})\nA_curr:\n{A_curr}\nB_curr\n{B_curr}\n", flush=True)
            C += np.matmul(A_curr, B_curr)

            # allgather A col
            if j != pcol - 1:
                MPI.Request.Waitall([A_send_request, A_receive_request])
                print(
                    f"{A.dtype} {B.dtype}, {C.dtype}, {A_curr.dtype} {B_curr.dtype}",
                    flush=True,
                )

                set_subtile(A, outer_index, prow, "c", A_next)

            # allgather B row
            if i != prow - 1:
                # see if this can be moved to the outerloop since we don't need it until then
                MPI.Request.Waitall([B_send_request, B_receive_request])
                set_subtile(B, inner_index, pcol, "r", B_next)

            # updates which subtile of the B matrix we use since we now have a different chunk of A after communication
            inner_index = (inner_index + 1) % pcol

        # make sure have B here

        # change the subtile of A since the entire B tile has been shuffled
        outer_index = (outer_index + 1) % prow


def test_matrix_multiply(algorithm, m, k, n, prow, pcol):
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

    A_row_pad = pad_amount(m, prow)
    K_pad = pad_amount(k, pcol * prow)
    B_col_pad = pad_amount(n, pcol)

    A = pad_matrix_with_zeros(A, A_row_pad, K_pad)
    B = pad_matrix_with_zeros(B, K_pad, B_col_pad)
    C = pad_matrix_with_zeros(C, A_row_pad, B_col_pad)
    # if rank == 0:
    #     print(A)
    #     print(B)
    #     print(C)

    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]

    A_width = k // (pcol * prow)
    # parallel_print(A_width)
    AIJ = A[
        I * (m // prow) : (I + 1) * (m // prow),
        get_step_indices(J * A_width, A.shape[1], pcol, A_width),
    ].copy()
    BIJ = B[
        I * (k // prow) : (I + 1) * (k // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()
    CIJ = C[
        I * (m // prow) : (I + 1) * (m // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()

    print("made it", flush=True)

    mult(AIJ, BIJ, CIJ, row_comm, col_comm, m, k, n, prow, pcol, I, J)


def main():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    print(f"size is {size}")

    # different suites
    if size == 2:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 3, 6, 4, 1, 2)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 120, 6, 4, 2, 1)
    if size == 6:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 3, 6, 4, 3, 2)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 3 * 500, 6 * 3, 4, 3, 2)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 2, 2 * 3, 3, 2, 3)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 3, 10, 4, 2, 3)  # padding needed
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 4, 12, 6, 2, 3)
    if size == 12:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 4 * 3, 12 * 7, 3 * 2, 4, 3)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 5, 15, 5, 4, 3)  # padding needed
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 3 * 16, 3 * 4 * 7, 4 * 17, 3, 4)
    if size == 9:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 9, 9, 9, 3, 3)
    if size == 25:
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 5, 5 * 5, 5, 5, 5)
        test_matrix_multiply(AG_A_COL_X_AG_B_ROW, 14, 37, 17, 5, 5)


if __name__ == "__main__":
    main()
