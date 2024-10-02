import numpy as np
from mpi4py import MPI


from GemmUtil.helper_2d import get_subtile, set_subtile


def AG_A_COL_X_AG_B_ROW(A, B, C, row_comm, col_comm, m, k, n, prow, pcol, I, J):
    """
    Perform a distributed matrix multiplication of A (columns) and B (rows)
    using an allgather approach across rows and columns.

    Args:
        A (np.ndarray): A tile.
        B (np.ndarray): B tile.
        C (np.ndarray): C tile.
        row_comm (MPI.Comm): The MPI communicator for a row of the grid.
        col_comm (MPI.Comm): The MPI communicator for a column of the grid.
        m (int): Number of rows in A.
        k (int): Number of columns in A and rows in B.
        n (int): Number of columns in B.
        prow (int): Number of processor rows.
        pcol (int): Number of processor columns.
        I (int): Row index for current processor.
        J (int): Column- index for current processor.

    Returns:
        None: The output matrix C is modified in place.
    """
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
                    np.ascontiguousarray(A_curr), (row_rank - 1) % pcol
                )
                A_next = np.empty(A_curr.shape)
                A_receive_request = row_comm.Irecv(A_next, (row_rank + 1) % pcol)

            B_curr = get_subtile(B, inner_index, pcol, "r")

            # allgather B row
            if i != prow - 1:
                # we only send the part of B that we have used up and won't need for this iteration anymore
                B_send_request = col_comm.Isend(
                    np.ascontiguousarray(B_curr), (col_rank - 1) % prow
                )
                B_next = np.empty(B_curr.shape)
                B_receive_request = col_comm.Irecv(B_next, (col_rank + 1) % prow)

            # parallel_print(f"Step ({i},{j})\nA_curr:\n{A_curr}\nB_curr\n{B_curr}\n", flush=True)

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

        # make sure have B here

        # change the subtile of A since the entire B tile has been shuffled
        outer_index = (outer_index + 1) % prow


def AG_A_COL_X_AG_B_COL(A, B, C, row_comm, col_comm, m, k, n, prow, pcol, I, J):
    pass
