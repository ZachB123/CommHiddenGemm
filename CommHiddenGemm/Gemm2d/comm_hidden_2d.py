import numpy as np
from mpi4py import MPI

from GemmUtil.constants import MPI_DTYPE, DEBUG_RANK

from GemmUtil.helper_general import parallel_print, rank_print

from GemmUtil.helper_2d import get_subtile, set_subtile, get_subtile2, set_subtile2


def AG_A_COL_X_AG_B_ROW(A, B, C, A_comm, B_comm, m, k, n, prow, pcol, I, J):
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

    A_rank = A_comm.Get_rank()
    B_rank = B_comm.Get_rank()

    # which chunk of the A matrix we are using
    outer_index = I

    # make the subtile blocks once to reuse and not reallocate space every time
    A_next = np.empty((m // prow, k // (prow * pcol)))
    B_next = np.empty((k // (prow * pcol), n // pcol))

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
                A_send_request = A_comm.Isend(
                    [np.ascontiguousarray(A_curr), MPI_DTYPE], (A_rank - 1) % pcol
                )
                A_receive_request = A_comm.Irecv(
                    [A_next, MPI_DTYPE], (A_rank + 1) % pcol
                )

            B_curr = get_subtile(B, inner_index, pcol, "r")

            # allgather B row
            if i != prow - 1:
                # we only send the part of B that we have used up and won't need for this iteration anymore
                B_send_request = B_comm.Isend(
                    [np.ascontiguousarray(B_curr), MPI_DTYPE], (B_rank - 1) % prow
                )
                B_receive_request = B_comm.Irecv(
                    [B_next, MPI_DTYPE], (B_rank + 1) % prow
                )

            debug_string = f"A_curr:\n{A_curr}\n" f"B_curr:\n{B_curr}\n"

            # rank_print(debug_string)

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


def AG_A_COL_X_AG_B_COL(A, B, C, A_comm, B_comm, m, k, n, prow, pcol):
    """
    Perform a distributed matrix multiplication of A (columns) and B (columns)
    using an allgather approach across rows and columns.

    Args:
        A (np.ndarray): A tile.
        B (np.ndarray): B tile.
        C (np.ndarray): C tile.
        A_comm (MPI.Comm): The MPI communicator for A matrix.
        B_comm (MPI.Comm): The MPI communicator for B matrix.
        m (int): Number of rows in A.
        k (int): Number of columns in A and rows in B.
        n (int): Number of columns in B.
        prow (int): Number of processor rows.
        pcol (int): Number of processor columns.


    Returns:
        None: The output matrix C is modified in place.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    A_rank = A_comm.Get_rank()
    B_rank = B_comm.Get_rank()

    # the actual max is one less this just makes the calculations easier
    max_B_subtile_row_plus_one = pcol  # size // pcol
    B_subtile_row = rank // prow  # max_B_subtile_row_plus_one
    C_subtile_column = B_comm.Get_rank()

    A_next = np.empty((m // prow, k // pcol))
    B_next = np.empty((k // pcol, n // size))

    for i in range(pcol):

        if i != pcol - 1:
            A_send_rank = (A_rank - 1) % A_comm.Get_size()
            A_receive_rank = (A_rank + 1) % A_comm.Get_size()
            A_send_request = A_comm.Isend(buf=(A, MPI_DTYPE), dest=A_send_rank)
            A_receive_request = A_comm.Irecv(
                buf=(A_next, MPI_DTYPE), source=A_receive_rank
            )

        for j in range(prow):

            A_curr = A
            # does this need to get copied to send probably not
            B_curr = get_subtile2(B, pcol, 1, B_subtile_row, 0)

            if j != prow - 1:
                B_send_rank = (B_rank - 1) % B_comm.Get_size()
                B_receive_rank = (B_rank + 1) % B_comm.Get_size()
                B_send_request = B_comm.Isend(buf=(B_curr, MPI_DTYPE), dest=B_send_rank)
                B_receive_request = B_comm.Irecv(
                    buf=(B_next, MPI_DTYPE), source=B_receive_rank
                )

            C_curr = get_subtile2(C, 1, prow, 0, C_subtile_column)

            debug_string = (
                f"(B_subtile_row, C_subtile_col)=({B_subtile_row, C_subtile_column})\n"
                f"(i, j, pcol)=({i},{j},{pcol})\n"
                f"A_CURR:\n{A}\n"
                f"B_row: {B_subtile_row}: B_CURR:\n{B_curr}\n"
            )
            # rank_print(debug_string)

            C_curr += np.matmul(A_curr, B_curr)

            set_subtile2(C, C_curr, 1, prow, 0, C_subtile_column)

            if j != prow - 1:
                MPI.Request.Waitall([B_send_request, B_receive_request])
                set_subtile2(B, B_next, pcol, 1, B_subtile_row, 0)

            C_subtile_column = (C_subtile_column + 1) % prow

        if i != pcol - 1:
            MPI.Request.Waitall([A_send_request, A_receive_request])
            # THIS COPY WAS THE ISSUE WHAT
            A = A_next.copy()

        B_subtile_row = (B_subtile_row + 1) % max_B_subtile_row_plus_one

    # parallel_print(C)


def AG_A_ROW_B_ROW(A, B, C, A_comm, B_comm, m, k, n, prow, pcol):
    """
    Perform a distributed matrix multiplication of A (rows) and B (rows)
    using an allgather approach across rows.

    Args:
        A (np.ndarray): A tile.
        B (np.ndarray): B tile.
        C (np.ndarray): C tile.
        A_comm (MPI.Comm): The MPI communicator for A matrix.
        B_comm (MPI.Comm): The MPI communicator for B matrix.
        m (int): Number of rows in A.
        k (int): Number of columns in A and rows in B.
        n (int): Number of columns in B.
        prow (int): Number of processor rows.
        pcol (int): Number of processor columns.


    Returns:
        None: The output matrix C is modified in place.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    A_rank = A_comm.Get_rank()
    B_rank = B_comm.Get_rank()

    A_next = np.empty((m // size, k // prow))
    B_next = np.empty((k // prow, n // pcol))

    A_subtile_index = 0  # fix
    C_subtile_index = 0  # fix

    B_curr = B

    for i in range(prow):

        if i != prow - 1:
            B_send_request = B_comm.Isend(
                [B_curr, MPI_DTYPE], (B_rank - 1) % B_comm.Get_size()
            )
            B_receive_request = B_comm.Irecv(
                [B_next, MPI_DTYPE], (B_rank + 1) % B_comm.Get_size()
            )

        for j in range(pcol):

            A_curr = get_subtile2(A, 1, prow, 0, A_subtile_index)

            C_curr = get_subtile2(C, pcol, 1, C_subtile_index, 0)

            C_curr += np.matmul(A_curr, B_curr)

            set_subtile2(C, C_curr, pcol, 1, C_subtile_index, 0)

            C_subtile_index = (C_subtile_index + 1) % pcol

        if i != prow - 1:
            MPI.Request.Waitall([B_send_request, B_receive_request])
            B_curr = B_next

        A_subtile_index = (A_subtile_index + 1) % prow
