import numpy as np
from mpi4py import MPI

from CommHiddenGemm.Util.util import (
    MATRIX_DTYPE,
    matrices_equal,
    get_step_indices,
    processor_rank_from_IJ,
)

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


def AG_A_COL_X_AG_B_ROW(A, B, C, m, k, n, prow, pcol, I, J):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    ktile = k // prow

    AIJ = A[
        I * (m // prow) : (I + 1) * (m // prow),
        get_step_indices(J * (ktile // pcol), MATRIX_A.shape[1], (k // prow) // 2),
    ].copy()
    BIJ = B[
        I * (k // prow) : (I + 1) * (k // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()
    CIJ = C[
        I * (m // prow) : (I + 1) * (m // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()

    # just get first square
    # column_block = rank // 2
    # Acurrent = A[]
    # C = C + np.matmul(A[:, 0: ktile // 2], B[0 : ktile // 2, :])
    # if rank == 0:
    #     print(C)
    ktile = k // prow
    # should these be copies?
    Acurrent = A[
        I * (m // prow) : (I + 1) * (m // prow),
        I * (k // prow)
        + J * (ktile // pcol) : I * (k // prow)
        + (J + 1) * (ktile // pcol),
    ].copy()
    Bcurrent = B[
        I * (k // prow)
        + J * (ktile // pcol) : I * (k // prow)
        + (J + 1) * (ktile // pcol),
        J * (n // pcol) : (J + 1) * (n // pcol),
    ].copy()
    # if rank == 1:
    #     print(Acurrent)
    for Krow in range(prow):
        Trow = (I - Krow) % prow
        for Kcol in range(pcol):
            Tcol = (J - Kcol) % pcol
            # if rank == 0:
            # print(f"Rank: {rank}, Trow: {Trow}, Tcol: {Tcol}")
            Acurrent = A[
                I * (m // prow) : (I + 1) * (m // prow),
                Trow * (k // prow)
                + Tcol * (ktile // pcol) : Trow * (k // prow)
                + (Tcol + 1) * (ktile // pcol),
            ].copy()
            Anext = np.empty(Acurrent.shape)
            A_send_request = comm.Isend(
                Acurrent, processor_rank_from_IJ(I, J - 1, prow, pcol)
            )
            A_receive_request = comm.Irecv(
                Anext, processor_rank_from_IJ(I, J + 1, prow, pcol)
            )

            Bcurrent = B[
                Trow * (k // prow)
                + Tcol * (ktile // pcol) : Trow * (k // prow)
                + (Tcol + 1) * (ktile // pcol),
                I * (m // prow) : (I + 1) * (m // prow),
            ].copy()
            Bnext = np.empty(Bcurrent.shape)
            B_send_request = comm.Isend(
                Bcurrent, processor_rank_from_IJ(I - 1, J, prow, pcol)
            )
            B_receive_request = comm.Irecv(
                Bnext, processor_rank_from_IJ(I + 1, J, prow, pcol)
            )

            if rank == 0:
                print(f"Acurrent:\n{Acurrent}\nBCurrent:\n{Bcurrent}\n")

            CIJ = CIJ + np.matmul(Acurrent, Bcurrent)

            MPI.Request.waitall(
                [A_send_request, A_receive_request, B_send_request, B_receive_request]
            )
            Acurrent = Anext
            Bcurrent = Bnext

    if rank == 0:
        print(CIJ)


def main():
    # processor distribution
    prow = 3
    pcol = 2
    # matrix dimensions
    m = 3
    k = 6
    n = 4

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

    # I and J start at 1 I think due to pseudocode
    I = rank // pcol
    J = rank % pcol

    # print(f"Rank: {rank}, I: {I}, J: {J}")

    AIJ = MATRIX_A[
        I * (m // prow) : (I + 1) * (m // prow),
        get_step_indices(J * (ktile // pcol), MATRIX_A.shape[1], (k // prow) // 2),
    ].copy()
    BIJ = MATRIX_B[
        I * (k // prow) : (I + 1) * (k // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()
    CIJ = MATRIX_C[
        I * (m // prow) : (I + 1) * (m // prow), J * (n // pcol) : (J + 1) * (n // pcol)
    ].copy()

    AG_A_COL_X_AG_B_ROW(MATRIX_A, MATRIX_B, MATRIX_C, m, k, n, prow, pcol, I, J)

    if rank == 0:
        print(f"Expected:\n{np.matmul(MATRIX_A,MATRIX_B) + MATRIX_C}")

    # print(f"Rank: {rank}\n{CIJ}")

    # comm.Barrier()

    # if rank == 1:
    # print(f"YUH\n{MATRIX_A[I * (m // prow) : (I + 1) * (m // prow), get_step_indices(J * (ktile // pcol), MATRIX_A.shape[1], (k // prow) // 2)]}")


def generate_matrix(row, col, min, max):
    # [min, max)
    return np.random.randint(min, max, size=(row, col)).astype(np.float64, copy=False)


if __name__ == "__main__":
    main()
    #  print(get_step_indices(20,3))
