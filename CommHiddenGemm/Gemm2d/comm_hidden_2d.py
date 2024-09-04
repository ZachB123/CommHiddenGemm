import numpy as np
from mpi4py import MPI

from CommHiddenGemm.Util.util import (
    MATRIX_DTYPE,
    matrices_equal,
    get_step_indices,
    processor_rank_from_IJ,
)


# Function to print with rank and color
def parallel_print(message):
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    def get_color_code(rank, num_colors):
        return f"\033[38;5;{rank % num_colors}m"
    
    color_code = get_color_code(rank, size)

    print(f"{color_code}[{rank}/{size - 1}]\n{message}\033[0m")

def proc_2d_indices_to_proc(I, J, prow, pcol):
    return I * pcol + J

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

    starting_A_index = I
    B_half = J # if 0 we use top half of B if 1 we use bottom half

    for i in range(prow):
        for j in range(pcol):
            # TODO there is an unecessary communication
            A_curr = A[:, (starting_A_index * A.shape[1]) // prow : ((starting_A_index + 1) * A.shape[1]) // prow]
            B_curr = B[B_half * (B.shape[0] // 2) : (B_half + 1) * (B.shape[0] // 2), :]

            A_send_request = comm.Isend(A_curr, proc_2d_indices_to_proc(I, (J + (-1)**(J % pcol)) % pcol, prow, pcol))
            A_next = np.empty((A.shape[0], (((starting_A_index + 1) * A.shape[1]) // prow) - ((starting_A_index * A.shape[1]) // prow)))
            A_receive_request = comm.Irecv(A_next, proc_2d_indices_to_proc(I, (J + (-1)**((J % pcol) + 1)) % pcol, prow, pcol))

            C = C + np.matmul(A_curr, B_curr)

            B_half = 1 - B_half
            
            MPI.Request.Waitall([A_send_request, A_receive_request])
            A[:, (starting_A_index * A.shape[1]) // prow : ((starting_A_index + 1) * A.shape[1]) // prow] = A_next

        starting_A_index = (starting_A_index + 1) % prow


        B_send_request = comm.Isend(B, proc_2d_indices_to_proc((I - 1) % prow, J, prow, pcol))
        B_next = np.empty(B.shape)
        B_receive_request = comm.Irecv(
            B_next, proc_2d_indices_to_proc((I + 1) % prow, J, prow, pcol)
        )
        MPI.Request.Waitall([B_send_request, B_receive_request])
        B = B_next

    parallel_print(C)


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

    # the local row and column processor indices
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

    AG_A_COL_X_AG_B_ROW(AIJ, BIJ, CIJ, m, k, n, prow, pcol, I, J)

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
    # parallel_print("test")
    main()
    #  print(get_step_indices(20,3))
