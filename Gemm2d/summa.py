import numpy as np
from mpi4py import MPI

np.random.seed(420)
MATRIX_DTYPE = np.float64

MATRIX_A = np.array([[ -4.0,  9.0,  4.0,  0.0, -3.0, -4.0,  8.0,  0.0,  0.0],
                     [ -7.0, -3.0, -8.0, -9.0,  1.0, -5.0, -9.0, -10.0, 1.0],
                     [  1.0,  6.0, -1.0,  5.0,  4.0,  4.0,  8.0,  1.0,  9.0],
                     [ -8.0, -6.0,  8.0, -4.0, -2.0, -4.0,  7.0, -7.0,  3.0],
                     [  7.0, -2.0, -9.0,  9.0,  4.0, -4.0,  1.0, -3.0,  4.0],
                     [ -8.0,  3.0,  6.0, -7.0,  7.0, -3.0, -7.0, -9.0, -5.0],
                     [ -1.0, -7.0,  7.0,  1.0, -9.0, -1.0, -7.0,  3.0,  5.0],
                     [  4.0, -3.0,  3.0, -3.0,  5.0,  2.0,  7.0,  4.0,  2.0],
                     [ -2.0,  4.0,  2.0,-10.0, -4.0, -2.0,-10.0,  1.0, -3.0]], dtype=MATRIX_DTYPE)

MATRIX_B = np.array([[  7.0,  -2.0,  -4.0,   9.0,   4.0,   0.0,  -2.0,   4.0,  -4.0],
                     [ -7.0,  -6.0, -10.0,   3.0,  -5.0,  -5.0,   5.0,   2.0,   1.0],
                     [ -2.0,  -8.0,  -5.0,  -5.0,   3.0,   5.0,   5.0,  -2.0,  -1.0],
                     [  1.0,  -1.0,  -3.0,  -1.0, -10.0,  -6.0,   4.0,   7.0,   5.0],
                     [  8.0,  -8.0,  -5.0,   9.0,  -6.0,   5.0,  -7.0,   4.0,   4.0],
                     [ -9.0,   6.0,   5.0,   2.0,  -8.0,   3.0,  -7.0,  -1.0,   8.0],
                     [ -2.0,   1.0,   1.0,   5.0,   0.0,  -6.0,  -1.0,   5.0,   6.0],
                     [  0.0,  -4.0,  -6.0,  -2.0,  -5.0,  -2.0,   1.0,   8.0,   3.0],
                     [-10.0,   5.0,  -7.0,   3.0,   0.0,   3.0,  -2.0,   8.0,  -1.0]], dtype=MATRIX_DTYPE)

MATRIX_C = np.zeros((9,9), dtype=MATRIX_DTYPE)

def generate_matrix(row, col, min, max):
    # [min, max)
    return np.random.randint(min, max, size=(row,col)).astype(MATRIX_DTYPE, copy=False)


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    expected = MATRIX_C + np.matmul(MATRIX_A, MATRIX_B)

    num_rows = 3
    num_cols = 3

    fr_group = comm.group.Incl(range(0,num_rows))
    fc_group = comm.group.Incl(range(0,size,num_rows))

    fr_comm = comm.Create(fr_group)
    fc_comm = comm.Create(fc_group)

    if fr_comm != MPI.COMM_NULL:
        fr_rank = fr_comm.Get_rank()
        print(f"Rank is {rank}. fr rank is {fr_rank}")
    if fc_comm != MPI.COMM_NULL:
        fc_rank = fc_comm.Get_rank()
        print(f"Rank is {rank}. fc rank is {fc_rank}")



if __name__ == "__main__":
    main()