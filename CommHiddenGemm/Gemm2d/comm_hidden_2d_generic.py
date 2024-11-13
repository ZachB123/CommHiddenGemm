from enum import Enum, auto
import numpy as np
from mpi4py import MPI

from GemmUtil.constants import MPI_DTYPE, DEBUG_RANK

from GemmUtil.helper_general import generate_matrix, parallel_print, rank_print

from GemmUtil.helper_2d import get_subtile, set_subtile, get_subtile2, set_subtile2


class Gemm1d(Enum):
    A_COL = auto()
    A_ROW = auto()
    B_COL = auto()
    B_ROW = auto()
    C_COL = auto()
    C_ROW = auto()

class Gemm2d(Enum):
    A_COL_A_ROW = auto()
    A_COL_B_COL = auto()
    A_COL_B_ROW = auto()
    A_COL_C_COL = auto()
    A_COL_C_ROW = auto()
    A_ROW_B_COL = auto()
    A_ROW_B_ROW = auto()
    A_ROW_C_COL = auto()
    A_ROW_C_ROW = auto()
    B_COL_B_ROW = auto()
    B_COL_C_COL = auto()
    B_COL_C_ROW = auto()
    B_ROW_C_COL = auto()
    B_ROW_C_ROW = auto()
    C_COL_C_ROW = auto()


class CurrentGemm():
    def __init__(self, m, k, n, p, prow, pcol):
        self.m = m
        self.k = k
        self.n = n
        self.p = p
        self.prow = prow
        self.pcol = pcol


class MatrixDimension(Enum):
    M = auto()
    K = auto()
    N = auto()


class Process(Enum):
    ONE = auto()
    P = auto()
    PROW = auto()
    PCOL = auto


class MatrixDimensionSplit:
    def __init__(self, matrix_dimension: MatrixDimension, proc_data: Process):
        self.matrix_dimension = matrix_dimension
        self.proc_data = proc_data

    def evaluate_split(self, config: CurrentGemm):
        if self.matrix_dimension == MatrixDimension.M:
            dim_value = config.m
        elif self.matrix_dimension == MatrixDimension.K:
            dim_value = config.k
        elif self.matrix_dimension == MatrixDimension.N:
            dim_value = config.n
        else:
            raise ValueError("Invalid matrix dimension")

        if self.proc_data == Process.ONE:
            proc_value = 1  # No split, so divide by 1
        elif self.proc_data == Process.P:
            proc_value = config.p
        elif self.proc_data == Process.PROW:
            proc_value = config.prow
        elif self.proc_data == Process.PCOL:
            proc_value = config.pcol
        else:
            raise ValueError("Invalid processor configuration")

        return dim_value // proc_value

    def __repr__(self):
        return f"({self.matrix_dimension.name} // {self.proc_data.name})"


class MatrixTile():
    def __init__(self, dim1_split: MatrixDimensionSplit, dim2_split: MatrixDimensionSplit):
        self.dim1_split = dim1_split
        self.dim2_split = dim2_split


class Gemm1dData():
    def __init__(self, A_subtile: MatrixTile, B_subtile: MatrixTile, C_subtile: MatrixTile):
        self.A_subtile = A_subtile
        self.B_subtile = B_subtile
        self.C_subtile = C_subtile


class Gemm2dData():
    def __init__(self, first_algo: Gemm1dData, second_algo: Gemm1dData):
        self.first_algo = first_algo
        self.second_algo = second_algo
        # combine these 2 to make the tiles and subtiles
        self.A_tile, self.A_subtile = self._combine_tiles(first_algo.A_subtile, second_algo.A_subtile)
        self.B_tile, self.B_subtile = self._combine_tiles(first_algo.B_subtile, second_algo.B_subtile)
        self.C_tile, self.C_subtile = self._combine_tiles(first_algo.C_subtile, second_algo.C_subtile)

    
    def _combine_tiles(tile1: MatrixTile, tile2: MatrixTile):
        pass

# move to constants file later
GEMM_1D = {
    Gemm1d.A_COL: Gemm1dData(
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.P          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.P          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.P          
            )
        )
    ),
    Gemm1d.A_ROW: Gemm1dData(
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.ONE          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.P          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.P          
            )
        )
    ),
    Gemm1d.B_COL: Gemm1dData(
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.ONE          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.P          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.ONE          
            )
        )
    ),
    Gemm1d.B_ROW: Gemm1dData(
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.ONE          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.ONE          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.ONE          
            )
        )
    ),
    Gemm1d.C_COL: Gemm1dData(
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.P          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.ONE          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.P          
            )
        )
    ),
    Gemm1d.C_ROW: Gemm1dData(
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.ONE
            ),
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.P          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.K,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.ONE          
            )
        ),
        MatrixTile(
            MatrixDimensionSplit(
                MatrixDimension.M,
                Process.P
            ),
            MatrixDimensionSplit(
                MatrixDimension.N,
                Process.ONE          
            )
        )
    ),
}

GEMM_2d = {
    Gemm2d.A_COL_A_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.A_ROW]
    ),
    Gemm2d.A_COL_B_COL: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.B_COL]
    ),
    Gemm2d.A_COL_B_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.B_ROW]
    ),
    Gemm2d.A_COL_C_COL: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.C_COL]
    ),
    Gemm2d.A_COL_C_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.C_ROW]
    ),
    Gemm2d.A_ROW_B_COL: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.B_COL]
    ),
    Gemm2d.A_ROW_B_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.B_ROW]
    ),
    Gemm2d.A_ROW_C_COL: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.C_COL]
    ),
    Gemm2d.A_ROW_C_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.A_COL],
        GEMM_1D[Gemm1d.C_ROW]
    ),
    Gemm2d.B_COL_B_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.B_COL],
        GEMM_1D[Gemm1d.B_ROW]
    ),
    Gemm2d.B_COL_C_COL: Gemm2dData(
        GEMM_1D[Gemm1d.B_COL],
        GEMM_1D[Gemm1d.C_COL]
    ),
    Gemm2d.B_COL_C_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.B_COL],
        GEMM_1D[Gemm1d.C_ROW]
    ),
    Gemm2d.B_ROW_C_COL: Gemm2dData(
        GEMM_1D[Gemm1d.B_ROW],
        GEMM_1D[Gemm1d.C_COL]
    ),
    Gemm2d.B_ROW_C_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.B_ROW],
        GEMM_1D[Gemm1d.C_ROW]
    ),
    Gemm2d.C_COL_C_ROW: Gemm2dData(
        GEMM_1D[Gemm1d.C_COL],
        GEMM_1D[Gemm1d.C_ROW]
    ),
}


def gemm_2d_driver(data: CurrentGemm, algorithm: Gemm2dData):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert data.m % ? == 0
    assert data.k % ? == 0
    assert data.n % ? == 0

    outer_comm = comm.Split(?, rank)
    inner_comm = comm.Split(?, rank)

    A = generate_matrix(data.m, data.k, -10, 10)
    B = generate_matrix(data.k, data.n, -10, 10)
    C = np.zeros((data.m, data.n))

    rank_print(f"A:\n{A}")
    rank_print(f"B\n{B}")

    A_I = get_subtile2(A, algorithm.A_subtile.dim1_split, algorithm.A_subtile.dim2_split, ?, ?).copy()
    B_I = get_subtile2(B, algorithm.B_subtile.dim1_split, algorithm.B_subtile.dim2_split, ?, ?).copy()
    C_I = get_subtile2(C, algorithm.C_subtile.dim1_split, algorithm.C_subtile.dim2_split, ?, ?).copy()

    # rank_print(f"A_I:\n{A_I}\nB_I:\n{B_I}")

    gemm2d(A_I, B_I, C_I, A_comm, C_comm, data, algorithm)


def gemm_2d(A: np.ndarray, B: np.ndarray, C: np.ndarray, outer_comm: MPI.Comm, inner_comm: MPI.Comm, data: CurrentGemm, tiling: Gemm2dData):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    outer_rank = outer_comm.Get_rank()
    inner_rank = inner_comm.Get_rank()

    outer_next = np.empty(?)
    inner_next = np.empty(?)

    outer_subtile_index = ?
    inner_subtile_index = ?

    for i in range(?):

        outer_curr = get_subtile2(?)

        if i != ? - 1:
            outer_send_rank = (outer_rank - 1) % outer_comm.Get_size()
            outer_receive_rank = (outer_rank + 1) % outer_comm.Get_size()
            outer_send_request = outer_comm.Isend(buf=(outer_curr, MPI_DTYPE), dest=outer_send_rank)
            outer_receive_request = outer_comm.Irecv(
                buf=(outer_next, MPI_DTYPE), source=outer_receive_rank
            )

        for j in range(?):

            inner_curr = get_subtile2(?)

            if j != ? - 1:
                inner_send_rank = (inner_rank - 1) % inner_comm.Get_size()
                inner_receive_rank = (inner_rank + 1) % inner_comm.Get_size()
                inner_send_request = inner_comm.Isend(
                    buf=(inner_curr, MPI_DTYPE), dest=inner_send_rank
                )
                inner_receive_request = inner_comm.Irecv(
                    buf=(inner_next, MPI_DTYPE), source=inner_receive_rank
                )

            other_curr = get_subtile2(?)

            # The question marks are outer_curr, inner_curr, other_curr
            ? += np.matmul(?, ?)    

            # setting C
            set_subtile2(?)

            inner_index = (inner_index + 1) % ?

            if j != pcol - 1:
                MPI.Request.Waitall([inner_send_request, inner_receive_request])
                set_subtile2(?)

        if i != prow - 1:
            MPI.Request.Waitall([outer_send_request, outer_receive_request])
            set_subtile2(?)

        outer_index = (outer_index + 1) % ?