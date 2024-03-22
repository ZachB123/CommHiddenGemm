import numpy as np

# seeding is currently required so all processes pull the same matrix randomly
np.random.seed(42)

# if change to 32 the margin for error will be not as close
MATRIX_DTYPE = np.float64
BENCHMARK_FILE = "benchmark.csv"

MINI_MATRIX_A = np.array([
    [-5, -10,  5,  -3,   6, -10,   5, -8],
    [-4,  -2, -4, -10,   6,   0,  -7,  2],
    [-10,  1,  3,  -4,   2,  -7,  -7, -6],
    [-9,   3,  8, -10,   5,  -7,   5,  7],
    [-9,  -4,  2,   1,   6,   6,  -1, -1],
    [8,    4,  0,   3,   7,  -1,   5,  5],
    [-2,   8,  0,  -6, -10,   9,  -2, -5],
    [-4, -10, -1,  -6,   5,  -2,   3, -2],
    [-6,   6,  6,  -4,   2,   5,   0,  6],
    [-10, -9, -9,  -6,   8,  -3,   1, -8],
    [6,    8,  6,  -5,  -9,  -1,  -4,  1],
    [-7,  -6,  4,   1,  -5,   8, -10, -8],
    [-5,   9, -7,  -8,  -1,   1,   0, -8],
    [-8,   9, -2,   1,  -1,   5,   6, -5],
    [-2,   4, -5,   3,  -3,  -8,   2,  2],
    [-7,   4,  2, -10,  -1,   3,  -5,  5],
], dtype=MATRIX_DTYPE)

MINI_MATRIX_B = np.array([
    [0, 8, 8, 5],
    [-7, 4, 6, 3],
    [-7, -5, 5, -7],
    [-3, 5, 8, -3],
    [8, 9, -5, 9],
    [5, 2, 5, 9],
    [-2, 1, 5, 7],
    [6, -6, 2, -2]
], dtype=MATRIX_DTYPE)

MINI_MATRIX_C = np.zeros((16, 4), dtype=MATRIX_DTYPE)  # Initialize MINI_MATRIX_C with zeros

EXPECTED_MINI_MATRIX_C = np.array([
    [-16, -33, -170, -66],
    [146, -35, -205, 33],
    [-57, -78, -183, -138],
    [-10, -156, -115, -59],
    [85, -22, -85, 29],
    [34, 131, 107, 122],
    [-99, -58, 59, 19],
    [107, -41, -169, 27],
    [5, -82, 13, 9],
    [143, 14, -293, 72],
    [-146, -68, 108, -93],
    [-17, -86, -65, -111],
    [-41, 32, -91, 91],
    [-77, 24, 38, 86],
    [-58, -13, -4, -61],
    [35, -138, -97, -34]
], dtype=MATRIX_DTYPE)


def generate_matrix(row, col, min, max):
    # [min, max)
    return np.random.randint(min, max, size=(row,col)).astype(MATRIX_DTYPE, copy=False)

def matrix_multiply(a,b,c):
    return np.matmul(a,b) + c 

def matrices_equal(A, B):
    # use machine epsilon in future pass in delta - depends on floating point number format
    # I have the tolerance really high rn
    return np.allclose(A, B)


def calculate_throughput(time, m, n, k):
    # Giga flops per second
    return (2 * m * n * k / time) * 1e-9

def split_matrix(matrix, axis, rank, size):
    if axis == "r":
        dimension_length = matrix.shape[0]
        return matrix[rank * (dimension_length // size) : (rank + 1) * (dimension_length // size), :]
    elif axis == "c":
        dimension_length = matrix.shape[1]
        return matrix[:, rank * (dimension_length // size) : (rank + 1) * (dimension_length // size)]
    raise ValueError("Invalid Axis")