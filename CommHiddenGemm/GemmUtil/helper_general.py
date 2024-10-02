import numpy as np
from mpi4py import MPI
from datetime import datetime

from GemmUtil.constants import MATRIX_DTYPE


def set_numpy_seed(seed):
    """
    Set the seed for NumPy's random number generator.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    np.random.seed(seed)


def parallel_print(message, flush=False):
    """
    Print a message with the MPI rank and total size, color-coded by rank.

    Args:
        message (str): The message to print.
        flush (bool, optional): Whether to flush the output immediately (default is False).
    """
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    def get_color_code(rank, num_colors):
        return f"\033[38;5;{rank % num_colors}m"

    color_code = get_color_code(rank, size)

    print(f"{color_code}[{rank}/{size - 1}]\n{message}\033[0m", flush=flush)


def generate_matrix(row, col, min, max):
    """
    Generate a matrix of specified shape with random integers.

    Args:
        row (int): Number of rows in the matrix.
        col (int): Number of columns in the matrix.
        min (int): Minimum integer (inclusive).
        max (int): Maximum integer (exclusive).

    Returns:
        np.ndarray: A matrix of shape (row, col) with random integers in [min, max).
    """
    return np.random.randint(min, max, size=(row, col)).astype(MATRIX_DTYPE, copy=False)


def matrix_multiply(a, b, c):
    """
    Perform matrix multiplication of a and b, then add c.

    Args:
        a (np.ndarray): The first matrix.
        b (np.ndarray): The second matrix.
        c (np.ndarray): The matrix to be added to the product of a and b.

    Returns:
        np.ndarray: The result of a * b + c.
    """
    return np.matmul(a, b) + c


def matrices_equal(A, B):
    """
    Check if two matrices are approximately equal within a tolerance.

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Returns:
        bool: True if the matrices are approximately equal, False otherwise.
    """
    # use machine epsilon in future pass in delta - depends on floating point number format
    # I have the tolerance really high rn
    return np.allclose(A, B, atol=0.1)


def calculate_throughput(time, m, n, k):
    """
    Calculate the throughput in GFLOPS.

    Args:
        time (float): The time taken for the computation in seconds.
        m (int): The number of rows in matrix A.
        n (int): The number of columns in matrix B.
        k (int): The common dimension between matrices A and B.

    Returns:
        float: The throughput in giga floating point operations per second (GFLOPS).
    """
    return (2 * m * n * k / time) * 1e-9


def get_date_string():
    """
    Get the current date as a formatted string.

    Returns:
        str: The current date formatted as mm-dd-yyyy.
    """
    current_date = datetime.now()
    formatted_date = current_date.strftime("%m-%d-%Y")
    return formatted_date