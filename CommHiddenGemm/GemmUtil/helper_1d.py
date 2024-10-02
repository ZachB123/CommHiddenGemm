import numpy as np


from GemmUtil.constants import MATRIX_DTYPE

from GemmUtil.helper_general import matrices_equal, generate_matrix


def split_matrix(matrix, axis, rank, size):
    """Split the matrix along a specified axis for a given rank.

    Args:
        matrix (ndarray): The input matrix to split.
        axis (str): The axis to split along ('r' for rows, 'c' for columns).
        rank (int): The rank or index of the current process.
        size (int): The total number of parts to split the matrix into.

    Returns:
        ndarray: The submatrix for the given rank.

    Raises:
        ValueError: If an invalid axis is provided.
    """
    if axis == "r":
        dimension_length = matrix.shape[0]
        return matrix[
            rank * (dimension_length // size) : (rank + 1) * (dimension_length // size),
            :,
        ].copy()
    elif axis == "c":
        dimension_length = matrix.shape[1]
        return matrix[
            :,
            rank * (dimension_length // size) : (rank + 1) * (dimension_length // size),
        ].copy()
    raise ValueError("Invalid axis")


def generate_local_matrix(m, n, axis, size, zeros=False):
    """Generate a local submatrix of specified dimensions, either filled with zeros or random integers.

    Args:
        m (int): Number of rows of the global matrix.
        n (int): Number of columns of the global matrix.
        axis (str): The axis to split along ('r' for rows, 'c' for columns).
        size (int): The total number of splits along the axis.
        zeros (bool): Whether to fill the submatrix with zeros.

    Returns:
        ndarray: The generated local submatrix.

    Raises:
        ValueError: If an invalid axis is provided.
    """
    if axis == "r":
        return (
            np.zeros((m // size, n), dtype=MATRIX_DTYPE)
            if zeros
            else generate_matrix(m // size, n, -10, 10)
        )
    elif axis == "c":
        return (
            np.zeros((m, n // size), dtype=MATRIX_DTYPE)
            if zeros
            else generate_matrix(m, n // size, -10, 10)
        )

    raise ValueError("Invalid axis")


def dump_unequal_matrices(
    file_name, MATRIX_A, MATRIX_B, MATRIX_C, expected, actual, other_info=""
):
    """
    Logs matrices and comparison results to a file when expected and actual matrices differ.

    Args:
        file_name (str): Name of the file to append log information.
        MATRIX_A (ndarray): Input matrix A.
        MATRIX_B (ndarray): Input matrix B.
        MATRIX_C (ndarray): Result matrix C.
        expected (ndarray): The expected result matrix.
        actual (ndarray): The actual result matrix computed.
        other_info (str, optional): Additional information to log.
    """
    # we need to show like the entire true false grid and see where they like differ provide row col
    current_print_options = np.get_printoptions()
    np.set_printoptions(
        threshold=np.inf
    )  # (max(MATRIX_A.shape[0],MATRIX_A.shape[1],MATRIX_B.shape[1]) + 1000))

    with open(file_name, "a") as file:
        file.write("FAILURE OF COMPUTATION\n")
        file.write(f"{other_info}\n")
        file.write(f"Matrices Equal: {matrices_equal(expected, actual)}\n")
        file.write(f"NP IS CLOSE: {np.isclose(expected, actual).all()}\n\n")
        file.write(f"{np.isclose(expected, actual)}\n\n")

        file.write(
            f"Matrix A:\n{np.array2string(MATRIX_A, separator=',', formatter={'int': lambda x: str(x)})}\n\n"
        )
        file.write(
            f"Matrix B:\n{np.array2string(MATRIX_B, separator=',', formatter={'int': lambda x: str(x)})}\n\n"
        )
        file.write(
            f"Matrix C:\n{np.array2string(MATRIX_C, separator=',', formatter={'int': lambda x: str(x)})}\n\n"
        )
        file.write(
            f"expected:\n{np.array2string(expected, separator=',', formatter={'int': lambda x: str(x)})}\n\n"
        )
        file.write(
            f"actual:\n{np.array2string(actual, separator=',', formatter={'int': lambda x: str(x)})}\n\n\n"
        )

    np.set_printoptions(**current_print_options)
