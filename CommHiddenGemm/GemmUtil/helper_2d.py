import numpy as np


def proc_2d_indices_to_proc(I, J, prow, pcol):
    """
    Convert a row and column processer indices (I, J) to the standard MPI rank
    based on the dimensions of the processor grid

    Parameters:
    -----------
    I : int
        The row index.
    J : int
        The column index.
    prow : int
        The number of rows in the process grid.
    pcol : int
        The number of columns in the process grid.

    Returns:
    --------
    int
        The corresponding 1D process index.
    """
    return I * pcol + J


def processor_rank_from_IJ(I, J, prow, pcol):
    """
    Calculate the processor rank based on the row and  column processor indices.

    Args:
        I (int): The row index.
        J (int): The column index.
        prow (int): The number of processor rows.
        pcol (int): The number of processor columns.

    Returns:
        int: The calculated processor rank.

    Note:
        The rank is computed based on the row and column indices modulo
        the number of processor rows and columns.
    """
    return (I % prow) * pcol + (J % pcol)


def pad_amount(a, b):
    """
    Calculate the minimum amount to add to `a` to make it divisible by `b`.

    Args:
        a (int): The value to be adjusted.
        b (int): The divisor.

    Returns:
        int: The amount to add to `a`.
    """
    remainder = a % b
    if remainder == 0:
        return 0
    else:
        return b - remainder


def pad_matrix_with_zeros(matrix, row_pad, col_pad):
    """
    Pad the input matrix with zeros.

    Args:
        matrix (np.ndarray): The matrix to be padded.
        row_pad (int): The number of rows to add.
        col_pad (int): The number of columns to add.

    Returns:
        np.ndarray: The padded matrix.
    """
    if row_pad > 0:
        matrix = np.vstack((matrix, np.zeros((row_pad, matrix.shape[1]))))

    if col_pad > 0:
        matrix = np.hstack((matrix, np.zeros((matrix.shape[0], col_pad))))

    return matrix


def remove_padding(matrix, row_pad, col_pad):
    """
    Remove padding from the input matrix.

    Args:
        matrix (np.ndarray): The padded matrix to modify.
        row_pad (int): The number of rows to remove.
        col_pad (int): The number of columns to remove.

    Returns:
        np.ndarray: The matrix after removing the specified padding.
    """
    if row_pad > 0:
        matrix = matrix[:-row_pad, :]
    if col_pad > 0:
        matrix = matrix[:, :-col_pad]
    return matrix


def get_subtile(tile, slice, n_slices, direction):
    """
    Extract a subtile from the given tile based on the specified slice and direction.

    Args:
        tile (np.ndarray): The input tile from which to extract a subtile.
        slice (int): The index of the slice to extract.
        n_slices (int): The total number of slices to split the tile into.
        direction (str): The direction for slicing ('r' for rows, 'c' for columns).

    Returns:
        np.ndarray: The extracted sub-tile.
    """
    # n_slices is like how many individual slices we will eventually make, slice is the index for n_slices
    # direction is a string for rows or columns and value will be r or c
    # if r we slice across the rows, so all columns preserved and vice versa
    tile = np.atleast_2d(tile)
    if direction == "r":
        rows = tile.shape[0]
        assert (
            rows % n_slices == 0
        ), f"Tiles rows {rows} are not divisible by slices {n_slices} requested"
        width = rows // n_slices
        return tile[width * slice : width * (slice + 1), :]
    else:
        cols = tile.shape[1]
        assert (
            cols % n_slices == 0
        ), f"Tiles columns {cols} are not divisible by slices {n_slices} requested"
        width = cols // n_slices
        return tile[:, width * slice : width * (slice + 1)]


def set_subtile(tile, slice, n_slices, direction, block):
    """
    Set a sub-tile in the given tile based on the specified slice and direction.

    Args:
        tile (np.ndarray): The input tile to modify.
        slice (int): The index of the slice to modify.
        n_slices (int): The total number of slices.
        direction (str): The direction for slicing ('r' for rows, 'c' for columns).
        block (np.ndarray): The block to set in the specified sub-tile.

    Returns:
        None: The input tile is modified in place.
    """
    tile = np.atleast_2d(tile)
    if direction == "r":
        rows = tile.shape[0]
        assert (
            rows % n_slices == 0
        ), f"Tiles rows {rows} are not divisible by slices {n_slices} requested"
        width = rows // n_slices
        tile[width * slice : width * (slice + 1), :] = block
    else:
        cols = tile.shape[1]
        assert (
            cols % n_slices == 0
        ), f"Tiles columns {cols} are not divisible by slices {n_slices} requested"
        width = cols // n_slices
        tile[:, width * slice : width * (slice + 1)] = block


def get_subtile2(tile, rows, columns, i, j):
    # fancier way to do subtiles
    # we split the tile into the rows and columns and can index it like a 2d array
    # if column is 1 then we just use j as an index

    subtile_rows = tile.shape[0] // rows
    subtile_columns = tile.shape[1] // columns

    start_row = i * subtile_rows
    end_row = start_row + subtile_rows

    start_col = j * subtile_columns
    end_col = start_col + subtile_columns

    return tile[start_row:end_row, start_col:end_col]


def set_subtile2(tile, subtile, rows, columns, i, j):
    subtile_rows = tile.shape[0] // rows
    subtile_columns = tile.shape[1] // columns

    start_row = i * subtile_rows
    end_row = start_row + subtile_rows

    start_col = j * subtile_columns
    end_col = start_col + subtile_columns

    # Set the subtile in the original tile
    tile[start_row:end_row, start_col:end_col] = subtile


def get_local_block(matrix, local_i, local_j, row_block_size, col_block_size):
    """Extract a local block from the matrix.

    Args:
        matrix (np.ndarray): The input matrix.
        local_i (int): The column index of the block to extract.
        local_j (int): The row index of the block to extract.
        row_block_size (int): The number of rows in the block.
        col_block_size (int): The number of columns in the block.

    Returns:
        np.ndarray: The extracted block of the matrix.
    """
    return matrix[
        local_j * col_block_size : (local_j + 1) * col_block_size,
        local_i * row_block_size : (local_i + 1) * row_block_size,
    ].copy()


def get_step_indices(start, k, partitions, width):
    """
    gets the indices to index an array for block cyclic tiling

    Args:
        start (int): The starting index for selection.
        k (int): The upper limit for indices.
        partitions (int): The number of partitions for splitting.
        width (int): The number of indices to select in each partition.

    Returns:
        list[int]: A list of selected indices.

    Raises:
        AssertionError: If k is not divisible by (partitions * width) or if
                        any of the input values are invalid.
    """
    assert (
        k % (partitions * width) == 0
    ), "The number of partitions and their width must split k"
    selected_indices = []
    for curr in range(start, k, partitions * width):
        selected_indices.extend(list(range(curr, curr + width)))
    return selected_indices
