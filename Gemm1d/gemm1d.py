import numpy as np
from mpi4py import MPI
import logging

from util import MATRIX_DTYPE

def allgather_A_col(A_I, B_I, C_I):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # A m x k
    # B k x n
    # C m x n
    m = A_I.shape[0]
    k = B_I.shape[0]
    n = C_I.shape[1] * size

    if rank == 0:
        print(f"calculated ({m},{k},{n})")

    prev_rank = (rank + size - 1) % size
    next_rank = (rank + 1) % size

    for cycle in range(size):

        # buffer has n rows and k/size columns
        buffer = np.empty((m, k // size), dtype=MATRIX_DTYPE) 
        send_request = comm.Isend(np.ascontiguousarray(A_I), next_rank, 0) # no Isendrecv?
        receive_request = comm.Irecv(buffer, prev_rank, MPI.ANY_TAG)

        # old implementation
        # b_inner_start_index = ((k // size) * (rank - cycle)) % k
        # for i in range(m):
        #     for j in range(n // size):
        #         a_inner = 0
        #         for b_index in range(b_inner_start_index, b_inner_start_index + (k // size)):
        #             C_I[i,j] = C_I[i,j] + A_I[i,a_inner] * B_I[b_index,j]
        #             a_inner += 1
        
        shared_k_index = ((k // size) * (rank - cycle)) % k
        relevant_b_part = B_I[shared_k_index : shared_k_index + (k // size), :]
        C_I = C_I + np.matmul(A_I, relevant_b_part)

        MPI.Request.waitall([send_request, receive_request])
        A_I = buffer


    # this provides an array for each like value
    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        return np.concatenate(gathered_result, axis=1)
    return None

def allgather_A_row(A_I, B_I, C_I):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # A m x k
    # B k x n
    # C m x n
    m = C_I.shape[0]
    k = B_I.shape[0]
    n = C_I.shape[1] * size

    if rank == 0:
        print(f"calculated ({m},{k},{n})")

    prev_rank = (rank + size - 1) % size
    next_rank = (rank + 1) % size

    for cycle in range(size):

        # buffer has m / sizes rows and k columns
        buffer = np.empty((m // size, k), dtype=MATRIX_DTYPE) 
        send_request = comm.Isend(np.ascontiguousarray(A_I), next_rank, 0) # no Isendrecv?
        receive_request = comm.Irecv(buffer, prev_rank, MPI.ANY_TAG)

        local_matrix = np.matmul(A_I, B_I)
        shared_m_index = ((m // size) * (rank - cycle)) % m
        C_I[shared_m_index : shared_m_index + (m // size),:] += local_matrix

        MPI.Request.waitall([send_request, receive_request])
        A_I = buffer


    # this provides an array for each like value
    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        return np.concatenate(gathered_result, axis=1)
    return None
    
def allgather_B_col(A_I, B_I, C_I):
    pass

def allgather_B_row(A_I, B_I, C_I):
    pass

def reducescatter_C_col(A_I, B_I, C_I):
    pass

def reducescatter_C_row(A_I, B_I, C_I):
    pass




