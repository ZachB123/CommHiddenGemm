import numpy as np
from mpi4py import MPI
import logging

from util import MATRIX_DTYPE

# this file is copied from gemm1d.py but with the computation part removed

def allgather_A_col_no_compute(A_I, B_I, C_I):
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


    for cycle in range(size - 1):

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
        # shared_k_index = ((k // size) * (rank - cycle)) % k
        # relevant_b_part = B_I[shared_k_index : shared_k_index + (k // size), :]
        # C_I = C_I + np.matmul(A_I, relevant_b_part)


        MPI.Request.waitall([send_request, receive_request])
        A_I = buffer


    # I have no idea why 2 * size + 1 works here
    # shared_k_index = ((k // size) * (rank - 2 * size  + 1)) % k
    # relevant_b_part = B_I[shared_k_index : shared_k_index + (k // size), :]
    # C_I = C_I + np.matmul(A_I, relevant_b_part)

    # this provides an array for each like value
    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        return np.concatenate(gathered_result, axis=1)
    return None

def allgather_A_row_no_compute(A_I, B_I, C_I):
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

    

    for cycle in range(size - 1):

        # buffer has m / sizes rows and k columns
        buffer = np.empty((m // size, k), dtype=MATRIX_DTYPE) 
        send_request = comm.Isend(np.ascontiguousarray(A_I), next_rank, 0) # no Isendrecv?
        receive_request = comm.Irecv(buffer, prev_rank, MPI.ANY_TAG)

        # local_matrix = np.matmul(A_I, B_I)
        # shared_m_index = ((m // size) * (rank - cycle)) % m
        # C_I[shared_m_index : shared_m_index + (m // size),:] += local_matrix

        MPI.Request.waitall([send_request, receive_request])
        A_I = buffer

    # local_matrix = np.matmul(A_I, B_I)
    # shared_m_index = ((m // size) * (rank - 2 * size + 1)) % m
    # C_I[shared_m_index : shared_m_index + (m // size),:] += local_matrix

    # this provides an array for each like value
    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        return np.concatenate(gathered_result, axis=1)
    return None
    
def allgather_B_col_no_compute(A_I, B_I, C_I):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # A m x k
    # B k x n
    # C m x n
    m = A_I.shape[0] * size
    k = B_I.shape[0]
    n = C_I.shape[1]

    if rank == 0:
        print(f"calculated ({m},{k},{n})")

    prev_rank = (rank + size - 1) % size
    next_rank = (rank + 1) % size
    

    for cycle in range(size - 1):

        # buffer has k rows and n // size columns
        buffer = np.empty((k, n // size), dtype=MATRIX_DTYPE) 

        send_request = comm.Isend(np.ascontiguousarray(B_I), next_rank, 0) # no Isendrecv?
        receive_request = comm.Irecv(buffer, prev_rank, MPI.ANY_TAG)

        # local_matrix = np.matmul(A_I, B_I)
        # shared_n_index = ((n // size) * (rank - cycle)) % n
        # C_I[:, shared_n_index : shared_n_index + (n // size)] += local_matrix

        MPI.Request.waitall([send_request, receive_request])
        B_I = buffer


    # local_matrix = np.matmul(A_I, B_I)
    # shared_n_index = ((n // size) * (rank - 2 * size + 1)) % n
    # C_I[:, shared_n_index : shared_n_index + (n // size)] += local_matrix

    # this provides an array for each like value
    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        return np.concatenate(gathered_result, axis=0)
    return None

def allgather_B_row_no_compute(A_I, B_I, C_I):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # A m x k
    # B k x n
    # C m x n
    m = A_I.shape[0] * size
    k = A_I.shape[1]
    n = C_I.shape[1]

    if rank == 0:
        print(f"calculated ({m},{k},{n})")

    prev_rank = (rank + size - 1) % size
    next_rank = (rank + 1) % size

    for cycle in range(size - 1):

        # buffer has k / size rows and n columns
        buffer = np.empty((k // size, n), dtype=MATRIX_DTYPE)

        send_request = comm.Isend(np.ascontiguousarray(B_I), next_rank, 0) # no Isendrecv?
        receive_request = comm.Irecv(buffer, prev_rank, MPI.ANY_TAG)
        
        # shared_k_index = ((k // size) * (rank - cycle)) % k
        # relevant_a_part = A_I[:, shared_k_index : shared_k_index + (k // size)]
        # C_I = C_I + np.matmul(relevant_a_part, B_I)

        MPI.Request.waitall([send_request, receive_request])
        B_I = buffer

    # shared_k_index = ((k // size) * (rank - 2 * size + 1)) % k
    # relevant_a_part = A_I[:, shared_k_index : shared_k_index + (k // size)]
    # C_I = C_I + np.matmul(relevant_a_part, B_I)

    # this provides an array for each like value
    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        return np.concatenate(gathered_result, axis=0)
    return None

def reducescatter_C_col_no_compute(A_I, B_I, C_I):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # A m x k
    # B k x n
    # C m x n
    m = A_I.shape[0]
    k = B_I.shape[0] * size
    n = B_I.shape[1]

    if rank == 0:
        print(f"calculated ({m},{k},{n})")

    prev_rank = (rank + size - 1) % size
    next_rank = (rank + 1) % size

    # have a computation done so we are ready to send something
    # initial_shared_n_index = ((n // size) * (rank)) % n
    # initial_relevant_b_part = B_I[:, initial_shared_n_index : initial_shared_n_index + (n // size)]
    # C_I = C_I + np.matmul(A_I, initial_relevant_b_part)

    buffer = np.empty((m, n // size), dtype=MATRIX_DTYPE) 

    for cycle in range(1, size):

        # buffer has m rows and n / size columns
        send_request = comm.Isend(np.ascontiguousarray(C_I), next_rank, 0) # no Isendrecv?
        receive_request = comm.Irecv(buffer, prev_rank, MPI.ANY_TAG)
        
        # shared_n_index = ((n // size) * (rank - cycle)) % n
        # relevant_b_part = B_I[:, shared_n_index : shared_n_index + (n // size)]
        
        # Temp_C_I = np.matmul(A_I, relevant_b_part)

        MPI.Request.waitall([send_request, receive_request])
        # C_I = Temp_C_I + buffer


    # this provides an array for each like value
    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        # rotate the array so the correct values line up nicely
        gathered_result = gathered_result[-1:] + gathered_result[:-1]
        return np.concatenate(gathered_result, axis=1)
    return None

def reducescatter_C_row_no_compute(A_I, B_I, C_I):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # A m x k
    # B k x n
    # C m x n
    m = A_I.shape[0]
    k = B_I.shape[0] * size
    n = C_I.shape[1]

    if rank == 0:
        print(f"calculated ({m},{k},{n})")

    prev_rank = (rank + size - 1) % size
    next_rank = (rank + 1) % size

    # have a computation done so we are ready to send something
    # initial_shared_m_index = ((m // size) * (rank)) % m
    # initial_relevant_a_part = A_I[initial_shared_m_index : initial_shared_m_index + (m // size), :]
        
    # C_I = C_I + np.matmul(initial_relevant_a_part, B_I)

    buffer = np.empty((m // size, n), dtype=MATRIX_DTYPE) # try moving this outside the loop at somepoint 

    for cycle in range(1, size):

        # buffer has m / size rows and n columns
        send_request = comm.Isend(np.ascontiguousarray(C_I), next_rank, 0) # no Isendrecv?
        receive_request = comm.Irecv(buffer, prev_rank, MPI.ANY_TAG)
        
        # shared_m_index = ((m // size) * (rank - cycle)) % m
        # relevant_a_part = A_I[shared_m_index : shared_m_index + (m // size), :]
        
        # Temp_C_I = np.matmul(relevant_a_part, B_I)

        MPI.Request.waitall([send_request, receive_request])
        # C_I = Temp_C_I + buffer


    # this provides an array for each like value
    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        # rotate the array so the correct values line up nicely
        gathered_result = gathered_result[-1:] + gathered_result[:-1]
        return np.concatenate(gathered_result, axis=0)
    return None

def broadcast_based_no_compute(A_I, B_I, C_I):
    # all matrices split by rows
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # A m x k
    # B k x n
    # C m x n
    m = A_I.shape[0] * size
    k = A_I.shape[1]
    n = C_I.shape[1]

    if rank == 0:
        print(f"calculated ({m},{k},{n})")
        # print

    K = 0
    for K in range(size):
        # Btmp = np.empty((k // size, n), dtype=MATRIX_DTYPE)
        Btmp = comm.bcast(B_I, K)
        # this shouldnt go overlength of array since we run the loop size times
        # relevant_a_part = A_I[:, K * (k // size) : (K + 1) * (k // size)]

        # C_I = C_I + np.matmul(relevant_a_part, Btmp)

    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        return np.concatenate(gathered_result, axis=0)
    return None

# broadcast based with overlap
def broadcast_based_with_overlap_no_compute(A_I, B_I, C_I):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # A m x k
    # B k x n
    # C m x n
    m = A_I.shape[0] * size
    k = A_I.shape[1]
    n = C_I.shape[1]

    # first broadcast in the algorithm is blocking
    Btmp = None
    if rank == 0:
        Btmp = comm.bcast(B_I, root=0)
    else:
        Btmp = comm.bcast(None, root=0)

    Bnext = np.empty((k // size, n), dtype=MATRIX_DTYPE)

    K = 0
    for K in range(1, size):
        bcast_request = None
        if rank == K:
            bcast_request = comm.Ibcast(B_I, root=K)
            Bnext = B_I
        else:
            bcast_request = comm.Ibcast(Bnext, root=K)

        # relevant_a_part = A_I[:, (K - 1) * (k // size) : K * (k // size)]
        # C_I = C_I + np.matmul(relevant_a_part, Btmp)

        status = MPI.Status()
        bcast_request.Wait(status=status)
        comm.Barrier() # THE CODE WILL RANDOMLY FAIL LIKE 10% OF THE TIME WITHOUT THIS LINE
        Btmp = Bnext

    

    K = size
    # relevant_a_part = A_I[:, (K - 1) * (k // size) : K * (k // size)]
    # C_I = C_I + np.matmul(relevant_a_part, Btmp)

    gathered_result = comm.gather(C_I, root=0)
    if rank == 0:
        return np.concatenate(gathered_result, axis=0)
    return None