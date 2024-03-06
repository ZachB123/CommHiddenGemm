import numpy as np
from mpi4py import MPI

from util import MATRIX_DTYPE

def allgather_A_col(A_I, B_I, C_I, out):


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    # A m x k
    # B k x n
    # C m x n
    m = A_I.shape[0]
    k = B_I.shape[0]
    n = C_I.shape[1] * size


    prev_rank = (rank + size - 1) % size
    next_rank = (rank + 1) % size

    for cycle in range(size):

        # buffer has n rows and k/size columns
        buffer = np.empty((m, int(k/size)), dtype=MATRIX_DTYPE) 
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


    # Any better way to do this than to transpose? it seems kinda awkward
    comm.Gather(C_I, out, root=0)






