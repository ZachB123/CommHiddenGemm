from mpi4py import MPI
import numpy as np
import csv
import sys

BENCHMARK_FOLDER = "benchmarks"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

assert size >= 2, "You must have 2 processors"

max_send = 30

if len(sys.argv) == 2:
    max_send = int(sys.argv[1])

send_buffer_size = 2
iterations = 0

print(f"Max send is 2**{max_send}")

while send_buffer_size <= 2**max_send:
    data, buffer = [None] * 2

    data = np.ones(send_buffer_size, dtype=np.float64)
    buffer = np.empty(send_buffer_size, dtype=np.float64)

    start_time = MPI.Wtime()
    if rank == 0:
        send_request = comm.Isend(np.ascontiguousarray(data), dest=1, tag=0)
        MPI.Request.wait(send_request)
    if rank == 1:
        recieve_request = comm.Irecv(buffer, source=0, tag=MPI.ANY_TAG)
        MPI.Request.wait(recieve_request)

    comm.Barrier()
    elapsed_time = MPI.Wtime() - start_time
    
    if rank == 0:
        with open(f"{BENCHMARK_FOLDER}/python-n{size}-sendbenchmark.csv", mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([size, send_buffer_size, elapsed_time])

    iterations += 1
    if iterations >= 10:
        send_buffer_size *= 2
        iterations = 0

# for whatever reason you can not doing a very large blocking p2p communication
# max I found on my lap top is sending 8677 float64s
# while send_buffer_size < 100000:
#     start_time = MPI.Wtime()
#     if rank == 0:
#         data = np.empty(send_buffer_size, dtype=np.float64)
#         comm.send(data, 1)
#     if rank == 1:
#         # buffer = np.empty(send_buffer_size, dtype=np.float64)
#         data = comm.recv(source = 1)
#     elapsed_time = MPI.Wtime() - start_time
#     with open(f"{BENCHMARK_FOLDER}/n{size}-pingpong", mode="a", newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([send_buffer_size, elapsed_time])

#     send_buffer_size += 1

# do i with isend and then with C