from mpi4py import MPI
import numpy as np
import csv
import sys
import os

from GemmUtil.constants import MATRIX_DTYPE

BENCHMARK_FOLDER = "basic-benchmarks"
NUM_TRIALS = 100

os.makedirs(BENCHMARK_FOLDER, exist_ok=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

assert size >= 2, "You must have 2 processors"

max_send = 30
num_cores_per_node = "unknown"


if len(sys.argv) == 2:
    max_send = int(sys.argv[1])
if len(sys.argv) == 3:
    num_cores_per_node = int(sys.argv[2])

send_buffer_size = 1024

print(f"Max send is 2**{max_send}")


while send_buffer_size <= 2**max_send:
    for _ in range(NUM_TRIALS):
        data, buffer = [None] * 2

        data = np.ones(send_buffer_size, dtype=MATRIX_DTYPE)
        buffer = np.empty(send_buffer_size, dtype=MATRIX_DTYPE)

        start_time = MPI.Wtime()
        if rank == 0:
            # mpi4py does not allow very large regular sends
            # so we do async send followed by a wait to simulate one
            send_request = comm.Isend(np.ascontiguousarray(data), dest=1, tag=0)
            MPI.Request.wait(send_request)
            recieve_request = comm.Irecv(buffer, source=1, tag=MPI.ANY_TAG)
            MPI.Request.wait(recieve_request)
        if rank == 1:
            recieve_request = comm.Irecv(buffer, source=0, tag=MPI.ANY_TAG)
            MPI.Request.wait(recieve_request)
            send_request = comm.Isend(np.ascontiguousarray(data), dest=0, tag=0)
            MPI.Request.wait(send_request)

        elapsed_time = MPI.Wtime() - start_time  # exclude barrier from time
        comm.Barrier()

        # do as ping pong instead
        # report average time per send receive
        # do like 50-100 experiments

        if rank == 0:
            with open(
                f"{BENCHMARK_FOLDER}/python-N{size}-n{num_cores_per_node}-pingpong.csv",
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                # 2x the buffer size since it sends it to rank 1 then it sends it back
                # keep in mind for calculations that these are float64 so 4bytes per item sent
                writer.writerow([size, 2 * send_buffer_size, elapsed_time])

    send_buffer_size *= 2


# for whatever reason you can not doing a very large blocking p2p communication
# max I found on my lap top is sending 8677 float64s
# while send_buffer_size < 100000:
#     start_time = MPI.Wtime()
#     if rank == 0:
#         data = np.empty(send_buffer_size, dtype=MATRIX_DTYPE)
#         comm.send(data, 1)
#     if rank == 1:
#         # buffer = np.empty(send_buffer_size, dtype=MATRIX_DTYPE)
#         data = comm.recv(source = 1)
#     elapsed_time = MPI.Wtime() - start_time
#     with open(f"{BENCHMARK_FOLDER}/n{size}-pingpong", mode="a", newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([send_buffer_size, elapsed_time])

#     send_buffer_size += 1

# do i with isend and then with C
