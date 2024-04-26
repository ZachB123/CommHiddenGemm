from mpi4py import MPI
import numpy as np
import csv
import sys
import os

BENCHMARK_FOLDER = "basic-benchmarks"
NUM_TRIALS = 100

os.makedirs(BENCHMARK_FOLDER, exist_ok=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

assert size >= 2, "You must have 2 processors"

max_send = 30
num_cores_per_node = 1

if len(sys.argv) >= 2:
    max_send = int(sys.argv[1])
if len(sys.argv) == 3:
    num_cores_per_node = int(sys.argv[2])

send_buffer_size = 1024
iterations = 0

print(f"Max send is 2**{max_send}")

while send_buffer_size <= 2**max_send:
    for _ in range(NUM_TRIALS):
        data, buffer = [None] * 2

        data = np.ones(send_buffer_size, dtype=np.float64)
        # buffer = np.empty(send_buffer_size, dtype=np.float64)

        start_time = MPI.Wtime()

        received_data = comm.bcast(data, 0)

        comm.Barrier()
        elapsed_time = MPI.Wtime() - start_time

        if rank == 0:
            with open(
                f"{BENCHMARK_FOLDER}/python-N{size}-n{num_cores_per_node}-broadcastbenchmark.csv",
                mode="a",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow([size, send_buffer_size, elapsed_time])

    send_buffer_size *= 2
