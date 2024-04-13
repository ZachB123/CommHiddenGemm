from mpi4py import MPI
import threading
import os
import time

# this file is just for running on the super computer so that I can verify that the number
# of processors is what I think it is


def mythread():
    time.sleep(1000)


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    cpus = os.cpu_count()
    hostname = MPI.Get_processor_name()
    version = MPI.Get_version()
    library_version = MPI.Get_library_version()

    print(
        f"Rank: {rank}, size: {size}, hostname: {hostname}, version: {version}, CPUS: {cpus}"
    )
    if rank == 0:
        print(f"library version: {library_version}")
    # print(f"CPUS: {os.cpu_count()}")


if __name__ == "__main__":
    main()
