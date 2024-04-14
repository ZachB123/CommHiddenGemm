# this file is to test if running sbatch jobs still writes to like a singular file
# use multiple nodes to check that too
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


with open(f"test{rank}.txt", "w") as file:
    file.write("yo")