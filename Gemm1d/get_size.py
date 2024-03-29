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
    print(f"Size is {size}.")
    print(f"CPUS: {os.cpu_count()}")
    # threads = 0     #thread counter
    # y = 1000000     #a MILLION of 'em!
    # for i in range(y):
    #     try:
    #         x = threading.Thread(target=mythread, daemon=True)
    #         threads += 1    #thread counter
    #         x.start()       #start each thread
    #     except RuntimeError:    #too many throws a RuntimeError
    #         break
    # print("{} threads created.\n".format(threads))

if __name__ == "__main__":
    main()