# hello-hpc-cluster #

"Hello, world!" for an MPI+OpenMP program.

## Platform: Hive cluster ##

**What the demo does on Hive**: The Hive system consists of over 400 nodes (i.e., physical servers) with two CPUs per node (two "sockets"), where each CPU has 12 cores. The Hive version of the demo will launch a little "Hello, world!" style program on the cluster using 8 MPI processes and 12 OpenMP threads per process (so 96 threads in total). The job script specifically asks for 4 physical nodes (e.g., "servers"), 2 CPU sockets per node, and 12 cores per socket, with one MPI process per CPU socket and 12 OpenMP threads per core. Each thread prints a message that reports its physical location in the cluster.

**How to build and run the demo**: Here are the steps to get the demo up and running on Hive:

1. Log into the Hive cluster (sub your GT login for `gburdell3`):
   `ssh gburdell3@login-hive.pace.gatech.edu`

2. Clone the "HPC Hello, World" demo (which can help verify an MPI+OpenMP configuration):
   `git clone https://github.gatech.edu/hpcgarage/hello-hpc-cluster.git`

3. Change into the demo directory:
   `cd hello-hpc-cluster`

4. Build (compile) the demo:
   `make`

5. Modify the job submission script, `pace-hive.sbatch` (search for the `@FIXME`):

6. Submit the job! When you do so, it will emit a "job number," which you might want to note.
   `sbatch pace-hive.sbatch`

7. To check on the job, use the following with the job number in place of `<JOBID>`:
   `squeue -j <JOBID>`

>  The job will probably execute quickly if the machine is lightly loaded, so it's possible the `squeue` command above will simply appear empty.

8. If everything went well, there should be a file with the name, `Report-<JOBID>.out` in the directory. Inspect its output to make sure there were no errors.

> For documentation on the Hive cluster documentation, refer to this link: [Getting Started with Hive - PACE Cluster Documentation](https://docs.pace.gatech.edu/hive/gettingStarted/)

## Platform: PACE-ICE (instructional cluster for classes) ##

If running on PACE-ICE, use `make` to build and `pace-ics.sbatch` as the sample job script.
