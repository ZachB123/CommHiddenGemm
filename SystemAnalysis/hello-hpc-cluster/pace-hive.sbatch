#!/usr/bin/env bash
# This is an example of an MPI Slurm script
#SBATCH -Jhello-hpc-cluster                 # Name your job! This default is fine for the demo
#SBATCH --account=hive-rvuduc3              # Account to charge your hours -- use Rich's account
#SBATCH --nodes=4                           # Number of nodes required
#SBATCH --exclusive                         # Request all CPUs
#SBATCH --mem=0                             # Memory (0=request all node memory)
#SBATCH -t0:10:00                           # Duration of the job (Here: 10 minutes)
#SBATCH -phive                              # QOS Name
#SBATCH -oReport-%j.out                     # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL          # When to notify you of job-status changes
#SBATCH --mail-user=gburdell3@gatech.edu    # E-mail address for notifications -- @FIXME: please use yours

# Diagnostic:
echo "Started on `/bin/hostname`"  # prints the name of the node job started on

# Build (again):
make

# Runs the parallel c program with mpi. `-np` value should be <= `#SBATCH --nodes` line
srun --nodes=4 --ntasks-per-node=2 --cpus-per-task=12 --cpu-bind=socket \
    ./hello-hpc-cluster

# eof
