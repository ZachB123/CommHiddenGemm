#!/usr/bin/env bash
# This is an example of an MPI Slurm script
#SBATCH -JBroadcast6C                # Name your job! This default is fine for the demo
#SBATCH --account=hive-rvuduc3              # Account to charge your hours -- use Rich's account
#SBATCH --nodes=1                           # Number of nodes required
#SBATCH --ntasks-per-cpu=6                         # Request all CPUs
#SBATCH --mem=0                             # Memory (0=request all node memory)
#SBATCH -t6:00:00                           # Duration of the job (Here: 10 minutes)
#SBATCH -phive                              # QOS Name
#SBATCH -0Broadcast6C-%j.out                     # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL          # When to notify you of job-status changes
#SBATCH --mail-user=zbuchholz3@gatech.edu    # E-mail address for notifications -- @FIXME: please use yours

# Diagnostic:
echo "Started on `/bin/hostname`"  # prints the name of the node job started on

# Build (again):
make

# Runs the parallel c program with mpi. `-np` value should be <= `#SBATCH --nodes` line
srun ./broadcastbenchmark

# eof