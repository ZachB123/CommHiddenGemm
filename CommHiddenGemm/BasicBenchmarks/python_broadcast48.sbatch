#!/usr/bin/env bash
#SBATCH -JBroadcast48Python                    # Name your job! This default is fine for the demo
#SBATCH --account=hive-rvuduc3              # Account to charge your hours -- use Rich's account
#SBATCH --nodes=4                           # Number of nodes required
#SBATCH --ntasks-per-node=12                 # Request all CPUs
#SBATCH --mem=0                             # Memory (0=request all node memory)
#SBATCH -t6:00:00                           # Duration of the job (Here: 10 minutes)
#SBATCH -phive                              # QOS Name
#SBATCH -oBroadcastPython48-%j.out                     # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL          # When to notify you of job-status changes
#SBATCH --mail-user=zbuchholz3@gatech.edu    # E-mail address for notifications -- @FIXME: please use yours

echo "Started on `/bin/hostname`"  # prints the name of the node job started on

module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn

# Runs the parallel c program with mpi. `-np` value should be <= `#SBATCH --nodes` line
srun python broadcast_benchmark.py