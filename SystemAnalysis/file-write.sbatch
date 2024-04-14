#!/usr/bin/env bash
# This is an example of an MPI Slurm script
#SBATCH -JWrite-Test                # Name your job! This default is fine for the demo
#SBATCH --account=hive-rvuduc3              # Account to charge your hours -- use Rich's account
#SBATCH --nodes=2                           # Number of nodes required
#SBATCH --ntasks-per-node=3                 # Request all CPUs
#SBATCH --mem=0                             # Memory (0=request all node memory)
#SBATCH -t0:10:00                           # Duration of the job (Here: 10 minutes)
#SBATCH -phive                              # QOS Name
#SBATCH -oReport-%j.out                     # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL          # When to notify you of job-status changes
#SBATCH --mail-user=zbuchholz3@gatech.edu    # E-mail address for notifications -- @FIXME: please use yours

echo "Started on `/bin/hostname`"  # prints the name of the node job started on

module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn

# Runs the parallel c program with mpi. `-np` value should be <= `#SBATCH --nodes` line
srun python file_write_test.py