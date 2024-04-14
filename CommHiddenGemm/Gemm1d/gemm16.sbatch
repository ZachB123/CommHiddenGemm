#!/usr/bin/env bash
#SBATCH -JGemm16                    # Name your job! This default is fine for the demo
#SBATCH --account=hive-rvuduc3              # Account to charge your hours -- use Rich's account
#SBATCH --nodes=2                           # Number of nodes required
#SBATCH --ntasks-per-node=8                 # Request all CPUs
#SBATCH --mem=0                             # Memory (0=request all node memory)
#SBATCH -t96:00:00                           # Duration of the job (Here: 10 minutes)
#SBATCH -phive                              # QOS Name
#SBATCH -oGemm16-%j.out                     # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL          # When to notify you of job-status changes
#SBATCH --mail-user=zbuchholz3@gatech.edu    # E-mail address for notifications -- @FIXME: please use yours

echo "Started on `/bin/hostname`"  # prints the name of the node job started on

module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn

srun python driver.py