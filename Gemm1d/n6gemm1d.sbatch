#!/bin/bash

#SBATCH --account hive-rvuduc3
#SBATCH -phive
#SBATCH -Jn6CHG
#SBATCH -N1 # number of nodes
#SBATCH -n6 # number of cores
#SBATCH --mem=0
#SBATCH -t45
#SBATCH -oReport-%j.out
#BATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zbuchholz3@gatech.edu


cd $SLURM_SUBMIT_DIR

module load anaconda3
module load py-mpi4py/3.1.2-mva2-rzdjbn


srun python driver.py