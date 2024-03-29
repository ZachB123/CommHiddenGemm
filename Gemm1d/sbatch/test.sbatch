#!/bin/bash

#SBATCH --account hive-rvuduc3
#SBATCH -phive
#SBATCH -JZACH_TEST
#SBATCH -N1 # number of nodes
#SBATCH -n24 # number of cores
#SBATCH --mem=0
#SBATCH -t1
#SBATCH -oReport-%j.out
#BATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zbuchholz3@gatech.edu


cd $SLURM_SUBMIT_DIR


module load libxml2 #/2.9.13-d4fgiv
module load zlib #/1.2.7-s3gked
module load gcc #/10.3.0-o57x6h
module load mvapich2 #/2.3.6-ouywal
module load anaconda3
# module load python/3.9.12-rkxvr6
module load py-mpi4py/3.1.2-mva2-rzdjbn
module load py-pip/21.3.1-c3xscg

srun pip install numpy

srun python driver.py -m 100 -k 100, -n 100


