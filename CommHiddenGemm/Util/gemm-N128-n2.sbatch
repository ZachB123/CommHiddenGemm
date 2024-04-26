#!/usr/bin/env bash
#SBATCH -JGemm-128:2
#SBATCH --account=hive-rvuduc3
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0
#SBATCH -t60:00:00
#SBATCH -phive
#SBATCH -oGemm-128:2-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zbuchholz3@gatech.edu
echo "Started on `/bin/hostname`"
module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn
pip install -r ../../requirements.txt
srun python driver.py 2
