#!/usr/bin/env bash
#SBATCH -Jpingpong_python_and_C
#SBATCH --account=hive-rvuduc3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -t24:00:00
#SBATCH -pPingPongPythonC
#SBATCH -ohive-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zbuchholz3@gatech.edu
echo "Started on `/bin/hostname`"
module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn
srun python pingpong.py 31 1
make
srun ./pingpong 31 1
