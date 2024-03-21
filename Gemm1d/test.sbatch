#!/bun/bash

#SBATCH -A hive-rvuduc3
#SBATCH -phive
#SBATCH -JZACH_TEST
#SBATCH -n4
#SBATCH --mem=0
#SBATCH -t0-00-00:20
#SBATCH -oReport-%j.out
#BATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zbuchholz3@gatech.edu


cd $SLURM_SUBMIT_DIR

module load anaconda3/2022.05
module load numpy
module load mpi4py

srun -n 4 python driver.py -m 100 -k 100, -n 100
