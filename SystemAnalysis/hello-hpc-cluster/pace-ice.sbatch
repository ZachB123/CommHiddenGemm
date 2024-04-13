#!/usr/bin/bash
#SBATCH --job-name=hello-hpc-cluster  # Name this job!
#SBATCH --nodes=4                     # Number of nodes
#SBATCH --ntasks-per-node=2           # Tasks (processes) per node
#SBATCH --cpus-per-task=12            # Cores per task
#SBATCH --time=00:05:00               # Time: `hh:mm:ss`
#SBATCH --exclusive                   # Request exclusive access to nodes
#SBATCH --mem=0                       # Memory per node (0 == all)

echo "Started on $(/bin/hostname) in $(pwd)"  # prints the name of the node job started on

echo "*** STARTED: `date` on `hostname` ***"
echo -e "\n\n"

set -x # echo commands to stdout for logging purposes

# Set up the environment with preferred software, e.g., compilers
#module load ...

# Build and run
make && env OMP_NUM_THREADS=12 srun ./hello-hpc-cluster

# eof