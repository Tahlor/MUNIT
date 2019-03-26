#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
##SBATCH --exclusive   # number of nodes
#SBATCH --mem-per-cpu=16000M   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --output="./slurm/handwriting_online2.slurm"
#SBATCH --constraint rhel7&pascal

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module

cat /etc/os-release
cat /etc/redhat-release

module purge
export PATH="/fslhome/tarch/anaconda3/envs/munit/bin:$PATH"
which python

cd "/fslhome/tarch/compute/research/handwriting/MUNIT"
python -u train.py --config ./configs/handwriting_online.yaml

# To run:
#sbatch ./run.sh
#squeue -u tarch
