#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem 10666
#SBATCH --ntasks 6
#SBATCH --output="/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr/slurm_scripts/scripts/log_baseline.slurm"
#SBATCH --time 36:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="/fslhome/tarch/anaconda3/envs/munit2/bin:$PATH"
which python

cd "/fslhome/tarch/compute/research/handwriting/MUNIT"

python -u train.py --config ./configs/handwriting_online.yaml --check_files

# To run:
#sbatch ./run.sh
#squeue -u tarch
