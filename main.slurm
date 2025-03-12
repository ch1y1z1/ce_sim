#!/bin/bash

#SBATCH --job-name=ce
#SBATCH --partition=64c512g
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=all
#SBATCH --mail-user=3045059088@qq.com
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

source .venv/bin/activate

python train.py -j $SLURM_JOB_ID -f ./Configuration/6bit.toml
