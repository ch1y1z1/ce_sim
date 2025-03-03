#!/bin/bash

#SBATCH --job-name=ce
#SBATCH --partition=64c512g
#SBATCH -n 20
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=all
#SBATCH --mail-user=3045059088@qq.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
source activate
conda activate ce

python layers.py
