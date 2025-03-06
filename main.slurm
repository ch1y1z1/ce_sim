#!/bin/bash

#SBATCH --job-name=ce
#SBATCH --partition=64c512g
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=all
#SBATCH --mail-user=3045059088@qq.com
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

module load miniconda3
source activate
conda activate ce

python layers.py
