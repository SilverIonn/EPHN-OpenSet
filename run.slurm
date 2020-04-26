#!/bin/bash
#SBATCH -N 1
#SBATCH -p large-gpu
#SBATCH -t 2-00:00:00 
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end  
#SBATCH --mail-user=xxxx@gwu.edu

srun python run.py --Data LMK --model R18 --dim 512 --lr 1e-2 --method EPSHN --imgsize 256 --g 2 --n 100 --c 64 --ep 20 --w 0.1