#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8192M
#SBATCH -J "minGPT ablations"
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=matthew.merrill.ward@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

python training/train.py "$@"
