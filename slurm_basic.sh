#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=test1.txt
#SBATCH --error=test1error.txt
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --partition=log_gpu_24gb

export PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate GTencoding

python main.py