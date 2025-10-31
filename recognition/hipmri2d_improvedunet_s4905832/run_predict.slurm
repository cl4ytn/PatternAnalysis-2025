#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=torch_test
#SBATCH --gres=gpu:a100
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710
cd ~/PatternAnalysis-2025

python recognition/hipmri2d_improvedunet_s4905832/predict.py

