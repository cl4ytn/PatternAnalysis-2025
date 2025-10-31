#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=torch_test
#SBATCH --gres=gpu:a100
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 1️⃣  Load your environment (critical)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710

# 2️⃣  Move to your project directory
cd ~/PatternAnalysis-2025

# 3️⃣  Run training
echo "Running training on $(hostname) with Python from $(which python)"
python recognition/hipmri2d_improvedunet_s4905832/train.py

echo "Training finished at $(date)"

