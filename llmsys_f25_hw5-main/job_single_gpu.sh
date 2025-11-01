#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 5:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --job-name=gpt2_single_gpu
#SBATCH --output=logs/single_gpu_%j.out
#SBATCH --error=logs/single_gpu_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load anaconda module
module load anaconda3/2024.10-1
module load cuda/12.4.0
# Activate conda environment
conda activate hw5

# Navigate to project directory
cd /jet/home/kshah10/cmu-11868-llm-systems/llmsys_f25_hw5-main

# Run single GPU training with 5 epochs
python project/run_data_parallel.py \
    --world_size 1 \
    --batch_size 64 \
    --n_epochs 20 \
    --workdir workdir_single

echo "Single GPU training completed!"

