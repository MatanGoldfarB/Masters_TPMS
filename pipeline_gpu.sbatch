#!/bin/bash
#SBATCH --partition=gpu                # GPU partition
#SBATCH --job-name=fem_gpu             # Job name
#SBATCH --output=fem_gpu-%J.out        # Stdout+stderr log
#SBATCH --time=0-04:00:00              # Max runtime (D-HH:MM:SS)
#SBATCH --cpus-per-task=4              # Cores
#SBATCH --mem=64G                      # Memory
#SBATCH --gpus=1                       # Request 1 GPU
#SBATCH --mail-user=golmatan@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# Load conda and activate your FEM env
module load anaconda
source activate jax-fem-env

# Go to your code directory
cd /home/golmatan/MSc/code2

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Input OBJ = $OBJ_FILE"

# Run the pipeline with GPU version
./run_pipeline_gpu.sh "$OBJ_FILE"
