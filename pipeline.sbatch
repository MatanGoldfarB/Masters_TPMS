#!/bin/bash
#SBATCH --partition=main               # CPU partition
#SBATCH --job-name=fem_pipeline        # Job name
#SBATCH --output=fem_pipeline-%J.out   # Stdout+stderr log
#SBATCH --time=0-08:00:00              # Max runtime (D-HH:MM:SS)
#SBATCH --cpus-per-task=8              # Cores *for this job*
#SBATCH --mem=256G                     # Memory for this job (increased for large meshes)
#SBATCH --mail-user=golmatan@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# Load conda and activate your FEM env
module load anaconda
source activate jax-fem-env

# Go to your code directory
cd /home/golmatan/MSc/code2

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Input OBJ = $OBJ_FILE"

# Run the pipeline with the OBJ file
./run_pipeline.sh "$OBJ_FILE"
