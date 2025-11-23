#!/bin/bash
#SBATCH --job-name=seld_conformer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load necessary modules (adjust based on Greene's environment)
module purge
module load python/3.9
module load cuda/11.3

# Activate virtual environment if you have one
# source /scratch/nd2746/env/bin/activate

# Install dependencies if needed (uncomment if running for the first time)
# pip install torch torchaudio torchvision matplotlib pandas tqdm

echo "Starting SELD Conformer Training..."
python main.py
echo "Training complete."
