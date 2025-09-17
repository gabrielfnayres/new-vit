#!/bin/bash
#SBATCH --job-name=mst-gpu
#SBATCH --output=slurm-%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=dgx-b200          # example GPU partition on Betty
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/mst"

export PYTHONPATH=$PWD:$PYTHONPATH

hostname
nvidia-smi || true
uv run /vast/projects/bbruno/breast-imaging/new-vit/scripts/main_train.py --dataset DUKE --model DinoV3ClassifierSlice