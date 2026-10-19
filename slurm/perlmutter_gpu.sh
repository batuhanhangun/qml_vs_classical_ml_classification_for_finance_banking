#!/bin/bash
#SBATCH -A m5297
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task 1
#SBATCH -J qmlfin-torch
#SBATCH -o logs/%x-%A_%a.out
# PyTorch and quantum models (340 jobs). From the repository root:
#   mkdir -p logs && sbatch --array=0-339%40 slurm/perlmutter_gpu.sh

module load pytorch
cd "$SLURM_SUBMIT_DIR"
python -m qmlfin.grid --group torch --task-id "$SLURM_ARRAY_TASK_ID" --device cuda
