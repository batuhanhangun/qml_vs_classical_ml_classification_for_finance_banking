#!/bin/bash
#SBATCH -A m5297
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -J qmlfin-sklearn
#SBATCH -o logs/%x-%A_%a.out
# Scikit-learn models (420 jobs). From the repository root:
#   mkdir -p logs && sbatch --array=0-419%60 slurm/perlmutter_cpu.sh

module load pytorch
cd "$SLURM_SUBMIT_DIR"
python -m qmlfin.grid --group sklearn --task-id "$SLURM_ARRAY_TASK_ID" --n-jobs 8
