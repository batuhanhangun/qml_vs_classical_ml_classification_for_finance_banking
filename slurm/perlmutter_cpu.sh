#!/bin/bash
#SBATCH -A m5297
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -J qmlfin-cpu
#SBATCH -o logs/%x-%A_%a.out
# Runs inside a podman-hpc container (set IMAGE to override). From the repository root:
#   mkdir -p logs
#   sbatch --array=0-419%60 slurm/perlmutter_cpu.sh                    # scikit-learn models (420 jobs)
#   GROUP=torch sbatch --array=0-339%60 slurm/perlmutter_cpu.sh        # PyTorch and quantum models (340 jobs)

IMAGE=${IMAGE:-localhost/quantum_ml:v5}
GROUP=${GROUP:-sklearn}
cd "$SLURM_SUBMIT_DIR"
podman-hpc run --rm -v "$PWD":/work -w /work -e PYTHONPATH=/work/src \
    -e OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" -e MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" "$IMAGE" \
    python -m qmlfin.grid --group "$GROUP" --task-id "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK" --device cpu
