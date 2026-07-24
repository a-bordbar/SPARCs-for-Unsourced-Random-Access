#!/bin/bash -l
#SBATCH -A C3SE2026-1-20
#SBATCH -p vera
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH --open-mode=truncate
#SBATCH -o /cephyr/users/bordbar/Vera/Massive_Random_Access_fig9_python_%j.out
#SBATCH -e /cephyr/users/bordbar/Vera/Massive_Random_Access_fig9_python_%j.out

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile ] && source /etc/profile
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
    [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi

ml numba/0.58.1-foss-2023a
ml matplotlib
ml tqdm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-fig9}"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python executable was not found after module initialization."
    module list 2>&1 || true
    exit 127
fi

python - <<'PY'
import numba
import matplotlib
import tqdm
print("Numba version:", numba.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("tqdm version:", tqdm.__version__)
PY

PROJECT_DIR="/cephyr/users/bordbar/Vera/Massive Random Access"
SCRIPT_PATH="${PROJECT_DIR}/fig9_faithful_empirical_amp_tree.py"

cd "${PROJECT_DIR}"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "Python executable: $(command -v python)"
echo "Script: ${SCRIPT_PATH}"
echo

srun python "${SCRIPT_PATH}" production --resume

echo
echo "Job finished: $(date)"
