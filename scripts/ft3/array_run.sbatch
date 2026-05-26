#!/usr/bin/env bash
#SBATCH --job-name=omp4py-npb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --qos=short

set -euo pipefail

export PATH="/bin:/usr/bin:/usr/local/bin:$PATH"
export FT3_ROOT="${FT3_ROOT:-$HOME/omp4py_TFM}"

source "$FT3_ROOT/scripts/ft3/env.sh"
ft3_export_python_runtime "${RUNNER_PYTHON_TAG:-3.15t}"

RUNNER_PYTHON="$(ft3_venv_python "${RUNNER_PYTHON_TAG:-3.15t}")"
cd "$FT3_ROOT"

"$RUNNER_PYTHON" "$FT3_ROOT/scripts/ft3/run_one.py" \
    --manifest "$MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --campaign-dir "$CAMPAIGN_DIR"
