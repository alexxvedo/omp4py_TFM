#!/usr/bin/env bash
#SBATCH --job-name=omp4py-prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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

rows_per_task="${ROWS_PER_TASK:-1}"
run_count="${RUN_COUNT:?RUN_COUNT is required}"
start_index="$(( (SLURM_ARRAY_TASK_ID - 1) * rows_per_task + 1 ))"
end_index="$(( SLURM_ARRAY_TASK_ID * rows_per_task ))"
if [ "$end_index" -gt "$run_count" ]; then
    end_index="$run_count"
fi

status=0
for index in $(seq "$start_index" "$end_index"); do
    "$RUNNER_PYTHON" "$FT3_ROOT/scripts/ft3/profile_one.py" \
        --manifest "$MANIFEST" \
        --index "$index" \
        --campaign-dir "$CAMPAIGN_DIR" || status=$?
done

exit "$status"
