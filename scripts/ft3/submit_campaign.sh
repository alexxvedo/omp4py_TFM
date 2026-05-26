#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

profile="${1:-pilot}"
if [ -n "${PARTITION:-}" ]; then
    partition="$PARTITION"
elif [ "$profile" = "pilot" ]; then
    partition="short"
else
    partition="medium"
fi
qos="${QOS:-$partition}"
constraint="${CONSTRAINT:-}"
cpus_per_task="${CPUS_PER_TASK:-32}"
mem_per_cpu="${MEM_PER_CPU:-4G}"
max_parallel="${MAX_PARALLEL:-8}"
rows_per_task="${ROWS_PER_TASK:-8}"
if [ -n "${TIME_LIMIT:-}" ]; then
    time_limit="$TIME_LIMIT"
elif [ "$profile" = "pilot" ]; then
    time_limit="06:00:00"
else
    time_limit="3-00:00:00"
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
campaign_dir="${CAMPAIGN_DIR:-$FT3_ROOT/results/${timestamp}_${profile}}"
manifest="$campaign_dir/manifest.csv"

mkdir -p "$campaign_dir/raw" "$campaign_dir/records" "$campaign_dir/slurm" "$campaign_dir/cache"

ft3_export_python_runtime 3.15t
runner_python="$(ft3_venv_python 3.15t)"
"$runner_python" "$SCRIPT_DIR/make_matrix.py" --profile "$profile" --output "$manifest"

run_count="$(( $(wc -l < "$manifest") - 1 ))"
if [ "$run_count" -le 0 ]; then
    printf 'empty manifest: %s\n' "$manifest" >&2
    exit 1
fi

{
    printf 'date = %s\n' "$(date -Is)"
    printf 'host = %s\n' "$(hostname)"
    printf 'root = %s\n' "$FT3_ROOT"
    printf 'profile = %s\n' "$profile"
    printf 'partition = %s\n' "$partition"
    printf 'qos = %s\n' "$qos"
    printf 'constraint = %s\n' "$constraint"
    printf 'cpus_per_task = %s\n' "$cpus_per_task"
    printf 'mem_per_cpu = %s\n' "$mem_per_cpu"
    printf 'max_parallel = %s\n' "$max_parallel"
    printf 'rows_per_task = %s\n' "$rows_per_task"
    printf 'time_limit = %s\n' "$time_limit"
    printf 'run_count = %s\n' "$run_count"
    "$runner_python" -c 'import sys; print("runner =", sys.version.replace("\n", " "))'
} > "$campaign_dir/env.txt"

array_task_count="$(( (run_count + rows_per_task - 1) / rows_per_task ))"

sbatch_args=(
    --partition="$partition" \
    --qos="$qos" \
    --cpus-per-task="$cpus_per_task" \
    --mem-per-cpu="$mem_per_cpu" \
    --time="$time_limit" \
    --array="1-${array_task_count}%${max_parallel}" \
    --output="$campaign_dir/slurm/%x-%A_%a.out" \
    --error="$campaign_dir/slurm/%x-%A_%a.err" \
    --export=ALL,FT3_ROOT="$FT3_ROOT",CAMPAIGN_DIR="$campaign_dir",MANIFEST="$manifest",RUN_COUNT="$run_count",ROWS_PER_TASK="$rows_per_task" \
)

if [ -n "$constraint" ]; then
    sbatch_args+=(--constraint="$constraint")
fi

/bin/sbatch "${sbatch_args[@]}" "$SCRIPT_DIR/array_run.sbatch"

printf 'campaign = %s\n' "$campaign_dir"
printf 'manifest = %s\n' "$manifest"
printf 'runs = %s\n' "$run_count"
printf 'array_tasks = %s\n' "$array_task_count"
