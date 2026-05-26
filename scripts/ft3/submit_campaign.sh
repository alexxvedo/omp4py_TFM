#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

profile="${1:-pilot}"
partition="${PARTITION:-short}"
qos="${QOS:-$partition}"
cpus_per_task="${CPUS_PER_TASK:-64}"
mem_per_cpu="${MEM_PER_CPU:-4G}"
max_parallel="${MAX_PARALLEL:-8}"
time_limit="${TIME_LIMIT:-06:00:00}"
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
    printf 'cpus_per_task = %s\n' "$cpus_per_task"
    printf 'mem_per_cpu = %s\n' "$mem_per_cpu"
    printf 'max_parallel = %s\n' "$max_parallel"
    printf 'time_limit = %s\n' "$time_limit"
    printf 'run_count = %s\n' "$run_count"
    "$runner_python" -c 'import sys; print("runner =", sys.version.replace("\n", " "))'
} > "$campaign_dir/env.txt"

/bin/sbatch \
    --partition="$partition" \
    --qos="$qos" \
    --cpus-per-task="$cpus_per_task" \
    --mem-per-cpu="$mem_per_cpu" \
    --time="$time_limit" \
    --array="1-${run_count}%${max_parallel}" \
    --output="$campaign_dir/slurm/%x-%A_%a.out" \
    --error="$campaign_dir/slurm/%x-%A_%a.err" \
    --export=ALL,FT3_ROOT="$FT3_ROOT",CAMPAIGN_DIR="$campaign_dir",MANIFEST="$manifest" \
    "$SCRIPT_DIR/array_run.sbatch"

printf 'campaign = %s\n' "$campaign_dir"
printf 'manifest = %s\n' "$manifest"
printf 'runs = %s\n' "$run_count"
