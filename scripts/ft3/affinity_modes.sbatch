#!/usr/bin/env bash
#SBATCH --job-name=omp4py-affmodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --qos=short
#
# Comprueba si el colapso de mascara por afinidad afecta a TODOS los modos
# o solo al 3. EP, clase S, modos 0/1/2/3, 1/8/32 hilos, pin vs nopin.
# Misma asignacion y nodo; lo unico que cambia entre bloques es la variable.

set -uo pipefail
export PATH="/bin:/usr/bin:/usr/local/bin:$PATH"
export FT3_ROOT="${FT3_ROOT:-$HOME/omp4py_TFM}"
source "$FT3_ROOT/scripts/ft3/env.sh"
ft3_export_python_runtime 3.15t
PY="$(ft3_venv_python 3.15t)"
cd "$FT3_ROOT"

OUT="${OUT:-$FT3_ROOT/results/affmodes_$SLURM_JOB_ID}"
mkdir -p "$OUT"
export PYTHONPATH="$FT3_ROOT:$FT3_ROOT/examples/common:$FT3_ROOT/examples${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_GIL=0
export OMP_DYNAMIC=FALSE
CSV="$OUT/timings.csv"
echo "cond,bench,mode,threads,time_s,verification" > "$CSV"

run_timing () {  # cond mode thr
  local cond=$1 mode=$2 t=$3 raw tm vf
  raw="$OUT/raw_${cond}_m${mode}_t${t}.txt"
  OMP_NUM_THREADS=$t OMP4PY_CACHE=0 \
    timeout 1800 "$PY" "$FT3_ROOT/examples/EP_Python.py" -c S -m "$mode" -t "$t" > "$raw" 2>&1
  tm=$(grep "Time in seconds" "$raw" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  vf=$(grep "Verification" "$raw" | tail -1 | grep -oE 'SUCCESSFUL|UNSUCCESSFUL|NOT PERFORMED')
  echo "${cond},EP,${mode},${t},${tm:-ERR},${vf:-?}" >> "$CSV"
  echo "[${cond}] EP m${mode} t=${t} -> ${tm:-ERR}s ${vf:-?}"
}

for cond in nopin pin; do
  if [ "$cond" = "pin" ]; then
    export OMP_PROC_BIND=close OMP_PLACES=cores
  else
    unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
  fi
  echo "================ condicion: $cond ================"
  for mode in 0 1 2 3; do
    for t in 1 8 32; do
      run_timing "$cond" "$mode" "$t"
    done
  done
done

echo "RESULTADOS EN: $OUT"; cat "$CSV"
