#!/usr/bin/env bash
#SBATCH --job-name=omp4py-affcheck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --qos=short
#
# Comprobacion controlada del efecto de afinidad en FT3.
# Todo corre en UNA sola asignacion y nodo, asi que pin y nopin comparten
# binario, version, NumPy y nodo. La unica variable que cambia entre ambos
# bloques son OMP_PROC_BIND / OMP_PLACES / GOMP_CPU_AFFINITY. Eso aisla la
# causa y responde a "no habreis cambiado nada mas".

set -uo pipefail
export PATH="/bin:/usr/bin:/usr/local/bin:$PATH"
export FT3_ROOT="${FT3_ROOT:-$HOME/omp4py_TFM}"
source "$FT3_ROOT/scripts/ft3/env.sh"
ft3_export_python_runtime 3.15t
PY="$(ft3_venv_python 3.15t)"
cd "$FT3_ROOT"

OUT="${OUT:-$FT3_ROOT/results/affcheck_$SLURM_JOB_ID}"
mkdir -p "$OUT"
export PYTHONPATH="$FT3_ROOT:$FT3_ROOT/examples/common:$FT3_ROOT/examples${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_GIL=0
export OMP_DYNAMIC=FALSE
CSV="$OUT/timings.csv"
echo "cond,bench,mode,threads,time_s,verification" > "$CSV"

# ---- 0. Que inyecta el entorno/SLURM ANTES de que toquemos nada -----------
{
  echo "### nodo: $(hostname)"
  echo "### Cpus_allowed_list (proceso del job):"
  grep -i Cpus_allowed_list /proc/self/status
  echo "### taskset del shell:"; taskset -cp $$ 2>/dev/null
  echo "### variables de afinidad/SLURM heredadas:"
  env | grep -iE 'OMP_|GOMP_|KMP_|AFFINITY|PROC_BIND|OMP_PLACES|SLURM_(CPU|TASK|DISTRIBUTION|BIND)' | sort
} > "$OUT/env_inicial.txt" 2>&1

# Sonda "tal cual": el entorno como lo deja SLURM/los modulos, sin unset ni set.
# Reproduce el estado preliminar. Si aqui la mascara ya sale reducida, la causa
# estaba en el entorno y no en OMP4Py.
"$PY" "$FT3_ROOT/scripts/ft3/affinity_check.py" > "$OUT/probe_asis.json" 2>&1

run_timing () {  # cond bench thr
  local cond=$1 bench=$2 t=$3 raw tm vf
  raw="$OUT/raw_${cond}_${bench}_t${t}.txt"
  OMP_NUM_THREADS=$t OMP4PY_CACHE=1 OMP4PY_CACHE_DIR="$OUT/cache/$cond" \
    timeout 1200 "$PY" "$FT3_ROOT/examples/${bench}_Python.py" -c S -m 3 -t "$t" > "$raw" 2>&1
  tm=$(grep "Time in seconds" "$raw" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  vf=$(grep "Verification" "$raw" | tail -1 | grep -oE 'SUCCESSFUL|UNSUCCESSFUL|NOT PERFORMED')
  echo "${cond},${bench},3,${t},${tm:-ERR},${vf:-?}" >> "$CSV"
  echo "[${cond}] ${bench} m3 t=${t} -> ${tm:-ERR}s ${vf:-?}"
}

for cond in nopin pin; do
  if [ "$cond" = "pin" ]; then
    export OMP_PROC_BIND=close OMP_PLACES=cores
  else
    unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
  fi
  echo "================ condicion: $cond ================"
  # 1. sonda de mascara (el eslabon que faltaba)
  "$PY" "$FT3_ROOT/scripts/ft3/affinity_check.py" > "$OUT/probe_${cond}.json" 2>&1
  # 2. A/B de tiempos: EP y FT (escalan) + CG (control, no escala)
  for bench in EP FT CG; do
    for t in 1 2 4 8 16 32; do
      run_timing "$cond" "$bench" "$t"
    done
  done
done

echo "RESULTADOS EN: $OUT"
echo "==== probe nopin ===="; cat "$OUT/probe_nopin.json"
echo "==== probe pin   ===="; cat "$OUT/probe_pin.json"
echo "==== timings ===="; cat "$CSV"
