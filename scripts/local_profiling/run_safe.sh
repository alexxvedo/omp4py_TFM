#!/usr/bin/env bash
# Memory-capped supplementary profiling runs (PC-safe: ulimit kills a runaway,
# not the machine). Fills MG and CG t4.
set -u
cd /home/av/Documents/Master/TFM_Separado/omp4py || exit 1
export PYTHONPATH=examples
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
PY315=$HOME/.pyenv/versions/3.15.0b1t/bin/python3.15t
OUT="$HOME/prof_local"
for spec in "MG 1" "MG 4" "MG 8" "CG 4"; do
  set -- $spec; B=$1; T=$2
  base="$OUT/${B}_S_m1_t${T}"
  echo "[$(date +%H:%M:%S)] $B t$T (cap 8GB, rate 1000)"
  OMP_NUM_THREADS=$T OMP_DYNAMIC=FALSE nice -n 10 bash -c "
    ulimit -v 8000000
    timeout 120 $PY315 -m profiling.sampling run -a --mode wall -r 1000 \
      --collapsed -o '${base}.collapsed' \
      examples/${B}_Python.py -c S -t $T -m 1" > "${base}.summary" 2>&1
  rc=$?
  ver=$(grep -oE "Verification[ =]+[A-Z]+" "${base}.summary" | grep -oE "[A-Z]+$" | head -1)
  st=$(wc -l < "${base}.collapsed" 2>/dev/null || echo 0)
  echo "    rc=$rc ver=${ver:-NA} stacks=$st"
  [ "$st" = "0" ] && rm -f "${base}.collapsed"
  sleep 2
done
echo "DONE $(date +%H:%M:%S)"
