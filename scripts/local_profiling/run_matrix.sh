#!/usr/bin/env bash
# Objective-5 profiling campaign with the new Python 3.15 sampling profiler.
# Mode 1 (hybrid: AST transform + real Python threads + pure runtime) so that
# the OMP4Py synchronization primitives are visible as Python frames.
set -u
cd /home/av/Documents/Master/TFM_Separado/omp4py || exit 1
export PYTHONPATH=examples
unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY
OUT="$HOME/prof_local"
mkdir -p "$OUT"
PY=python3.15t
RATE=2000
CLASS=S
MODE=1
echo "benchmark,class,mode,threads,rc,nas_seconds,verification" > "$OUT/runs_index.csv"
for B in EP FT CG MG; do
  for T in 1 4 8 16; do
    base="$OUT/${B}_${CLASS}_m${MODE}_t${T}"
    echo "[$(date +%H:%M:%S)] running $B c$CLASS m$MODE t$T"
    OMP_NUM_THREADS=$T OMP_DYNAMIC=FALSE timeout 500 \
      $PY -m profiling.sampling run -a --mode wall -r $RATE \
        --collapsed -o "${base}.collapsed" \
        examples/${B}_Python.py -c $CLASS -t $T -m $MODE > "${base}.summary" 2>&1
    rc=$?
    nas=$(grep -oE "Time in seconds[ =]+[0-9.]+" "${base}.summary" | grep -oE "[0-9.]+$" | head -1)
    ver=$(grep -oE "Verification[ =]+[A-Z]+" "${base}.summary" | grep -oE "[A-Z]+$" | head -1)
    echo "${B},${CLASS},${MODE},${T},${rc},${nas:-NA},${ver:-NA}" >> "$OUT/runs_index.csv"
    echo "    rc=$rc nas=${nas:-NA}s ver=${ver:-NA} stacks=$(wc -l < "${base}.collapsed" 2>/dev/null || echo 0)"
  done
done
echo "DONE $(date +%H:%M:%S)"
