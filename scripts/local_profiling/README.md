# Objective-5 profiling: Python 3.15 sampling profiler

Complementary profiling study for the TFM (objective 5: synchronization cost,
load balance, resource usage with the new Python 3.15 profiling system).

Run on a local 16-core free-threaded **Python 3.15.0b1t** node, because the
`profiling.sampling` profiler cannot attach to the FT3 build of the beta
interpreter (it fails to locate the runtime in the shared `libpython`). The
FT3 `perf stat` campaign remains the resource-usage reference; this study adds
the per-function / per-thread attribution of synchronization time.

- `run_matrix.sh`  — EP/FT/CG/MG x {1,4,8,16} threads, class S, mode 1, sampler
  (`profiling.sampling run -a --mode wall`), collapsed output.
- `run_safe.sh`    — memory-capped (`ulimit -v`) supplementary runs (CG/MG churn
  thousands of ephemeral threads in mode 1 and can exhaust RAM under the sampler).
- `parse.py`       — categorizes leaf frames into compute / synchronization /
  init and computes per-thread load balance -> `prof_summary.csv`.
- `gen_tables.py`  — emits the LaTeX tables `profiling_py315_{sync,churn}.tex`.
- `prof_summary.csv` — parsed results behind the tables.

Mode 1 (interpreted, real Python threads) is used so the OMP4Py barrier/reduction
primitives appear as Python frames; in mode 3 they compile to native code and are
invisible to a Python stack sampler.
