# Finisterrae III benchmark results

The final benchmark data used by the TFM memory is under `ft3_20260601`.
These campaigns were run on Finisterrae III with OpenMP affinity pinning disabled
by default (`OMP_PROC_BIND`, `OMP_PLACES` and `GOMP_CPU_AFFINITY` unset by the
runner scripts).

Final campaigns:

- `ft3_20260601/20260531_202503_full315_nopin`: main Python 3.15.0b1t
  free-threaded campaign. It contains 630 successful runs for CG, EP, FT, IS and
  MG across classes S, W and A.
- `ft3_20260601/20260531_202503_versions_nopin`: base Python-version comparison
  campaign. It contains 960 successful runs for Python 3.13.13t, 3.14.4t and
  3.15.0b1t across classes S and W, modes 2 and 3, using 1, 4, 16 and
  32 threads.
- `ft3_20260601/20260531_202503_versions_extra_threads_nopin`: additional
  comparison campaign for the missing powers of two. It contains 480 successful
  runs with 2 and 8 threads.
- `ft3_20260601/20260531_202503_profile_versions_core_nopin`: `perf stat`
  profiling campaign. It contains 43 successful profiling runs used for the
  diagnostic chapter.

Each campaign includes:

- `manifest.csv`: execution matrix.
- `env.txt`: Slurm, Python and affinity metadata.
- `summary.csv` or `profile_summary.csv`: parsed result rows.
- `summary_best.csv` or `profile_summary_best.csv`: grouped best values.

Earlier `ft3_20260526`, `ft3_20260528` and `ft3_20260530` directories are
historical snapshots from preliminary or fill-in campaigns. They are kept only
for traceability and are not the source of the final TFM tables.
