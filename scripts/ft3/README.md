# Finisterrae III benchmark scripts

These scripts keep the NAS/OMP4Py runs isolated from older work in
`~/Master/TFM`. The expected root on FT3 is:

```bash
~/omp4py_TFM
```

Prepare the virtual environments:

```bash
cd ~/omp4py_TFM
bash scripts/ft3/setup_venvs.sh
```

Submit a short pilot:

```bash
cd ~/omp4py_TFM
bash scripts/ft3/submit_campaign.sh pilot
```

Submit the full Python 3.15 campaign:

```bash
bash scripts/ft3/submit_campaign.sh full315
```

Submit the Python-version comparison:

```bash
bash scripts/ft3/submit_campaign.sh versions
```

Submit only the missing 2- and 8-thread rows for the Python-version comparison:

```bash
bash scripts/ft3/submit_campaign.sh versions_extra_threads
```

Submit the Python-version comparison for interpreter-dominated execution:

```bash
bash scripts/ft3/submit_campaign.sh versions_interpreted_s
```

This campaign uses Python 3.13t, 3.14t and 3.15t, class S, modes 0 and 1,
threads 1, 2, 4, 8, 16 and 32, and five repetitions per combination.

Submit the profiling campaign used to compare the Python-version scaling:

```bash
bash scripts/ft3/submit_profile_campaign.sh profile_versions_core
```

Submit the Python 3.15 standard-library profiler campaign (sampling + tracing):

```bash
bash scripts/ft3/submit_py315_profile_campaign.sh py315_both
```

This campaign uses class S, mode 1, Python 3.15t, no explicit OpenMP affinity,
and real Python threads so that the OMP4Py runtime synchronization functions are
visible to `profiling.sampling` and `profiling.tracing`. Use
`py315_sampling` or `py315_tracing` to submit only one half.

`full315` and `versions` default to `partition=medium`, `qos=medium`,
`cpus-per-task=32`, and a 3-day Slurm limit. `versions_interpreted_s`
defaults to the short queue with a 6-hour Slurm limit. Set `CONSTRAINT=clk`,
`hwl`, `epyc`, or `ilk` to keep all runs on one node family.

`ROWS_PER_TASK=8` by default, so each Slurm array element executes eight
manifest rows sequentially. This keeps large campaigns below common array and
submit-count limits.

Every campaign writes one folder under `results/` with:

- `manifest.csv`: complete run matrix.
- `raw/`: semantic raw logs, named by Python, benchmark, class, mode, thread count and repetition.
- `records/`: one JSON result per run.
- `summary.csv`: flat CSV for all runs.
- `summary_best.csv`: grouped best/mean values.
- `env.txt`: campaign metadata.

By default, each run gets its own OMP4Py cache directory. This avoids races when
Slurm starts repeated runs of the same benchmark at the same time. To reuse one
cache per Python version, export `OMP4PY_CACHE_SCOPE=tag` only after a warmup has
already populated the cache.

Refresh summaries after a campaign finishes:

```bash
python scripts/ft3/parse_results.py results/<campaign>
```

Refresh profiling summaries after a profiling campaign finishes:

```bash
python scripts/ft3/parse_profile_results.py results/<campaign>
```

Refresh Python 3.15 profiler summaries after a sampling/tracing campaign
finishes:

```bash
python scripts/ft3/parse_py315_profile_results.py results/<campaign>
```
