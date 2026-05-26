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
PARTITION=medium QOS=medium TIME_LIMIT=1-00:00:00 bash scripts/ft3/submit_campaign.sh full315
```

Submit the Python-version comparison:

```bash
PARTITION=medium QOS=medium TIME_LIMIT=1-00:00:00 bash scripts/ft3/submit_campaign.sh versions
```

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
