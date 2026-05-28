# Finisterrae III benchmark results

This directory contains the final CESGA Finisterrae III benchmark campaigns used
for the TFM results and Python-version comparison chapters.

Source commit:

```text
6d2ef3c Group FT3 campaign rows per array task
```

Campaigns:

- `20260526_130149_full315`: main Python 3.15.0b1t free-threaded campaign.
  It contains 630 successful runs across classes S, W and A.
- `20260526_130136_versions`: Python free-threaded comparison campaign.
  It contains 960 successful runs for Python 3.13.13t, 3.14.4t and 3.15.0b1t
  across classes S and W.

Each campaign includes:

- `manifest.csv`: full execution matrix.
- `env.txt`: Slurm and campaign metadata.
- `summary.csv`: one row per completed benchmark run.
- `summary_best.csv`: grouped best and mean values.
- `records/*.json`: one structured result record per run.

The raw Slurm and benchmark logs were not included in this snapshot because the
structured records and summaries contain the data used for the analysis.
