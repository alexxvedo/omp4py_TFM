# Finisterrae III benchmark results (superseded snapshot)

This directory contains an earlier CESGA Finisterrae III benchmark snapshot. It
is kept for traceability, but it is not the final source used by the current TFM
memory. The final campaigns are documented in `../README.md` and stored under
`../ft3_20260601`.

Source commit:

```text
6d2ef3c Group FT3 campaign rows per array task
```

Historical campaigns:

- `20260526_130149_full315`: earlier Python 3.15.0b1t free-threaded campaign.
  It contains 630 successful runs across classes S, W and A.
- `20260526_130136_versions`: earlier Python free-threaded comparison campaign.
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
