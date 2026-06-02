#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SUMMARY_FIELDS = [
    "python_tag",
    "benchmark",
    "class",
    "mode",
    "threads",
    "rep",
    "returncode",
    "timed_out",
    "verification",
    "npb_seconds",
    "mops",
    "wall_seconds",
    "warmup_wall_seconds",
    "perf_task_clock_ms",
    "perf_cpus_utilized",
    "perf_cycles",
    "perf_instructions",
    "perf_ipc",
    "perf_cache_references",
    "perf_cache_misses",
    "perf_cache_miss_ratio",
    "perf_context_switches",
    "perf_cpu_migrations",
    "python_version",
    "node",
    "raw_log",
]

BEST_FIELDS = [
    "python_tag",
    "benchmark",
    "class",
    "mode",
    "threads",
    "runs",
    "successful_runs",
    "best_seconds",
    "mean_seconds",
    "best_cpus_utilized",
    "mean_cpus_utilized",
    "best_ipc",
    "mean_ipc",
    "best_cache_miss_ratio",
    "mean_cache_miss_ratio",
]


def load_records(records_dir):
    rows = []
    for path in sorted(records_dir.glob("*.json")):
        with path.open() as fh:
            rows.append(json.load(fh))
    return rows


def flatten(row):
    perf = row.get("perf") or {}
    out = dict(row)
    out["perf_task_clock_ms"] = perf.get("task-clock")
    out["perf_cpus_utilized"] = perf.get("cpus_utilized")
    out["perf_cycles"] = perf.get("cycles")
    out["perf_instructions"] = perf.get("instructions")
    out["perf_ipc"] = perf.get("ipc")
    out["perf_cache_references"] = perf.get("cache-references")
    out["perf_cache_misses"] = perf.get("cache-misses")
    out["perf_cache_miss_ratio"] = perf.get("cache_miss_ratio")
    out["perf_context_switches"] = perf.get("context-switches")
    out["perf_cpu_migrations"] = perf.get("cpu-migrations")
    return out


def write_csv(path, rows, fields):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row.get("python_tag"),
            row.get("benchmark"),
            row.get("class"),
            int(row.get("mode")),
            int(row.get("threads")),
        )
        groups[key].append(row)

    summary = []
    for key, values in sorted(groups.items()):
        successful = [
            row
            for row in values
            if row.get("returncode") == 0
            and not row.get("timed_out")
            and row.get("npb_seconds") is not None
            and row.get("verification") in ("SUCCESSFUL", "NOT PERFORMED")
        ]
        seconds = [float(row["npb_seconds"]) for row in successful]
        cpus = [float(row["perf_cpus_utilized"]) for row in successful if row.get("perf_cpus_utilized") is not None]
        ipc = [float(row["perf_ipc"]) for row in successful if row.get("perf_ipc") is not None]
        cache = [
            float(row["perf_cache_miss_ratio"])
            for row in successful
            if row.get("perf_cache_miss_ratio") is not None
        ]
        summary.append(
            {
                "python_tag": key[0],
                "benchmark": key[1],
                "class": key[2],
                "mode": key[3],
                "threads": key[4],
                "runs": len(values),
                "successful_runs": len(successful),
                "best_seconds": min(seconds) if seconds else None,
                "mean_seconds": mean(seconds),
                "best_cpus_utilized": max(cpus) if cpus else None,
                "mean_cpus_utilized": mean(cpus),
                "best_ipc": max(ipc) if ipc else None,
                "mean_ipc": mean(ipc),
                "best_cache_miss_ratio": min(cache) if cache else None,
                "mean_cache_miss_ratio": mean(cache),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Parse FT3 profiling records")
    parser.add_argument("campaign_dir")
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    records = [flatten(row) for row in load_records(campaign_dir / "records")]
    records.sort(
        key=lambda row: (
            row.get("python_tag") or "",
            row.get("benchmark") or "",
            row.get("class") or "",
            int(row.get("mode") or 0),
            int(row.get("threads") or 0),
            int(row.get("rep") or 0),
        )
    )

    write_csv(campaign_dir / "profile_summary.csv", records, SUMMARY_FIELDS)
    write_csv(campaign_dir / "profile_summary_best.csv", summarize(records), BEST_FIELDS)
    print("wrote %d profile records" % len(records))
    print(campaign_dir / "profile_summary.csv")
    print(campaign_dir / "profile_summary_best.csv")


if __name__ == "__main__":
    main()
