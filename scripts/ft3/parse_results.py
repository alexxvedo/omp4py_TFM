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
    "python_version",
    "node",
    "slurm_job_id",
    "slurm_array_task_id",
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
    "best_mops",
    "mean_mops",
]


def load_records(records_dir):
    rows = []
    for path in sorted(records_dir.glob("*.json")):
        with path.open() as fh:
            rows.append(json.load(fh))
    return rows


def write_csv(path, rows, fields):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


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
        mops = [float(row["mops"]) for row in successful if row.get("mops") is not None]
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
                "mean_seconds": (sum(seconds) / len(seconds)) if seconds else None,
                "best_mops": max(mops) if mops else None,
                "mean_mops": (sum(mops) / len(mops)) if mops else None,
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Parse FT3 benchmark records")
    parser.add_argument("campaign_dir")
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    records = load_records(campaign_dir / "records")
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

    write_csv(campaign_dir / "summary.csv", records, SUMMARY_FIELDS)
    write_csv(campaign_dir / "summary_best.csv", summarize(records), BEST_FIELDS)
    print("wrote %d records" % len(records))
    print(campaign_dir / "summary.csv")
    print(campaign_dir / "summary_best.csv")


if __name__ == "__main__":
    main()
