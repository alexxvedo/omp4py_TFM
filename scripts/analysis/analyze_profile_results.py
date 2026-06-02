#!/usr/bin/env python3
"""Summarize FT3 profiling records for the TFM diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


THREADS_BY_BENCH = {
    "EP": [1, 8, 16, 32],
    "FT": [1, 8, 16, 32],
    "CG": [1, 8, 32],
    "IS": [1, 8, 32],
    "MG": [1, 8, 32],
}


def load_records(campaign: Path) -> list[dict]:
    records = []
    for path in sorted((campaign / "records").glob("*.json")):
        with path.open() as fh:
            row = json.load(fh)
        row["_path"] = str(path)
        records.append(row)
    return records


def best_by_key(records: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        key = (row["python_tag"], row["benchmark"], row["class"], int(row["mode"]), int(row["threads"]))
        groups[key].append(row)
    out = {}
    for key, rows in groups.items():
        ok = [
            row
            for row in rows
            if row.get("returncode") == 0
            and not row.get("timed_out")
            and row.get("verification") == "SUCCESSFUL"
            and row.get("npb_seconds") is not None
        ]
        if ok:
            out[key] = min(ok, key=lambda row: float(row["npb_seconds"]))
    return out


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "---"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "---"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def perf(row: dict, key: str) -> float | None:
    value = (row.get("perf") or {}).get(key)
    return None if value is None else float(value)


def speedup(summary: dict, py: str, bench: str, cls: str, mode: int, threads: int) -> float | None:
    base = summary.get((py, bench, cls, mode, 1))
    row = summary.get((py, bench, cls, mode, threads))
    if not base or not row:
        return None
    t1 = float(base["npb_seconds"])
    tt = float(row["npb_seconds"])
    return None if tt <= 0 else t1 / tt


def row_line(summary: dict, py: str, bench: str, threads: int) -> str:
    row = summary.get((py, bench, "W", 3, threads))
    if not row:
        return f"| {bench} | {py} | {threads} | --- | --- | --- | --- | --- | --- |"
    cpus = perf(row, "cpus_utilized")
    ipc = perf(row, "ipc")
    cache = perf(row, "cache_miss_ratio")
    cpu_eff = None if cpus is None or threads <= 0 else cpus / threads
    return (
        f"| {bench} | {py} | {threads} | {fmt(float(row['npb_seconds']))} | "
        f"{fmt(speedup(summary, py, bench, 'W', 3, threads))} | {fmt(cpus)} | "
        f"{fmt(cpu_eff)} | {fmt(ipc)} | {fmt(None if cache is None else cache * 100)} |"
    )


def make_markdown(records: list[dict]) -> str:
    summary = best_by_key(records)
    ok = sum(
        row.get("returncode") == 0
        and not row.get("timed_out")
        and row.get("verification") == "SUCCESSFUL"
        for row in records
    )
    lines = [
        "# Profiling summary",
        "",
        f"Records: {len(records)}",
        f"Successful: {ok}",
        "",
        "## EP/FT: 3.14t vs 3.15t",
        "",
        "| Bench. | Python | Hilos | NAS (s) | Speedup | CPUs usadas | CPUs/hilo | IPC | Cache miss % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bench in ["EP", "FT"]:
        for py in ["3.14t", "3.15t"]:
            for threads in THREADS_BY_BENCH[bench]:
                lines.append(row_line(summary, py, bench, threads))

    lines.extend(
        [
            "",
            "## Benchmarks sin escalabilidad clara",
            "",
            "| Bench. | Python | Hilos | NAS (s) | Speedup | CPUs usadas | CPUs/hilo | IPC | Cache miss % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bench in ["CG", "IS", "MG"]:
        for py in ["3.13t", "3.14t", "3.15t"]:
            for threads in THREADS_BY_BENCH[bench]:
                lines.append(row_line(summary, py, bench, threads))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FT3 profiling records")
    parser.add_argument("campaign_dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    campaign = Path(args.campaign_dir).expanduser().resolve()
    markdown = make_markdown(load_records(campaign))
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
