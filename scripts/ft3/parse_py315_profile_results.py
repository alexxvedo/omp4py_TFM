#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import pstats
import re
import statistics
from pathlib import Path


SUMMARY_FIELDS = [
    "profiler",
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
    "rate",
    "python_version",
    "node",
    "output_path",
    "raw_log",
]

SAMPLING_FIELDS = [
    "benchmark",
    "class",
    "mode",
    "threads",
    "total_samples",
    "pct_compute",
    "pct_sync",
    "pct_init",
    "pct_other",
    "sync_of_parallel",
    "n_workers",
    "worker_cv",
    "worker_balance",
]

TRACING_FIELDS = [
    "benchmark",
    "class",
    "mode",
    "threads",
    "regions",
    "threads_created",
    "barriers",
    "sync_barriers",
    "omp_parallel_calls",
    "parallel_body_calls",
    "pstats_path",
]


def load_records(campaign_dir):
    records = []
    for path in sorted((campaign_dir / "records").glob("*.json")):
        with path.open() as fh:
            row = json.load(fh)
        row["_record_path"] = str(path)
        records.append(row)
    return records


def write_csv(path, rows, fields):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def categorize(leaf):
    if any(s in leaf for s in (
        "Condition.wait",
        "Condition._acquire",
        "Condition.acquire",
        "_acquire_restore",
        "sync_barrier",
        "task_barrier",
        "task_wait",
        "barrier.py",
        "lock.py",
        "Lock.acquire",
        "Semaphore",
        "Event.wait",
    )):
        return "sync"
    if "threading.py:" in leaf and ("wait" in leaf or "acquire" in leaf):
        return "sync"
    if "__omp_parallel" in leaf:
        return "compute"
    if any(s in leaf for s in (
        "<module>",
        "importlib",
        "_bootstrap",
        "<frozen",
        "_compile_bytecode",
        "runpy",
        "<GC>",
        "makea",
        "sparse",
        "FastMachine",
        "thread.py:init",
        "set_omp",
        "npbparams",
        "randlc",
        "icnvrt",
        "vecset",
    )):
        return "init"
    if re.search(r"_Python\.py:", leaf):
        return "compute"
    return "other"


def parse_sampling_collapsed(path):
    cat = {"compute": 0, "sync": 0, "init": 0, "other": 0}
    per_tid = {}
    per_tid_compute = {}
    main_hits = {}

    with path.open(errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                stack, cnt = line.rsplit(" ", 1)
                n = int(cnt)
            except ValueError:
                continue
            frames = stack.split(";")
            tid = frames[0]
            leaf = frames[-1]
            category = categorize(leaf)
            cat[category] += n
            per_tid[tid] = per_tid.get(tid, 0) + n
            if category == "compute":
                per_tid_compute[tid] = per_tid_compute.get(tid, 0) + n
            if "_run_module_as_main" in stack:
                main_hits[tid] = main_hits.get(tid, 0) + n

    main_tid = max(main_hits, key=main_hits.get) if main_hits else None
    total = sum(cat.values()) or 1
    parallel = cat["sync"] + cat["compute"]
    sync_parallel = cat["sync"] / parallel if parallel else 0.0

    workers = {tid: per_tid_compute.get(tid, 0) for tid in per_tid if tid != main_tid}
    worker_values = [value for value in workers.values() if value > 0]
    if len(worker_values) >= 2:
        mean = statistics.mean(worker_values)
        cv = statistics.pstdev(worker_values) / mean if mean else 0.0
        balance = min(worker_values) / max(worker_values) if max(worker_values) else 0.0
    else:
        cv, balance = 0.0, 1.0

    return {
        "total_samples": total,
        "pct_compute": 100 * cat["compute"] / total,
        "pct_sync": 100 * cat["sync"] / total,
        "pct_init": 100 * cat["init"] / total,
        "pct_other": 100 * cat["other"] / total,
        "sync_of_parallel": 100 * sync_parallel,
        "n_workers": len(worker_values),
        "worker_cv": cv,
        "worker_balance": balance,
    }


def stat_value(stats, name, filename_contains=None):
    total = 0
    for func, stat in stats.stats.items():
        _cc, nc, _tt, _ct, _callers = stat
        filename, _line, func_name = func
        if func_name != name:
            continue
        if filename_contains and filename_contains not in filename:
            continue
        total += int(nc)
    return total


def stat_contains(stats, name_contains, filename_contains=None):
    total = 0
    for func, stat in stats.stats.items():
        _cc, nc, _tt, _ct, _callers = stat
        filename, _line, func_name = func
        if name_contains not in func_name:
            continue
        if filename_contains and filename_contains not in filename:
            continue
        total += int(nc)
    return total


def parse_tracing_pstats(path):
    stats = pstats.Stats(str(path))
    regions = stat_value(stats, "parallel_run", "omp4py/runtime/parallelism.py")
    if not regions:
        regions = stat_value(stats, "omp_parallel", "omp4py/runtime/parallelism.py")
    return {
        "regions": regions,
        "threads_created": stat_contains(stats, "start_joinable_thread"),
        "barriers": stat_value(stats, "task_barrier", "omp4py/runtime/common/barrier.py"),
        "sync_barriers": stat_value(stats, "sync_barrier", "omp4py/runtime/synchronization.py"),
        "omp_parallel_calls": stat_value(stats, "omp_parallel", "omp4py/runtime/parallelism.py"),
        "parallel_body_calls": stat_contains(stats, "__omp_parallel"),
        "pstats_path": str(path),
    }


def is_ok(row):
    return (
        row.get("returncode") == 0
        and not row.get("timed_out")
        and row.get("verification") in ("SUCCESSFUL", "NOT PERFORMED")
        and row.get("output_path")
        and Path(row["output_path"]).exists()
    )


def add_metadata(metrics, row):
    metrics.update(
        {
            "benchmark": row.get("benchmark"),
            "class": row.get("class"),
            "mode": row.get("mode"),
            "threads": row.get("threads"),
        }
    )
    return metrics


def fmt_pct(value):
    if value is None:
        return "---"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "---"
    if math.isnan(value) or math.isinf(value):
        return "---"
    return "%.1f\\%%" % value


def write_latex_tables(campaign_dir, sampling_rows, tracing_rows):
    table_dir = campaign_dir / "latex"
    table_dir.mkdir(parents=True, exist_ok=True)

    by_sampling = {(row["benchmark"], int(row["threads"])): row for row in sampling_rows}
    threads = [1, 4, 8, 16]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Benchmark & 1 hilo & 4 hilos & 8 hilos & 16 hilos \\",
        r"\hline",
    ]
    for bench in ["EP", "FT", "CG", "IS", "MG"]:
        cells = [fmt_pct((by_sampling.get((bench, thread)) or {}).get("sync_of_parallel")) for thread in threads]
        lines.append("%s & %s \\\\" % (bench, " & ".join(cells)))
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Coste de sincronizacion medido con \texttt{profiling.sampling} en FT3, Python~3.15t, clase~S, modo~1.}",
        r"\label{tab:profiling-py315-sync-ft3}",
        r"\end{table}",
    ]
    (table_dir / "profiling_py315_sync_ft3.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Benchmark & Regiones & Hilos creados & Barreras \\",
        r"\hline",
    ]
    for row in sorted(tracing_rows, key=lambda r: ["EP", "FT", "CG", "IS", "MG"].index(r["benchmark"])):
        lines.append(
            "%s & %s & %s & %s \\\\" % (
                row["benchmark"],
                row.get("regions", "---"),
                row.get("threads_created", "---"),
                row.get("barriers", "---"),
            )
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Recuentos exactos con \texttt{profiling.tracing} en FT3, Python~3.15t, clase~S, modo~1 y 8~hilos.}",
        r"\label{tab:profiling-py315-trace-ft3}",
        r"\end{table}",
    ]
    (table_dir / "profiling_py315_trace_ft3.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parse Python 3.15 profiler campaign results")
    parser.add_argument("campaign_dir")
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    records = load_records(campaign_dir)
    records.sort(
        key=lambda row: (
            row.get("profiler") or "",
            row.get("benchmark") or "",
            int(row.get("threads") or 0),
            int(row.get("rep") or 0),
        )
    )
    write_csv(campaign_dir / "py315_profile_summary.csv", records, SUMMARY_FIELDS)

    sampling_rows = []
    tracing_rows = []
    for row in records:
        if not is_ok(row):
            continue
        output_path = Path(row["output_path"])
        if row.get("profiler") == "sampling":
            sampling_rows.append(add_metadata(parse_sampling_collapsed(output_path), row))
        elif row.get("profiler") == "tracing":
            tracing_rows.append(add_metadata(parse_tracing_pstats(output_path), row))

    sampling_rows.sort(key=lambda row: (row["benchmark"], int(row["threads"])))
    tracing_rows.sort(key=lambda row: (row["benchmark"], int(row["threads"])))
    write_csv(campaign_dir / "py315_sampling_summary.csv", sampling_rows, SAMPLING_FIELDS)
    write_csv(campaign_dir / "py315_tracing_summary.csv", tracing_rows, TRACING_FIELDS)
    write_latex_tables(campaign_dir, sampling_rows, tracing_rows)

    print("records:", len(records))
    print("sampling parsed:", len(sampling_rows))
    print("tracing parsed:", len(tracing_rows))
    print(campaign_dir / "py315_profile_summary.csv")
    print(campaign_dir / "py315_sampling_summary.csv")
    print(campaign_dir / "py315_tracing_summary.csv")
    print(campaign_dir / "latex")


if __name__ == "__main__":
    main()
