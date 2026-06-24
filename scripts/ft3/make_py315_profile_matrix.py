#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SAMPLING_BENCHMARKS = ["EP", "FT", "CG", "IS", "MG"]
TRACING_BENCHMARKS = ["EP", "FT", "CG", "IS", "MG"]


def add_sampling(rows, benchmarks, threads, reps, timeout, rate):
    for benchmark in benchmarks:
        for thread_count in threads:
            for rep in range(1, reps + 1):
                rows.append(
                    {
                        "profiler": "sampling",
                        "python_tag": "3.15t",
                        "benchmark": benchmark,
                        "class": "S",
                        "mode": 1,
                        "threads": thread_count,
                        "rep": rep,
                        "timeout": timeout,
                        "rate": rate,
                    }
                )


def add_tracing(rows, benchmarks, threads, reps, timeout, rate):
    for benchmark in benchmarks:
        for thread_count in threads:
            for rep in range(1, reps + 1):
                rows.append(
                    {
                        "profiler": "tracing",
                        "python_tag": "3.15t",
                        "benchmark": benchmark,
                        "class": "S",
                        "mode": 1,
                        "threads": thread_count,
                        "rep": rep,
                        "timeout": timeout,
                        "rate": rate,
                    }
                )


def build_profile(profile):
    rows = []
    if profile in ("py315_sampling", "py315_both"):
        add_sampling(rows, SAMPLING_BENCHMARKS, [1, 4, 8, 16], 1, 7200, 2000)
    if profile in ("py315_tracing", "py315_both"):
        add_tracing(rows, TRACING_BENCHMARKS, [8], 1, 21600, 2000)
    if profile not in ("py315_sampling", "py315_tracing", "py315_both"):
        raise ValueError("unknown profile: %s" % profile)
    return rows


def parse_csv_ints(value):
    return [int(item) for item in value.split(",") if item]


def parse_csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate Python 3.15 profiler matrices for FT3")
    parser.add_argument(
        "--profile",
        choices=["py315_sampling", "py315_tracing", "py315_both"],
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmarks", help="Comma-separated override, e.g. CG,EP,FT,IS,MG")
    parser.add_argument("--threads", help="Comma-separated override, e.g. 1,4,8,16")
    parser.add_argument("--reps", type=int, help="Override repetitions")
    parser.add_argument("--timeout", type=int, help="Override per-run timeout in seconds")
    parser.add_argument("--rate", type=int, help="Override sampling rate")
    args = parser.parse_args()

    rows = build_profile(args.profile)

    if any([args.benchmarks, args.threads, args.reps, args.timeout, args.rate]):
        profilers = sorted({r["profiler"] for r in rows})
        benchmarks = parse_csv_strings(args.benchmarks) if args.benchmarks else sorted({r["benchmark"] for r in rows})
        threads = parse_csv_ints(args.threads) if args.threads else sorted({int(r["threads"]) for r in rows})
        reps = args.reps if args.reps else max(int(r["rep"]) for r in rows)
        timeout = args.timeout if args.timeout else max(int(r["timeout"]) for r in rows)
        rate = args.rate if args.rate else max(int(r["rate"]) for r in rows)
        rows = []
        if "sampling" in profilers:
            add_sampling(rows, benchmarks, threads, reps, timeout, rate)
        if "tracing" in profilers:
            add_tracing(rows, benchmarks, threads, reps, timeout, rate)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        fieldnames = [
            "profiler",
            "python_tag",
            "benchmark",
            "class",
            "mode",
            "threads",
            "rep",
            "timeout",
            "rate",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("wrote %d runs to %s" % (len(rows), output))


if __name__ == "__main__":
    main()
