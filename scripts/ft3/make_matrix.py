#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


BENCHMARKS = ["CG", "EP", "FT", "IS", "MG"]
THREADS_FULL = [1, 2, 4, 8, 16, 32]
THREADS_COMPARE = [1, 4, 16, 32]
LONG_TIMEOUT = 259000


def add_rows(rows, python_tags, benchmarks, classes, modes, threads, reps, timeout):
    for python_tag in python_tags:
        for benchmark in benchmarks:
            for class_name in classes:
                for mode in modes:
                    for thread_count in threads:
                        for rep in range(1, reps + 1):
                            rows.append(
                                {
                                    "python_tag": python_tag,
                                    "benchmark": benchmark,
                                    "class": class_name,
                                    "mode": mode,
                                    "threads": thread_count,
                                    "rep": rep,
                                    "timeout": timeout,
                                }
                            )


def add_rows_interleaved_python(rows, python_tags, benchmarks, classes, modes, threads, reps, timeout):
    for benchmark in benchmarks:
        for class_name in classes:
            for mode in modes:
                for thread_count in threads:
                    for rep in range(1, reps + 1):
                        for python_tag in python_tags:
                            rows.append(
                                {
                                    "python_tag": python_tag,
                                    "benchmark": benchmark,
                                    "class": class_name,
                                    "mode": mode,
                                    "threads": thread_count,
                                    "rep": rep,
                                    "timeout": timeout,
                                }
                            )


def build_profile(profile):
    rows = []

    if profile == "pilot":
        add_rows(rows, ["3.15t"], BENCHMARKS, ["S"], [2, 3], [1, 4], 1, 1800)
    elif profile == "full315":
        add_rows(rows, ["3.15t"], BENCHMARKS, ["S"], [0, 1, 2, 3], THREADS_FULL, 3, LONG_TIMEOUT)
        add_rows(rows, ["3.15t"], BENCHMARKS, ["W"], [2, 3], THREADS_FULL, 3, LONG_TIMEOUT)
        add_rows(rows, ["3.15t"], BENCHMARKS, ["A"], [3], THREADS_FULL, 3, LONG_TIMEOUT)
    elif profile == "versions":
        tags = ["3.13t", "3.14t", "3.15t"]
        add_rows_interleaved_python(rows, tags, BENCHMARKS, ["S"], [2, 3], THREADS_COMPARE, 5, LONG_TIMEOUT)
        add_rows_interleaved_python(rows, tags, BENCHMARKS, ["W"], [2, 3], THREADS_COMPARE, 3, LONG_TIMEOUT)
    else:
        raise ValueError("unknown profile: %s" % profile)

    return rows


def parse_csv_ints(value):
    return [int(item) for item in value.split(",") if item]


def parse_csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate FT3 benchmark matrices")
    parser.add_argument("--profile", choices=["pilot", "full315", "versions"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python-tags", help="Comma-separated override, e.g. 3.15t")
    parser.add_argument("--benchmarks", help="Comma-separated override, e.g. CG,EP,FT,IS,MG")
    parser.add_argument("--classes", help="Comma-separated override, e.g. S,W,A")
    parser.add_argument("--modes", help="Comma-separated override, e.g. 2,3")
    parser.add_argument("--threads", help="Comma-separated override, e.g. 1,4,16,32")
    parser.add_argument("--reps", type=int, help="Override repetitions")
    parser.add_argument("--timeout", type=int, help="Override per-run timeout in seconds")
    args = parser.parse_args()

    rows = build_profile(args.profile)

    if any([args.python_tags, args.benchmarks, args.classes, args.modes, args.threads, args.reps, args.timeout]):
        python_tags = parse_csv_strings(args.python_tags) if args.python_tags else sorted({r["python_tag"] for r in rows})
        benchmarks = parse_csv_strings(args.benchmarks) if args.benchmarks else sorted({r["benchmark"] for r in rows})
        classes = parse_csv_strings(args.classes) if args.classes else sorted({r["class"] for r in rows})
        modes = parse_csv_ints(args.modes) if args.modes else sorted({int(r["mode"]) for r in rows})
        threads = parse_csv_ints(args.threads) if args.threads else sorted({int(r["threads"]) for r in rows})
        reps = args.reps if args.reps else max(int(r["rep"]) for r in rows)
        timeout = args.timeout if args.timeout else max(int(r["timeout"]) for r in rows)
        rows = []
        add_rows(rows, python_tags, benchmarks, classes, modes, threads, reps, timeout)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        fieldnames = ["python_tag", "benchmark", "class", "mode", "threads", "rep", "timeout"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("wrote %d runs to %s" % (len(rows), output))


if __name__ == "__main__":
    main()
