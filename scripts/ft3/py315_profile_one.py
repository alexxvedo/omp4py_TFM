#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path


PYTHON_PREFIX_DEFAULT = {
    "3.15t": "~/opt/python/3.15.0b1t",
}


def repo_root():
    return Path(os.environ.get("FT3_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def python_prefix(tag):
    return Path(os.environ.get("PYTHON_315T_PREFIX", PYTHON_PREFIX_DEFAULT[tag])).expanduser()


def venv_python(root, tag):
    return root / (".venv-%s" % tag) / "bin" / "python"


def read_manifest_row(manifest, index):
    with open(manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        for row_number, row in enumerate(reader, start=1):
            if row_number == index:
                return row
    raise IndexError("manifest index out of range: %d" % index)


def safe_name(row):
    return "%s__py%s__%s__%s__m%s__t%02d__r%02d" % (
        row["profiler"],
        row["python_tag"].replace(".", ""),
        row["benchmark"],
        row["class"],
        row["mode"],
        int(row["threads"]),
        int(row["rep"]),
    )


def parse_output(text):
    patterns = {
        "seconds": r"Time in seconds\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        "mops": r"Mop/s total\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        "verification": r"Verification\s*=\s*([A-Z ]+)",
        "python_version": r"Python Version\s*=\s*(.+)",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        parsed[key] = match.group(1).strip() if match else None
    for key in ("seconds", "mops"):
        if parsed[key] is not None:
            parsed[key] = float(parsed[key])
    return parsed


def run_command(command, env, cwd, timeout):
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return {
            "timed_out": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "wall_seconds": time.time() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "timed_out": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "wall_seconds": time.time() - started,
        }


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(content)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def run(row, campaign_dir):
    root = repo_root()
    tag = row["python_tag"]
    bench = row["benchmark"]
    class_name = row["class"]
    mode = int(row["mode"])
    threads = int(row["threads"])
    timeout = int(row.get("timeout") or 7200)
    rate = int(row.get("rate") or 2000)
    profiler = row["profiler"]
    name = safe_name(row)

    python_exe = venv_python(root, tag)
    if not python_exe.exists():
        raise FileNotFoundError("missing venv Python: %s" % python_exe)

    prefix = python_prefix(tag)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "%s%s%s" % (
        prefix / "lib",
        ":" if env.get("LD_LIBRARY_PATH") else "",
        env.get("LD_LIBRARY_PATH", ""),
    )
    env["PYTHON_GIL"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "%s:%s%s%s" % (
        root,
        root / "examples",
        ":" if env.get("PYTHONPATH") else "",
        env.get("PYTHONPATH", ""),
    )
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env.pop("OMP_PROC_BIND", None)
    env.pop("OMP_PLACES", None)
    env.pop("GOMP_CPU_AFFINITY", None)

    benchmark_cmd = [
        str(root / "examples" / ("%s_Python.py" % bench)),
        "-c",
        class_name,
        "-m",
        str(mode),
        "-t",
        str(threads),
    ]

    raw_path = campaign_dir / "raw" / ("%s.txt" % name)
    record_path = campaign_dir / "records" / ("%s.json" % name)
    out_dir = campaign_dir / profiler
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = None
    if profiler == "sampling":
        output_path = out_dir / ("%s.collapsed" % name)
        command = [
            str(python_exe),
            "-m",
            "profiling.sampling",
            "run",
            "-a",
            "--mode",
            "wall",
            "-r",
            str(rate),
            "--collapsed",
            "-o",
            str(output_path),
        ] + benchmark_cmd
    elif profiler == "tracing":
        output_path = out_dir / ("%s.pstats" % name)
        command = [
            str(python_exe),
            "-m",
            "profiling.tracing",
            "-o",
            str(output_path),
        ] + benchmark_cmd
    else:
        raise ValueError("unknown profiler: %s" % profiler)

    measured = run_command(command, env, root, timeout)
    combined = measured["stdout"] + "\n" + measured["stderr"]
    parsed = parse_output(combined)

    raw_content = [
        "command = %s" % " ".join(command),
        "cwd = %s" % root,
        "slurm_job_id = %s" % os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id = %s" % os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "returncode = %s" % measured["returncode"],
        "timed_out = %s" % measured["timed_out"],
        "wall_seconds = %.6f" % measured["wall_seconds"],
        "output_path = %s" % output_path,
        "",
        "----- stdout -----",
        measured["stdout"],
        "",
        "----- stderr -----",
        measured["stderr"],
    ]
    write_text(raw_path, "\n".join(raw_content))

    payload = {
        "profiler": profiler,
        "python_tag": tag,
        "benchmark": bench,
        "class": class_name,
        "mode": mode,
        "threads": threads,
        "rep": int(row["rep"]),
        "rate": rate,
        "returncode": measured["returncode"],
        "timed_out": measured["timed_out"],
        "wall_seconds": measured["wall_seconds"],
        "npb_seconds": parsed.get("seconds"),
        "mops": parsed.get("mops"),
        "verification": parsed.get("verification"),
        "python_version": parsed.get("python_version"),
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "output_path": str(output_path),
        "raw_log": str(raw_path),
    }
    write_json(record_path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run one Python 3.15 profiling row")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--campaign-dir", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    row = read_manifest_row(args.manifest, args.index)
    payload = run(row, campaign_dir)
    print("%s %s %s t%s rc=%s timeout=%s verification=%s wall=%.2fs" % (
        payload["profiler"],
        payload["benchmark"],
        payload["class"],
        payload["threads"],
        payload["returncode"],
        payload["timed_out"],
        payload["verification"],
        payload["wall_seconds"],
    ))


if __name__ == "__main__":
    main()
