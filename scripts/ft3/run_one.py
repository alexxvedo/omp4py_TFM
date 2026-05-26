#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PYTHON_PREFIX_ENV = {
    "3.13t": "PYTHON_313T_PREFIX",
    "3.14t": "PYTHON_314T_PREFIX",
    "3.15t": "PYTHON_315T_PREFIX",
}

PYTHON_PREFIX_DEFAULT = {
    "3.13t": "~/opt/python/3.13.13t",
    "3.14t": "~/opt/python/3.14.4t",
    "3.15t": "~/opt/python/3.15.0b1t",
}


def repo_root():
    return Path(os.environ.get("FT3_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def python_prefix(tag):
    if tag not in PYTHON_PREFIX_ENV:
        raise ValueError("unknown Python tag: %s" % tag)
    return Path(os.environ.get(PYTHON_PREFIX_ENV[tag], PYTHON_PREFIX_DEFAULT[tag])).expanduser()


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
    return "py%s__%s__%s__m%s__t%02d__r%02d" % (
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
    timeout = int(row.get("timeout") or 3600)

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
    env["PYTHONPATH"] = "%s%s%s" % (
        root,
        ":" if env.get("PYTHONPATH") else "",
        env.get("PYTHONPATH", ""),
    )
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    env["OMP4PY_CACHE"] = "1"
    env["OMP4PY_CACHE_DIR"] = str(campaign_dir / "cache" / tag)

    command = [
        str(python_exe),
        str(root / "examples" / ("%s_Python.py" % bench)),
        "-c",
        class_name,
        "-m",
        str(mode),
        "-t",
        str(threads),
    ]

    name = safe_name(row)
    raw_path = campaign_dir / "raw" / ("%s.txt" % name)
    record_path = campaign_dir / "records" / ("%s.json" % name)

    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        timed_out = False
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

    ended = time.time()
    combined = stdout + "\n" + stderr
    parsed = parse_output(combined)

    raw_content = [
        "command = %s" % " ".join(command),
        "cwd = %s" % root,
        "slurm_job_id = %s" % os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id = %s" % os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "wall_seconds = %.6f" % (ended - started),
        "",
        "----- stdout -----",
        stdout,
        "----- stderr -----",
        stderr,
    ]
    write_text(raw_path, "\n".join(raw_content))

    record = {
        "python_tag": tag,
        "benchmark": bench,
        "class": class_name,
        "mode": mode,
        "threads": threads,
        "rep": int(row["rep"]),
        "timeout": timeout,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": ended - started,
        "npb_seconds": parsed["seconds"],
        "mops": parsed["mops"],
        "verification": parsed["verification"],
        "python_version": parsed["python_version"],
        "raw_log": str(raw_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    write_json(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description="Run one FT3 benchmark manifest row")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True, help="1-based row index excluding CSV header")
    parser.add_argument("--campaign-dir", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    row = read_manifest_row(args.manifest, args.index)
    record = run(row, campaign_dir)
    print(json.dumps(record, sort_keys=True))
    if record["returncode"] not in (0, None):
        sys.exit(record["returncode"])
    if record["timed_out"]:
        sys.exit(124)


if __name__ == "__main__":
    main()
