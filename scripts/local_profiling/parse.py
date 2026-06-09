#!/usr/bin/env python3
"""Parse profiling.sampling collapsed stacks into synchronization/compute/init
categories and per-thread load balance, for the objective-5 study."""
import os, glob, re, csv, statistics

OUT = os.path.expanduser("~/prof_local")


def categorize(leaf):
    l = leaf
    # Synchronization: thread blocked at a barrier / reduction / lock wait.
    if any(s in l for s in (
        "Condition.wait", "Condition._acquire", "Condition.acquire",
        "_acquire_restore", "sync_barrier", "task_barrier", "task_wait",
        "barrier.py", "lock.py", "Lock.acquire", "Semaphore", "Event.wait",
    )):
        return "sync"
    if "threading.py:" in l and ("wait" in l or "acquire" in l):
        return "sync"
    # Compute: parallelized region bodies and benchmark kernels.
    if "__omp_parallel" in l:
        return "compute"
    # Init / setup / import / GC.
    if any(s in l for s in (
        "<module>", "importlib", "_bootstrap", "<frozen", "_compile_bytecode",
        "runpy", "<GC>", "makea", "sparse", "FastMachine", "thread.py:init",
        "set_omp", "npbparams", "randlc", "icnvrt", "vecset",
    )):
        return "init"
    # Remaining benchmark-file leaves are kernel helpers -> compute.
    if re.search(r"_Python\.py:", l):
        return "compute"
    return "other"


def parse_file(path):
    cat = {"compute": 0, "sync": 0, "init": 0, "other": 0}
    per_tid = {}                 # tid -> total samples
    per_tid_compute = {}         # tid -> compute samples
    main_tid = None
    main_hits = {}
    with open(path) as fh:
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
            c = categorize(leaf)
            cat[c] += n
            per_tid[tid] = per_tid.get(tid, 0) + n
            if c == "compute":
                per_tid_compute[tid] = per_tid_compute.get(tid, 0) + n
            if "_run_module_as_main" in stack:
                main_hits[tid] = main_hits.get(tid, 0) + n
    if main_hits:
        main_tid = max(main_hits, key=main_hits.get)
    total = sum(cat.values()) or 1
    sync_parallel = cat["sync"] / (cat["sync"] + cat["compute"]) if (cat["sync"] + cat["compute"]) else 0.0

    # Load balance across worker threads (exclude main).
    workers = {t: per_tid_compute.get(t, 0) for t in per_tid if t != main_tid}
    wvals = [v for v in workers.values() if v > 0]
    if len(wvals) >= 2:
        mean = statistics.mean(wvals)
        cv = statistics.pstdev(wvals) / mean if mean else 0.0
        balance = min(wvals) / max(wvals) if max(wvals) else 0.0
    else:
        cv, balance = 0.0, 1.0
    return {
        "total": total,
        "pct_compute": 100 * cat["compute"] / total,
        "pct_sync": 100 * cat["sync"] / total,
        "pct_init": 100 * cat["init"] / total,
        "pct_other": 100 * cat["other"] / total,
        "sync_of_parallel": 100 * sync_parallel,
        "n_workers": len(wvals),
        "worker_cv": cv,
        "worker_balance": balance,
    }


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, "*.collapsed"))):
        base = os.path.basename(path)[:-len(".collapsed")]
        m = re.match(r"([A-Z]+)_([A-Z])_m(\d+)_t(\d+)", base)
        if not m:
            continue
        bench, klass, mode, threads = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        r = parse_file(path)
        r.update(benchmark=bench, klass=klass, mode=mode, threads=threads)
        rows.append(r)

    rows.sort(key=lambda r: (r["benchmark"], r["threads"]))
    cols = ["benchmark", "klass", "mode", "threads", "total", "pct_compute",
            "pct_sync", "pct_init", "pct_other", "sync_of_parallel",
            "n_workers", "worker_cv", "worker_balance"]
    csv_path = os.path.join(OUT, "prof_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    print(f"{'bench':5} {'t':>3} {'samples':>8} {'compute%':>8} {'sync%':>7} "
          f"{'init%':>6} {'sync/par%':>9} {'workers':>7} {'balance':>7}")
    for r in rows:
        print(f"{r['benchmark']:5} {r['threads']:>3} {r['total']:>8} "
              f"{r['pct_compute']:>8.1f} {r['pct_sync']:>7.1f} {r['pct_init']:>6.1f} "
              f"{r['sync_of_parallel']:>9.1f} {r['n_workers']:>7} {r['worker_balance']:>7.2f}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
