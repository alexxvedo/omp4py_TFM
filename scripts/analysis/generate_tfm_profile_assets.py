#!/usr/bin/env python3
"""Generate LaTeX tables and PNG plots for the profiling chapter."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from generate_tfm_results_assets import Canvas, COLORS, fmt_num, save_mpl, table_env, write, y_lin_map


ROOT = Path(__file__).resolve().parents[2]
TFM = Path("/home/av/Documents/Master/TFM/omp4py/memoria/Memoria_TFM_MHPC_USC")
TABLE_DIR = TFM / "contenido" / "tablas"
FIG_DIR = TFM / "imagenes" / "profiling"

DEFAULT_CAMPAIGNS = [
    ROOT / "results" / "ft3_20260601" / "20260531_202503_profile_versions_core_nopin",
]

PYS = ["3.13t", "3.14t", "3.15t"]
THREADS_EP_FT = [1, 8, 16, 32]
THREADS_OTHER = [1, 8, 32]


def fnum(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(campaign_dirs: list[Path]) -> list[dict]:
    rows = []
    for campaign in campaign_dirs:
        summary = campaign / "profile_summary.csv"
        if not summary.exists():
            continue
        with summary.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["_campaign"] = str(campaign)
                rows.append(row)
    return rows


def best_by_key(rows: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("returncode") != "0":
            continue
        if row.get("timed_out") == "True":
            continue
        if row.get("verification") != "SUCCESSFUL":
            continue
        if fnum(row.get("npb_seconds")) is None:
            continue
        key = (
            row["python_tag"],
            row["benchmark"],
            row["class"],
            int(row["mode"]),
            int(row["threads"]),
        )
        groups[key].append(row)
    return {key: min(rs, key=lambda r: fnum(r.get("npb_seconds")) or float("inf")) for key, rs in groups.items()}


def row(summary: dict, py: str, bench: str, threads: int) -> dict | None:
    return summary.get((py, bench, "W", 3, threads))


def value(summary: dict, py: str, bench: str, threads: int, column: str) -> float | None:
    r = row(summary, py, bench, threads)
    return None if r is None else fnum(r.get(column))


def speedup(summary: dict, py: str, bench: str, threads: int) -> float | None:
    base = value(summary, py, bench, 1, "npb_seconds")
    current = value(summary, py, bench, threads, "npb_seconds")
    if base is None or current is None or current <= 0:
        return None
    return base / current


def fmt_s(v: float | None) -> str:
    if v is None or math.isnan(v) or math.isinf(v):
        return "---"
    if v >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_x(v: float | None) -> str:
    if v is None or math.isnan(v) or math.isinf(v):
        return "---"
    return f"{v:.2f}$\\times$"


def fmt_pct(v: float | None) -> str:
    if v is None or math.isnan(v) or math.isinf(v):
        return "---"
    return f"{v * 100:.1f}"


def make_tables(summary: dict) -> None:
    rows = []
    for bench in ["EP", "FT"]:
        for py in ["3.14t", "3.15t"]:
            for threads in THREADS_EP_FT:
                rows.append(
                    f"{bench} & {py} & {threads} & "
                    f"{fmt_s(value(summary, py, bench, threads, 'npb_seconds'))} & "
                    f"{fmt_x(speedup(summary, py, bench, threads))} & "
                    f"{fmt_s(value(summary, py, bench, threads, 'perf_cpus_utilized'))} & "
                    f"{fmt_s(value(summary, py, bench, threads, 'perf_ipc'))} & "
                    f"{fmt_pct(value(summary, py, bench, threads, 'perf_cache_miss_ratio'))} \\\\"
                )
    write(
        TABLE_DIR / "profiling_ep_ft_perf.tex",
        table_env(
            "tab:profiling-ep-ft-perf",
            "Perfil con \\texttt{perf stat} de EP y FT en clase~W, modo~3.",
            "llrrrrrr",
            "Bench. & Python & Hilos & NAS (s) & Speedup & CPUs usadas & IPC & Fallos cache (\\%)",
            rows,
            resize=True,
        ),
    )

    rows = []
    for bench in ["CG", "IS", "MG"]:
        for py in PYS:
            for threads in THREADS_OTHER:
                if value(summary, py, bench, threads, "npb_seconds") is None:
                    continue
                rows.append(
                    f"{bench} & {py} & {threads} & "
                    f"{fmt_s(value(summary, py, bench, threads, 'npb_seconds'))} & "
                    f"{fmt_x(speedup(summary, py, bench, threads))} & "
                    f"{fmt_s(value(summary, py, bench, threads, 'perf_cpus_utilized'))} & "
                    f"{fmt_s(value(summary, py, bench, threads, 'perf_ipc'))} & "
                    f"{fmt_pct(value(summary, py, bench, threads, 'perf_cache_miss_ratio'))} \\\\"
                )
    write(
        TABLE_DIR / "profiling_cg_is_mg_perf.tex",
        table_env(
            "tab:profiling-cg-is-mg-perf",
            "Perfil con \\texttt{perf stat} de CG, IS y MG en clase~W, modo~3.",
            "llrrrrrr",
            "Bench. & Python & Hilos & NAS (s) & Speedup & CPUs usadas & IPC & Fallos cache (\\%)",
            rows,
            resize=True,
        ),
    )


def line_plot(path: Path, title: str, xlabels: list[str], series: list[tuple[str, list[float | None]]], ylabel: str) -> None:
    c = Canvas(1200, 720)
    left, right, top, bottom = 105, 990, 85, 590
    c.text(90, 26, title, scale=3)
    vals = [v for _, ys in series for v in ys if v is not None]
    if not vals:
        return
    ymax = max(1.1, max(vals) * 1.15)
    c.line(left, bottom, right, bottom, (40, 40, 40), 2)
    c.line(left, top, left, bottom, (40, 40, 40), 2)
    for tick in [0, 1, 2, 4, 8, 12, 16]:
        if tick > ymax:
            continue
        y = y_lin_map(tick, 0, ymax, top, bottom)
        c.line(left - 5, y, right, y, (224, 224, 224), 1)
        c.text(25, int(y - 10), fmt_num(tick), scale=2)
    for i, label in enumerate(xlabels):
        x = left + i * (right - left) / (len(xlabels) - 1)
        c.line(x, bottom, x, bottom + 8, (40, 40, 40), 2)
        c.text(int(x - 10), bottom + 22, label, scale=2)
    custom_colors = {
        "EP314": (200, 75, 55),
        "EP315": (110, 120, 130),
        "FT314": (70, 145, 80),
        "FT315": (45, 95, 160),
        "CG": COLORS["CG"],
        "IS": COLORS["IS"],
        "MG": COLORS["MG"],
    }
    for name, ys in series:
        pts = []
        for i, v in enumerate(ys):
            if v is None:
                continue
            x = left + i * (right - left) / (len(xlabels) - 1)
            y = y_lin_map(v, 0, ymax, top, bottom)
            pts.append((x, y))
        color = custom_colors.get(name, COLORS.get(name, (0, 0, 0)))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            c.line(x0, y0, x1, y1, color, 4)
        for x, y in pts:
            c.circle(round(x), round(y), 6, color)
    lx, ly = 1015, 95
    for name, _ in series:
        color = custom_colors.get(name, COLORS.get(name, (0, 0, 0)))
        c.rect(lx, ly + 4, lx + 18, ly + 18, color)
        c.text(lx + 28, ly, name, scale=2)
        ly += 34
    c.text(20, 625, ylabel, scale=2)
    c.write_png(path)


def make_figures(summary: dict) -> None:
    if make_figures_matplotlib(summary):
        return

    def pct(v: float | None) -> float | None:
        return None if v is None else v * 100.0

    xlabels = [str(t) for t in THREADS_EP_FT]
    line_plot(
        FIG_DIR / "ep_ft_cpus.png",
        "EP FT CPUS UTILIZADAS PERF",
        xlabels,
        [
            ("EP314", [value(summary, "3.14t", "EP", t, "perf_cpus_utilized") for t in THREADS_EP_FT]),
            ("EP315", [value(summary, "3.15t", "EP", t, "perf_cpus_utilized") for t in THREADS_EP_FT]),
            ("FT314", [value(summary, "3.14t", "FT", t, "perf_cpus_utilized") for t in THREADS_EP_FT]),
            ("FT315", [value(summary, "3.15t", "FT", t, "perf_cpus_utilized") for t in THREADS_EP_FT]),
        ],
        "CPUs usadas",
    )
    line_plot(
        FIG_DIR / "ep_ft_ipc.png",
        "EP FT IPC PERF",
        xlabels,
        [
            ("EP314", [value(summary, "3.14t", "EP", t, "perf_ipc") for t in THREADS_EP_FT]),
            ("EP315", [value(summary, "3.15t", "EP", t, "perf_ipc") for t in THREADS_EP_FT]),
            ("FT314", [value(summary, "3.14t", "FT", t, "perf_ipc") for t in THREADS_EP_FT]),
            ("FT315", [value(summary, "3.15t", "FT", t, "perf_ipc") for t in THREADS_EP_FT]),
        ],
        "IPC",
    )
    line_plot(
        FIG_DIR / "ep_ft_cache.png",
        "EP FT FALLOS CACHE PERF",
        xlabels,
        [
            ("EP314", [pct(value(summary, "3.14t", "EP", t, "perf_cache_miss_ratio")) for t in THREADS_EP_FT]),
            ("EP315", [pct(value(summary, "3.15t", "EP", t, "perf_cache_miss_ratio")) for t in THREADS_EP_FT]),
            ("FT314", [pct(value(summary, "3.14t", "FT", t, "perf_cache_miss_ratio")) for t in THREADS_EP_FT]),
            ("FT315", [pct(value(summary, "3.15t", "FT", t, "perf_cache_miss_ratio")) for t in THREADS_EP_FT]),
        ],
        "Fallos cache (%)",
    )

    xlabels = [str(t) for t in THREADS_OTHER]
    for py in PYS:
        line_plot(
            FIG_DIR / f"cg_is_mg_cpus_{py.replace('.', '')}.png",
            f"CG IS MG CPUS PYTHON {py}",
            xlabels,
            [
                ("CG", [value(summary, py, "CG", t, "perf_cpus_utilized") for t in THREADS_OTHER]),
                ("IS", [value(summary, py, "IS", t, "perf_cpus_utilized") for t in THREADS_OTHER]),
                ("MG", [value(summary, py, "MG", t, "perf_cpus_utilized") for t in THREADS_OTHER]),
            ],
            "CPUs usadas",
        )
        line_plot(
            FIG_DIR / f"cg_is_mg_ipc_{py.replace('.', '')}.png",
            f"CG IS MG IPC PYTHON {py}",
            xlabels,
            [
                ("CG", [value(summary, py, "CG", t, "perf_ipc") for t in THREADS_OTHER]),
                ("IS", [value(summary, py, "IS", t, "perf_ipc") for t in THREADS_OTHER]),
                ("MG", [value(summary, py, "MG", t, "perf_ipc") for t in THREADS_OTHER]),
            ],
            "IPC",
        )
        line_plot(
            FIG_DIR / f"cg_is_mg_cache_{py.replace('.', '')}.png",
            f"CG IS MG FALLOS CACHE PYTHON {py}",
            xlabels,
            [
                ("CG", [pct(value(summary, py, "CG", t, "perf_cache_miss_ratio")) for t in THREADS_OTHER]),
                ("IS", [pct(value(summary, py, "IS", t, "perf_cache_miss_ratio")) for t in THREADS_OTHER]),
                ("MG", [pct(value(summary, py, "MG", t, "perf_cache_miss_ratio")) for t in THREADS_OTHER]),
            ],
            "Fallos cache (%)",
        )


def make_figures_matplotlib(summary: dict) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 5.8),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
        }
    )

    colors = {
        "EP314": "#c84b37",
        "EP315": "#777f87",
        "FT314": "#479052",
        "FT315": "#2d5fa0",
        "CG": "#1f5f9f",
        "IS": "#965faa",
        "MG": "#d79037",
    }

    def metric_value(py: str, bench: str, threads: int, column: str, scale: float = 1.0) -> float | None:
        v = value(summary, py, bench, threads, column)
        return None if v is None else v * scale

    def plot_metric(path: Path, title: str, threads: list[int], series: list[tuple[str, list[float | None]]], ylabel: str) -> None:
        fig, ax = plt.subplots()
        for name, ys in series:
            if any(v is not None for v in ys):
                ax.plot(threads, ys, marker="o", linewidth=2, label=name, color=colors.get(name))
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads)
        ax.set_xticklabels([str(t) for t in threads])
        ax.set_xlabel("Hilos")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(ncol=min(4, len(series)), loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save_mpl(fig, path)
        plt.close(fig)

    ep_ft_series = [
        ("EP314", "3.14t", "EP"),
        ("EP315", "3.15t", "EP"),
        ("FT314", "3.14t", "FT"),
        ("FT315", "3.15t", "FT"),
    ]
    plot_metric(
        FIG_DIR / "ep_ft_cpus",
        "EP y FT: CPUs utilizadas en perf stat",
        THREADS_EP_FT,
        [(name, [metric_value(py, bench, t, "perf_cpus_utilized") for t in THREADS_EP_FT]) for name, py, bench in ep_ft_series],
        "CPUs utilizadas",
    )
    plot_metric(
        FIG_DIR / "ep_ft_ipc",
        "EP y FT: IPC en perf stat",
        THREADS_EP_FT,
        [(name, [metric_value(py, bench, t, "perf_ipc") for t in THREADS_EP_FT]) for name, py, bench in ep_ft_series],
        "IPC",
    )
    plot_metric(
        FIG_DIR / "ep_ft_cache",
        "EP y FT: fallos de caché en perf stat",
        THREADS_EP_FT,
        [(name, [metric_value(py, bench, t, "perf_cache_miss_ratio", 100.0) for t in THREADS_EP_FT]) for name, py, bench in ep_ft_series],
        "Fallos de caché (%)",
    )

    for py in PYS:
        tag = py.replace(".", "")
        plot_metric(
            FIG_DIR / f"cg_is_mg_cpus_{tag}",
            f"CG, IS y MG: CPUs utilizadas en Python {py}",
            THREADS_OTHER,
            [(bench, [metric_value(py, bench, t, "perf_cpus_utilized") for t in THREADS_OTHER]) for bench in ["CG", "IS", "MG"]],
            "CPUs utilizadas",
        )
        plot_metric(
            FIG_DIR / f"cg_is_mg_ipc_{tag}",
            f"CG, IS y MG: IPC en Python {py}",
            THREADS_OTHER,
            [(bench, [metric_value(py, bench, t, "perf_ipc") for t in THREADS_OTHER]) for bench in ["CG", "IS", "MG"]],
            "IPC",
        )
        plot_metric(
            FIG_DIR / f"cg_is_mg_cache_{tag}",
            f"CG, IS y MG: fallos de caché en Python {py}",
            THREADS_OTHER,
            [(bench, [metric_value(py, bench, t, "perf_cache_miss_ratio", 100.0) for t in THREADS_OTHER]) for bench in ["CG", "IS", "MG"]],
            "Fallos de caché (%)",
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TFM profiling tables and figures")
    parser.add_argument("campaign_dir", nargs="*", type=Path)
    args = parser.parse_args()

    campaign_dirs = args.campaign_dir or DEFAULT_CAMPAIGNS
    rows = load_rows(campaign_dirs)
    summary = best_by_key(rows)
    make_tables(summary)
    make_figures(summary)
    print(f"Loaded {len(rows)} profiling rows from {len(campaign_dirs)} campaign dirs")
    print(f"Generated profiling tables in {TABLE_DIR}")
    print(f"Generated profiling figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
