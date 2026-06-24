#!/usr/bin/env python3
"""Generate LaTeX tables and PNG plots for the TFM results chapters."""

from __future__ import annotations

import json
import csv
import math
import os
import statistics
import struct
import zlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_20260601 = ROOT / "results" / "ft3_20260601"
FULL = RESULTS_20260601 / "20260531_202503_full315_nopin" / "summary.csv"
VERSIONS = [
    RESULTS_20260601 / "20260531_202503_versions_nopin" / "summary.csv",
    RESULTS_20260601 / "20260531_202503_versions_extra_threads_nopin" / "summary.csv",
]
TFM = ROOT / "memoria" / "Memoria_TFM_MHPC_USC"
TABLE_DIR = TFM / "contenido" / "tablas"
FIG_RES_DIR = TFM / "imagenes" / "resultados"
FIG_VER_DIR = TFM / "imagenes" / "versiones"

BENCHES = ["CG", "EP", "FT", "IS", "MG"]
CLASSES = ["S", "W", "A"]
THREADS_FULL = [1, 2, 4, 8, 16, 32]
THREADS_VERSIONS = [1, 2, 4, 8, 16, 32]
MODES = [0, 1, 2, 3]
PYS = ["3.13t", "3.14t", "3.15t"]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def parse_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def normalize_summary_row(row: dict) -> dict:
    out = dict(row)
    for key in ("mode", "threads", "rep", "returncode"):
        out[key] = parse_int(out.get(key))
    for key in ("npb_seconds", "mops", "wall_seconds"):
        out[key] = parse_float(out.get(key))
    out["timed_out"] = parse_bool(out.get("timed_out"))
    return out


def load_summary_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [normalize_summary_row(row) for row in csv.DictReader(fh)]


def load_records(path: Path) -> list[dict]:
    if path.is_file() and path.suffix == ".csv":
        return load_summary_csv(path)
    if path.is_dir() and (path / "summary.csv").exists():
        return load_summary_csv(path / "summary.csv")

    records = []
    for p in sorted(path.glob("*.json")):
        with p.open() as f:
            records.append(json.load(f))
    return records


def load_record_sets(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        records.extend(load_records(path))
    return records


def summarize(records: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["python_tag"], r["benchmark"], r["class"], r["mode"], r["threads"])].append(r)

    out = {}
    for key, rows in groups.items():
        seconds = [float(r["npb_seconds"]) for r in rows if r.get("npb_seconds") is not None]
        mops = [float(r["mops"]) for r in rows if r.get("mops") is not None]
        wall = [float(r["wall_seconds"]) for r in rows if r.get("wall_seconds") is not None]
        best_row = min(rows, key=lambda r: float("inf") if r.get("npb_seconds") is None else r["npb_seconds"])
        out[key] = {
            "runs": len(rows),
            "ok": sum(
                r.get("returncode") == 0
                and not r.get("timed_out")
                and r.get("verification") == "SUCCESSFUL"
                for r in rows
            ),
            "best": min(seconds) if seconds else None,
            "mean": statistics.mean(seconds) if seconds else None,
            "median": statistics.median(seconds) if seconds else None,
            "best_mops": max(mops) if mops else None,
            "mean_mops": statistics.mean(mops) if mops else None,
            "best_wall": min(wall) if wall else None,
            "mean_wall": statistics.mean(wall) if wall else None,
            "node": best_row.get("node"),
        }
    return out


def value(summary: dict, py: str, bench: str, cls: str, mode: int, threads: int) -> float | None:
    row = summary.get((py, bench, cls, mode, threads))
    return None if row is None else row["best"]


def mops(summary: dict, py: str, bench: str, cls: str, mode: int, threads: int) -> float | None:
    row = summary.get((py, bench, cls, mode, threads))
    return None if row is None else row["best_mops"]


def speedup(summary: dict, py: str, bench: str, cls: str, mode: int, threads: int) -> float | None:
    t1 = value(summary, py, bench, cls, mode, 1)
    tt = value(summary, py, bench, cls, mode, threads)
    if t1 is None or tt is None or t1 <= 0 or tt <= 0:
        return None
    return t1 / tt


def fmt_s(x: float | None) -> str:
    if x is None:
        return "---"
    if x >= 100:
        return f"{x:.1f}"
    return f"{x:.2f}"


def fmt_x(x: float | None) -> str:
    if x is None or math.isnan(x) or math.isinf(x):
        return "---"
    return f"{x:.2f}$\\times$"


def fmt_num(x: float | None) -> str:
    if x is None:
        return "---"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b <= 0:
        return None
    return a / b


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table_env(
    label: str,
    caption: str,
    spec: str,
    header: str,
    rows: list[str],
    resize: bool = False,
    font_size: str = "\\small",
    arraystretch: float | None = None,
) -> str:
    body = "\n".join(rows)
    tabular = f"\\begin{{tabular}}{{{spec}}}\n\\hline\n{header} \\\\\n\\hline\n{body}\n\\hline\n\\end{{tabular}}"
    if resize:
        tabular = "\\resizebox{\\textwidth}{!}{%\n" + tabular + "\n}"
    arraystretch_cmd = f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}\n" if arraystretch is not None else ""
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"{font_size}\n"
        f"{arraystretch_cmd}"
        f"{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def make_tables(full: dict, versions: dict, full_records: list[dict], version_records: list[dict]) -> None:
    rows = [
        "Campaña principal Python~3.15t & 630 & 630 & 100\\% & S: modos 0--3; W: modos 2--3; A: modo 3 \\\\",
        "Comparación de versiones & 1440 & 1440 & 100\\% & Python~3.13t, 3.14t y 3.15t; clases S y W; modos 2--3 \\\\",
    ]
    write(
        TABLE_DIR / "resultados_cobertura_global.tex",
        table_env(
            "tab:resultados-cobertura-global",
            "Cobertura de las campañas finales ejecutadas en Finisterrae~III.",
            "lrrrr",
            "Campaña & Ejecuciones & Correctas & Éxito & Alcance",
            rows,
            resize=True,
        ),
    )

    rows = []
    for bench in BENCHES:
        vals = [value(full, "3.15t", bench, "S", m, 1) for m in MODES]
        m0m3 = ratio(vals[0], vals[3])
        m2m3 = ratio(vals[2], vals[3])
        rows.append(
            f"{bench} & {fmt_s(vals[0])} & {fmt_s(vals[1])} & {fmt_s(vals[2])} & "
            f"{fmt_s(vals[3])} & {fmt_x(m0m3)} & {fmt_x(m2m3)} \\\\"
        )
    write(
        TABLE_DIR / "resultados_s_t1_modos.tex",
        table_env(
            "tab:resultados-s-t1-modos",
            "Mejor tiempo NAS en clase~S con un hilo para los cuatro modos de ejecución en Python~3.15t.",
            "lrrrrrr",
            "Benchmark & M0 (s) & M1 (s) & M2 (s) & M3 (s) & M0/M3 & M2/M3",
            rows,
        ),
    )

    rows = []
    for bench in BENCHES:
        vals = [speedup(full, "3.15t", bench, "S", m, 32) for m in MODES]
        rows.append(f"{bench} & {fmt_x(vals[0])} & {fmt_x(vals[1])} & {fmt_x(vals[2])} & {fmt_x(vals[3])} \\\\")
    write(
        TABLE_DIR / "resultados_s_speedup32_modos.tex",
        table_env(
            "tab:resultados-s-speedup32-modos",
            "Speedup a 32~hilos respecto a 1~hilo en clase~S para cada modo de Python~3.15t.",
            "lrrrr",
            "Benchmark & M0 & M1 & M2 & M3",
            rows,
        ),
    )

    rows = []
    for bench in BENCHES:
        m2 = value(full, "3.15t", bench, "W", 2, 1)
        m3 = value(full, "3.15t", bench, "W", 3, 1)
        rows.append(f"{bench} & {fmt_s(m2)} & {fmt_s(m3)} & {fmt_x(ratio(m2, m3))} \\\\")
    write(
        TABLE_DIR / "resultados_w_t1_modos.tex",
        table_env(
            "tab:resultados-w-t1-modos",
            "Mejor tiempo NAS en clase~W con un hilo para los modos compilados de Python~3.15t.",
            "lrrr",
            "Benchmark & M2 (s) & M3 (s) & M2/M3",
            rows,
        ),
    )

    rows = []
    for bench in BENCHES:
        rows.append(
            f"{bench} & {fmt_x(speedup(full, '3.15t', bench, 'W', 2, 32))} & "
            f"{fmt_x(speedup(full, '3.15t', bench, 'W', 3, 32))} \\\\"
        )
    write(
        TABLE_DIR / "resultados_w_speedup32_modos.tex",
        table_env(
            "tab:resultados-w-speedup32-modos",
            "Speedup a 32~hilos respecto a 1~hilo en clase~W para los modos compilados de Python~3.15t.",
            "lrr",
            "Benchmark & M2 & M3",
            rows,
        ),
    )

    rows = []
    for bench in BENCHES:
        vals = [value(full, "3.15t", bench, "A", 3, t) for t in THREADS_FULL]
        rows.append(f"{bench} & " + " & ".join(fmt_s(v) for v in vals) + " \\\\")
    write(
        TABLE_DIR / "resultados_a_mode3_tiempos.tex",
        table_env(
            "tab:resultados-a-mode3-tiempos",
            "Mejor tiempo NAS en clase~A, modo~3, Python~3.15t.",
            "lrrrrrr",
            "Benchmark & 1 & 2 & 4 & 8 & 16 & 32",
            rows,
        ),
    )

    rows = []
    for bench in BENCHES:
        vals = [speedup(full, "3.15t", bench, "A", 3, t) for t in THREADS_FULL]
        rows.append(f"{bench} & " + " & ".join(fmt_x(v) for v in vals) + " \\\\")
    write(
        TABLE_DIR / "resultados_a_mode3_speedups.tex",
        table_env(
            "tab:resultados-a-mode3-speedups",
            "Speedup por número de hilos en clase~A, modo~3, Python~3.15t.",
            "lrrrrrr",
            "Benchmark & 1 & 2 & 4 & 8 & 16 & 32",
            rows,
        ),
    )

    rows = []
    for cls in CLASSES:
        for bench in BENCHES:
            vals = [(t, value(full, "3.15t", bench, cls, 3, t)) for t in THREADS_FULL]
            vals = [(t, v) for t, v in vals if v is not None]
            if not vals:
                continue
            best_t, best_v = min(vals, key=lambda item: item[1])
            t1 = value(full, "3.15t", bench, cls, 3, 1)
            t32 = value(full, "3.15t", bench, cls, 3, 32)
            rows.append(
                f"{cls} & {bench} & {best_t} & {fmt_s(best_v)} & {fmt_s(t1)} & "
                f"{fmt_x(ratio(t1, best_v))} & {fmt_s(t32)} & {fmt_x(ratio(t1, t32))} \\\\"
            )
    write(
        TABLE_DIR / "resultados_mode3_optimos.tex",
        table_env(
            "tab:resultados-mode3-optimos",
            "Mejor número de hilos y speedup asociado en modo~3, Python~3.15t.",
            "llrrrrrr",
            "Clase & Bench. & Mejor hilo & Mejor tiempo (s) & T1 (s) & Speedup ópt. & T32 (s) & Speedup32",
            rows,
            resize=True,
        ),
    )

    rows = []
    for cls in CLASSES:
        for bench in BENCHES:
            rows.append(
                f"{cls} & {bench} & {fmt_s(value(full, '3.15t', bench, cls, 3, 1))} & "
                f"{fmt_s(value(full, '3.15t', bench, cls, 3, 32))} & "
                f"{fmt_x(speedup(full, '3.15t', bench, cls, 3, 32))} & "
                f"{fmt_num(mops(full, '3.15t', bench, cls, 3, 32))} \\\\"
            )
    write(
        TABLE_DIR / "resultados_mode3_t32_sintesis.tex",
        table_env(
            "tab:resultados-mode3-t32-sintesis",
            "Síntesis de tiempos, speedup y Mop/s a 32~hilos en modo~3, Python~3.15t.",
            "llrrrr",
            "Clase & Bench. & T1 (s) & T32 (s) & Speedup32 & Mop/s32",
            rows,
            resize=True,
        ),
    )

    rows = []
    for cls in ["S", "W"]:
        for bench in BENCHES:
            vals = [value(versions, py, bench, cls, 3, 1) for py in PYS]
            rows.append(f"{cls} & {bench} & " + " & ".join(fmt_s(v) for v in vals) + " \\\\")
    write(
        TABLE_DIR / "versiones_t1_mode3.tex",
        table_env(
            "tab:versiones-t1-mode3",
            "Mejor tiempo NAS con 1~hilo en modo~3 para cada versión de Python.",
            "llrrr",
            "Clase & Bench. & 3.13t (s) & 3.14t (s) & 3.15t (s)",
            rows,
        ),
    )

    rows = []
    for cls in ["S", "W"]:
        for bench in BENCHES:
            vals = [value(versions, py, bench, cls, 3, 32) for py in PYS]
            rows.append(f"{cls} & {bench} & " + " & ".join(fmt_s(v) for v in vals) + " \\\\")
    write(
        TABLE_DIR / "versiones_t32_mode3.tex",
        table_env(
            "tab:versiones-t32-mode3",
            "Mejor tiempo NAS con 32~hilos en modo~3 para cada versión de Python.",
            "llrrr",
            "Clase & Bench. & 3.13t (s) & 3.14t (s) & 3.15t (s)",
            rows,
        ),
    )

    rows = []
    for cls in ["S", "W"]:
        for bench in BENCHES:
            vals = [speedup(versions, py, bench, cls, 3, 32) for py in PYS]
            rows.append(f"{cls} & {bench} & " + " & ".join(fmt_x(v) for v in vals) + " \\\\")
    write(
        TABLE_DIR / "versiones_speedup32_mode3.tex",
        table_env(
            "tab:versiones-speedup32-mode3",
            "Speedup a 32~hilos respecto a 1~hilo en modo~3 para cada versión de Python.",
            "llrrr",
            "Clase & Bench. & 3.13t & 3.14t & 3.15t",
            rows,
        ),
    )

    rows = []
    for cls in ["S", "W"]:
        for bench in BENCHES:
            for py in PYS:
                vals = [value(versions, py, bench, cls, 3, t) for t in THREADS_VERSIONS]
                rows.append(f"{cls} & {bench} & {py} & " + " & ".join(fmt_s(v) for v in vals) + " \\\\")
    write(
        TABLE_DIR / "versiones_mode3_tiempos_completos.tex",
        table_env(
            "tab:versiones-mode3-tiempos-completos",
            "Mejor tiempo NAS por número de hilos en modo~3 para cada versión de Python.",
            "lllrrrrrr",
            "Clase & Bench. & Python & 1 & 2 & 4 & 8 & 16 & 32",
            rows,
            resize=True,
        ),
    )

    rows = []
    for cls in ["S", "W"]:
        for bench in BENCHES:
            for py in PYS:
                vals = [speedup(versions, py, bench, cls, 3, t) for t in THREADS_VERSIONS]
                rows.append(f"{cls} & {bench} & {py} & " + " & ".join(fmt_x(v) for v in vals) + " \\\\")
    write(
        TABLE_DIR / "versiones_mode3_speedups_completos.tex",
        table_env(
            "tab:versiones-mode3-speedups-completos",
            "Speedup por número de hilos en modo~3 para cada versión de Python.",
            "lllrrrrrr",
            "Clase & Bench. & Python & 1 & 2 & 4 & 8 & 16 & 32",
            rows,
            resize=True,
        ),
    )

    rows = []
    for cls in ["S", "W"]:
        for bench in BENCHES:
            for py in PYS:
                vals = [(t, value(versions, py, bench, cls, 3, t)) for t in THREADS_VERSIONS]
                vals = [(t, v) for t, v in vals if v is not None]
                best_t, best_v = min(vals, key=lambda item: item[1])
                t1 = value(versions, py, bench, cls, 3, 1)
                t32 = value(versions, py, bench, cls, 3, 32)
                rows.append(
                    f"{cls} & {bench} & {py} & {best_t} & {fmt_s(best_v)} & "
                    f"{fmt_x(ratio(t1, best_v))} & {fmt_s(t32)} & {fmt_x(ratio(t1, t32))} \\\\"
                )
    write(
        TABLE_DIR / "versiones_mode3_optimos.tex",
        table_env(
            "tab:versiones-mode3-optimos",
            "Mejor número de hilos en modo~3 para cada versión, clase y benchmark.",
            "lllrrrrr",
            "Clase & Bench. & Python & Mejor hilo & Mejor tiempo (s) & Speedup ópt. & T32 (s) & Speedup32",
            rows,
            resize=True,
        ),
    )

    rows = []
    for cls in ["S", "W"]:
        for bench in BENCHES:
            vals = [(py, value(versions, py, bench, cls, 3, 32)) for py in PYS]
            vals = [(py, v) for py, v in vals if v is not None]
            ordered = sorted(vals, key=lambda item: item[1])
            winner, best_v = ordered[0]
            next_v = ordered[1][1] if len(ordered) > 1 else None
            margin = None if next_v is None or best_v <= 0 else (next_v / best_v - 1) * 100
            rows.append(
                f"{cls} & {bench} & {winner} & {fmt_s(best_v)} & "
                f"{fmt_s(next_v)} & {fmt_num(margin)}\\% \\\\"
            )
    write(
        TABLE_DIR / "versiones_ganador_t32_mode3.tex",
        table_env(
            "tab:versiones-ganador-t32-mode3",
            "Versión más rápida con 32~hilos en modo~3 y margen sobre la segunda mejor.",
            "llrrrr",
            "Clase & Bench. & Ganadora & Tiempo (s) & Segunda (s) & Margen",
            rows,
        ),
    )

    rows = []
    for cls in ["S", "W"]:
        for bench in BENCHES:
            for py in PYS:
                m2 = value(versions, py, bench, cls, 2, 1)
                m3 = value(versions, py, bench, cls, 3, 1)
                rows.append(f"{cls} & {bench} & {py} & {fmt_s(m2)} & {fmt_s(m3)} & {fmt_x(ratio(m2, m3))} \\\\")
    write(
        TABLE_DIR / "versiones_tipado_t1.tex",
        table_env(
            "tab:versiones-tipado-t1",
            "Efecto del tipado Cython con 1~hilo en la campaña de comparación de versiones.",
            "lllrrr",
            "Clase & Bench. & Python & M2 (s) & M3 (s) & M2/M3",
            rows,
            resize=True,
            font_size="\\scriptsize",
            arraystretch=0.92,
        ),
    )


COLORS = {
    "CG": (35, 95, 160),
    "EP": (200, 75, 55),
    "FT": (70, 145, 80),
    "IS": (150, 95, 170),
    "MG": (215, 145, 55),
    "3.13t": (40, 105, 180),
    "3.14t": (205, 80, 70),
    "3.15t": (75, 150, 90),
    "M0": (80, 80, 80),
    "M1": (90, 125, 190),
    "M2": (205, 125, 50),
    "M3": (70, 150, 90),
    "S": (40, 105, 180),
    "W": (205, 80, 70),
    "A": (75, 150, 90),
}


class Canvas:
    def __init__(self, width: int, height: int, bg=(255, 255, 255)):
        self.w = width
        self.h = height
        self.px = bytearray(bg * (width * height))

    def set(self, x: int, y: int, color: tuple[int, int, int]):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.px[i : i + 3] = bytes(color)

    def line(self, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int], width: int = 2):
        x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for s in range(steps + 1):
            t = s / max(steps, 1)
            x = round(x0 + (x1 - x0) * t)
            y = round(y0 + (y1 - y0) * t)
            r = width // 2
            for yy in range(y - r, y + r + 1):
                for xx in range(x - r, x + r + 1):
                    self.set(xx, yy, color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], fill=True):
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        if fill:
            for y in range(max(0, y0), min(self.h, y1 + 1)):
                for x in range(max(0, x0), min(self.w, x1 + 1)):
                    self.set(x, y, color)
        else:
            self.line(x0, y0, x1, y0, color, 1)
            self.line(x1, y0, x1, y1, color, 1)
            self.line(x1, y1, x0, y1, color, 1)
            self.line(x0, y1, x0, y0, color, 1)

    def circle(self, cx: int, cy: int, r: int, color: tuple[int, int, int]):
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.set(x, y, color)

    def text(self, x: int, y: int, text: str, color=(30, 30, 30), scale: int = 2):
        cursor = x
        for ch in text:
            glyph = FONT.get(ch, FONT.get("?", []))
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cursor + gx * scale, y + gy * scale, cursor + (gx + 1) * scale - 1, y + (gy + 1) * scale - 1, color)
            cursor += 6 * scale

    def write_png(self, path: Path):
        raw = b"".join(b"\x00" + self.px[y * self.w * 3 : (y + 1) * self.w * 3] for y in range(self.h))
        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)


FONT = {
    " ": ["00000"] * 7,
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "01010", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "a": ["00000", "00000", "01110", "00001", "01111", "10001", "01111"],
    "b": ["10000", "10000", "11110", "10001", "10001", "10001", "11110"],
    "c": ["00000", "00000", "01111", "10000", "10000", "10000", "01111"],
    "d": ["00001", "00001", "01111", "10001", "10001", "10001", "01111"],
    "e": ["00000", "00000", "01110", "10001", "11111", "10000", "01110"],
    "f": ["00110", "01000", "01000", "11100", "01000", "01000", "01000"],
    "g": ["00000", "01111", "10001", "10001", "01111", "00001", "01110"],
    "h": ["10000", "10000", "11110", "10001", "10001", "10001", "10001"],
    "i": ["00100", "00000", "01100", "00100", "00100", "00100", "01110"],
    "l": ["01100", "00100", "00100", "00100", "00100", "00100", "01110"],
    "m": ["00000", "00000", "11010", "10101", "10101", "10101", "10101"],
    "n": ["00000", "00000", "11110", "10001", "10001", "10001", "10001"],
    "o": ["00000", "00000", "01110", "10001", "10001", "10001", "01110"],
    "p": ["00000", "00000", "11110", "10001", "11110", "10000", "10000"],
    "r": ["00000", "00000", "10110", "11001", "10000", "10000", "10000"],
    "s": ["00000", "00000", "01111", "10000", "01110", "00001", "11110"],
    "t": ["01000", "01000", "11100", "01000", "01000", "01000", "00110"],
    "u": ["00000", "00000", "10001", "10001", "10001", "10011", "01101"],
    "v": ["00000", "00000", "10001", "10001", "01010", "01010", "00100"],
    "x": ["00000", "00000", "10001", "01010", "00100", "01010", "10001"],
    "?": ["01110", "10001", "00010", "00100", "00100", "00000", "00100"],
}


def y_log_map(v: float, ymin: float, ymax: float, top: int, bottom: int) -> float:
    v = max(v, ymin)
    lv = math.log10(v)
    return bottom - (lv - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * (bottom - top)


def y_lin_map(v: float, ymin: float, ymax: float, top: int, bottom: int) -> float:
    return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)


def line_chart(path: Path, title: str, xlabels: list[str], series: list[tuple[str, list[float | None]]], ylabel: str, logy=False):
    c = Canvas(1200, 720)
    left, right, top, bottom = 100, 1040, 80, 600
    c.text(100, 25, title, scale=3)
    vals = [v for _, ys in series for v in ys if v is not None and v > 0]
    if not vals:
        return
    ymin = min(vals)
    ymax = max(vals)
    if logy:
        ymin = 10 ** math.floor(math.log10(max(ymin, 0.001)))
        ymax = 10 ** math.ceil(math.log10(ymax))
    else:
        ymin = 0
        ymax = max(1.2, ymax * 1.15)
    c.line(left, bottom, right, bottom, (40, 40, 40), 2)
    c.line(left, top, left, bottom, (40, 40, 40), 2)
    for i, lab in enumerate(xlabels):
        x = left + i * (right - left) / (len(xlabels) - 1)
        c.line(x, bottom, x, bottom + 8, (40, 40, 40), 2)
        c.text(int(x - 10), bottom + 22, lab, scale=2)
    ticks = [0.25, 0.5, 1, 2, 4, 8] if not logy else [ymin * (10 ** i) for i in range(int(round(math.log10(ymax / ymin))) + 1)]
    for tick in ticks:
        if tick < ymin or tick > ymax:
            continue
        y = y_log_map(tick, ymin, ymax, top, bottom) if logy else y_lin_map(tick, ymin, ymax, top, bottom)
        c.line(left - 5, y, right, y, (220, 220, 220), 1)
        c.text(20, int(y - 10), fmt_num(tick), scale=2)
    for name, ys in series:
        pts = []
        for i, v in enumerate(ys):
            if v is None or v <= 0:
                continue
            x = left + i * (right - left) / (len(xlabels) - 1)
            y = y_log_map(v, ymin, ymax, top, bottom) if logy else y_lin_map(v, ymin, ymax, top, bottom)
            pts.append((x, y))
        color = COLORS.get(name, (0, 0, 0))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            c.line(x0, y0, x1, y1, color, 4)
        for x, y in pts:
            c.circle(round(x), round(y), 6, color)
    lx, ly = 1060, 90
    for name, _ in series:
        color = COLORS.get(name, (0, 0, 0))
        c.rect(lx, ly + 4, lx + 18, ly + 18, color)
        c.text(lx + 28, ly, name, scale=2)
        ly += 34
    c.text(18, 630, ylabel, scale=2)
    c.write_png(path)


def grouped_bar(path: Path, title: str, groups: list[str], bars: list[tuple[str, list[float | None]]], ylabel: str, logy=False):
    c = Canvas(1200, 720)
    left, right, top, bottom = 100, 1020, 80, 600
    c.text(100, 25, title, scale=3)
    vals = [v for _, ys in bars for v in ys if v is not None and v > 0]
    if not vals:
        return
    ymin = min(vals) if logy else 0
    ymax = max(vals)
    if logy:
        ymin = 10 ** math.floor(math.log10(max(ymin, 0.001)))
        ymax = 10 ** math.ceil(math.log10(ymax))
    else:
        ymax = max(1.2, ymax * 1.15)
    c.line(left, bottom, right, bottom, (40, 40, 40), 2)
    c.line(left, top, left, bottom, (40, 40, 40), 2)
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32] if not logy else [ymin * (10 ** i) for i in range(int(round(math.log10(ymax / ymin))) + 1)]
    for tick in ticks:
        if tick < ymin or tick > ymax:
            continue
        y = y_log_map(tick, ymin, ymax, top, bottom) if logy else y_lin_map(tick, ymin, ymax, top, bottom)
        c.line(left - 5, y, right, y, (220, 220, 220), 1)
        c.text(20, int(y - 10), fmt_num(tick), scale=2)
    slot = (right - left) / len(groups)
    barw = max(6, int(slot / (len(bars) + 2)))
    for gi, group in enumerate(groups):
        center = left + gi * slot + slot / 2
        c.text(int(center - len(group) * 6), bottom + 22, group, scale=2)
        for bi, (name, ys) in enumerate(bars):
            v = ys[gi]
            if v is None or v <= 0:
                continue
            x0 = int(center - (len(bars) * barw) / 2 + bi * barw)
            x1 = x0 + barw - 3
            y = y_log_map(v, ymin, ymax, top, bottom) if logy else y_lin_map(v, ymin, ymax, top, bottom)
            c.rect(x0, int(y), x1, bottom, COLORS.get(name, (0, 0, 0)))
    lx, ly = 1040, 90
    for name, _ in bars:
        c.rect(lx, ly + 4, lx + 18, ly + 18, COLORS.get(name, (0, 0, 0)))
        c.text(lx + 28, ly, name, scale=2)
        ly += 34
    c.text(18, 630, ylabel, scale=2)
    c.write_png(path)


def mix_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat_color_speedup(value_: float | None, vmax: float) -> tuple[int, int, int]:
    if value_ is None or math.isnan(value_) or math.isinf(value_):
        return (238, 238, 238)
    red = (190, 70, 65)
    yellow = (244, 212, 112)
    green = (68, 150, 90)
    if value_ <= 1.0:
        return mix_color(red, yellow, value_)
    return mix_color(yellow, green, (value_ - 1.0) / max(vmax - 1.0, 1.0))


def heat_color_sequential(value_: float | None, vmax: float) -> tuple[int, int, int]:
    if value_ is None or math.isnan(value_) or math.isinf(value_):
        return (238, 238, 238)
    pale = (255, 242, 190)
    orange = (230, 130, 65)
    red = (160, 45, 55)
    t = min(1.0, max(0.0, value_ / vmax))
    if t <= 0.5:
        return mix_color(pale, orange, t * 2.0)
    return mix_color(orange, red, (t - 0.5) * 2.0)


def color_luminance(color: tuple[int, int, int]) -> float:
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def canvas_heatmap(
    path: Path,
    title: str,
    columns: list[str],
    rows: list[str],
    values: list[list[float | None]],
    color_fn,
    value_fmt,
    legend: str,
) -> None:
    c = Canvas(1200, 720)
    c.text(80, 32, title, scale=3)
    left, top = 220, 145
    plot_w, plot_h = 760, 420
    cell_w = plot_w // len(columns)
    cell_h = plot_h // len(rows)

    for j, label in enumerate(columns):
        x = left + j * cell_w + cell_w // 2 - len(label) * 9
        c.text(x, top - 44, label, scale=3)
    for i, label in enumerate(rows):
        y = top + i * cell_h + cell_h // 2 - 11
        c.text(70, y, label, scale=3)

    for i, row_values in enumerate(values):
        for j, value_ in enumerate(row_values):
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            color = color_fn(value_)
            c.rect(x0, y0, x0 + cell_w - 2, y0 + cell_h - 2, color)
            c.rect(x0, y0, x0 + cell_w - 2, y0 + cell_h - 2, (255, 255, 255), fill=False)
            text = value_fmt(value_)
            text_color = (255, 255, 255) if color_luminance(color) < 115 else (25, 25, 25)
            c.text(x0 + cell_w // 2 - len(text) * 9, y0 + cell_h // 2 - 11, text, text_color, scale=3)

    c.text(left, top + plot_h + 38, legend, color=(45, 45, 45), scale=2)
    c.write_png(path)


def save_mpl(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")


def make_figures_matplotlib(full: dict, versions: dict) -> bool:
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
            "font.size": 20,
            "axes.titlesize": 21,
            "axes.labelsize": 20,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 19,
            "lines.linewidth": 2.8,
            "lines.markersize": 10,
            "legend.frameon": False,
        }
    )
    color = {
        "CG": "#1f5f9f",
        "EP": "#c84b37",
        "FT": "#479052",
        "IS": "#965faa",
        "MG": "#d79037",
        "3.13t": "#2869b4",
        "3.14t": "#cd5046",
        "3.15t": "#4b965a",
        "M0": "#555555",
        "M1": "#5a7dbe",
        "M2": "#cd7d32",
        "M3": "#46965a",
        "S": "#2869b4",
        "W": "#cd5046",
        "A": "#4b965a",
    }

    def z(values):
        return [0 if v is None else v for v in values]

    x = list(range(len(BENCHES)))
    width = 0.18
    fig, ax = plt.subplots()
    for i, m in enumerate(MODES):
        vals = [value(full, "3.15t", b, "S", m, 1) for b in BENCHES]
        ax.bar([j + (i - 1.5) * width for j in x], vals, width, label=f"Modo {m}", color=color[f"M{m}"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHES)
    ax.set_ylabel("Tiempo NAS (s, escala log)")
    ax.set_title("Clase S: efecto del modo de ejecución con 1 hilo")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_mpl(fig, FIG_RES_DIR / "clase_s_t1_modos")
    plt.close(fig)

    for cls in CLASSES:
        fig, ax = plt.subplots()
        for bench in BENCHES:
            vals = z([speedup(full, "3.15t", bench, cls, 3, t) for t in THREADS_FULL])
            ax.plot(THREADS_FULL, vals, marker="o", linewidth=2, label=bench, color=color[bench])
        ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS_FULL)
        ax.set_xticklabels([str(t) for t in THREADS_FULL])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Hilos")
        ax.set_ylabel("Speedup frente a 1 hilo")
        ax.set_title(f"Python 3.15t, modo 3: escalabilidad en clase {cls}")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save_mpl(fig, FIG_RES_DIR / f"mode3_speedup_clase_{cls.lower()}")
        plt.close(fig)

    for cls in CLASSES:
        fig, ax = plt.subplots()
        for bench in BENCHES:
            vals = [value(full, "3.15t", bench, cls, 3, t) for t in THREADS_FULL]
            if any(v is not None and v > 0 for v in vals):
                ax.plot(THREADS_FULL, vals, marker="o", linewidth=2, label=bench, color=color[bench])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(THREADS_FULL)
        ax.set_xticklabels([str(t) for t in THREADS_FULL])
        ax.set_xlabel("Hilos")
        ax.set_ylabel("Tiempo NAS (s, escala log)")
        ax.set_title(f"Python 3.15t, modo 3: tiempos en clase {cls}")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save_mpl(fig, FIG_RES_DIR / f"mode3_tiempo_clase_{cls.lower()}")
        plt.close(fig)

    for bench in BENCHES:
        fig, ax = plt.subplots()
        for cls in CLASSES:
            vals = [speedup(full, "3.15t", bench, cls, 3, t) for t in THREADS_FULL]
            if any(v is not None for v in vals):
                ax.plot(THREADS_FULL, vals, marker="o", linewidth=2, label=f"Clase {cls}", color=color[cls])
        ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS_FULL)
        ax.set_xticklabels([str(t) for t in THREADS_FULL])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Hilos")
        ax.set_ylabel("Speedup frente a 1 hilo")
        ax.set_title(f"{bench}: speedup por clase en Python 3.15t, modo 3")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save_mpl(fig, FIG_RES_DIR / f"benchmark_{bench.lower()}_speedup_clases")
        plt.close(fig)

    for bench in BENCHES:
        fig, ax = plt.subplots()
        for cls in CLASSES:
            vals = [value(full, "3.15t", bench, cls, 3, t) for t in THREADS_FULL]
            if any(v is not None and v > 0 for v in vals):
                ax.plot(THREADS_FULL, vals, marker="o", linewidth=2, label=f"Clase {cls}", color=color[cls])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(THREADS_FULL)
        ax.set_xticklabels([str(t) for t in THREADS_FULL])
        ax.set_xlabel("Hilos")
        ax.set_ylabel("Tiempo NAS (s, escala log)")
        ax.set_title(f"{bench}: tiempos por clase en Python 3.15t, modo 3")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save_mpl(fig, FIG_RES_DIR / f"benchmark_{bench.lower()}_tiempos_clases")
        plt.close(fig)

    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    heatmap_data = np.array(
        [[speedup(full, "3.15t", bench, cls, 3, 32) or float("nan") for bench in BENCHES] for cls in CLASSES],
        dtype=float,
    )
    vmax = max(1.1, float(np.nanmax(heatmap_data)))
    masked_heatmap = np.ma.masked_invalid(heatmap_data)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#efefef")
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    im = ax.imshow(
        masked_heatmap,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax),
        aspect="auto",
    )
    ax.set_xticks(range(len(BENCHES)))
    ax.set_xticklabels(BENCHES)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([f"Clase {cls}" for cls in CLASSES])
    ax.set_xlabel("Benchmark", fontsize=16)
    ax.set_ylabel("Clase", fontsize=16)
    ax.set_title("Python 3.15t, modo 3: speedup a 32 hilos", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    for i, cls in enumerate(CLASSES):
        for j, bench in enumerate(BENCHES):
            v = heatmap_data[i, j]
            if np.isnan(v):
                ax.text(j, i, "---", ha="center", va="center", color="#555555", fontsize=14)
                continue
            text_color = "white" if v < 0.45 or v > 7.0 else "black"
            ax.text(j, i, f"{v:.2f}x", ha="center", va="center", color=text_color, fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Speedup", fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    save_mpl(fig, FIG_RES_DIR / "mode3_speedup32_global")
    plt.close(fig)

    groups = [f"{cls}-{bench}" for cls in ["S", "W"] for bench in BENCHES]
    x = list(range(len(groups)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for i, py in enumerate(PYS):
        vals = [value(versions, py, bench, cls, 3, 32) for cls in ["S", "W"] for bench in BENCHES]
        ax.bar([j + (i - 1) * width for j in x], vals, width, label=py, color=color[py])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45)
    ax.set_ylabel("Tiempo NAS (s, escala log)")
    ax.set_title("Modo 3: tiempo con 32 hilos por versión")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_mpl(fig, FIG_VER_DIR / "mode3_t32_versiones")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for i, py in enumerate(PYS):
        vals = z([speedup(versions, py, bench, cls, 3, 32) for cls in ["S", "W"] for bench in BENCHES])
        ax.bar([j + (i - 1) * width for j in x], vals, width, label=py, color=color[py])
    ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45)
    ax.set_ylabel("Speedup a 32 hilos")
    ax.set_title("Modo 3: speedup a 32 hilos por versión")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_mpl(fig, FIG_VER_DIR / "mode3_speedup32_versiones")
    plt.close(fig)

    for cls in ["S", "W"]:
        for bench in BENCHES:
            fig, ax = plt.subplots()
            for py in PYS:
                vals = z([speedup(versions, py, bench, cls, 3, t) for t in THREADS_VERSIONS])
                ax.plot(THREADS_VERSIONS, vals, marker="o", linewidth=2, label=py, color=color[py])
            ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
            ax.set_xscale("log", base=2)
            ax.set_xticks(THREADS_VERSIONS)
            ax.set_xticklabels([str(t) for t in THREADS_VERSIONS])
            ax.set_xlabel("Hilos")
            ax.set_ylabel("Speedup frente a 1 hilo")
            ax.set_title(f"{bench} clase {cls}, modo 3: comparación entre versiones")
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            save_mpl(fig, FIG_VER_DIR / f"{bench.lower()}_{cls.lower()}_speedup_versiones")
            plt.close(fig)

            fig, ax = plt.subplots()
            for py in PYS:
                vals = [value(versions, py, bench, cls, 3, t) for t in THREADS_VERSIONS]
                if any(v is not None and v > 0 for v in vals):
                    ax.plot(THREADS_VERSIONS, vals, marker="o", linewidth=2, label=py, color=color[py])
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(THREADS_VERSIONS)
            ax.set_xticklabels([str(t) for t in THREADS_VERSIONS])
            ax.set_xlabel("Hilos")
            ax.set_ylabel("Tiempo NAS (s, escala log)")
            ax.set_title(f"{bench} clase {cls}, modo 3: tiempos por versión")
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            save_mpl(fig, FIG_VER_DIR / f"{bench.lower()}_{cls.lower()}_tiempos_versiones")
            plt.close(fig)
    return True


def make_figures(full: dict, versions: dict) -> None:
    if make_figures_matplotlib(full, versions):
        return

    groups = BENCHES
    bars = []
    for m in MODES:
        bars.append((f"M{m}", [value(full, "3.15t", b, "S", m, 1) for b in BENCHES]))
    grouped_bar(FIG_RES_DIR / "clase_s_t1_modos.png", "Clase S: tiempo con 1 hilo por modo", groups, bars, "segundos log", logy=True)

    for cls in CLASSES:
        series = []
        for bench in BENCHES:
            series.append((bench, [speedup(full, "3.15t", bench, cls, 3, t) for t in THREADS_FULL]))
        line_chart(
            FIG_RES_DIR / f"mode3_speedup_clase_{cls.lower()}.png",
            f"Python 3.15t modo 3: speedup clase {cls}",
            [str(t) for t in THREADS_FULL],
            series,
            "speedup",
            logy=False,
        )

    for cls in CLASSES:
        series = []
        for bench in BENCHES:
            series.append((bench, [value(full, "3.15t", bench, cls, 3, t) for t in THREADS_FULL]))
        line_chart(
            FIG_RES_DIR / f"mode3_tiempo_clase_{cls.lower()}.png",
            f"Python 3.15t modo 3: tiempos clase {cls}",
            [str(t) for t in THREADS_FULL],
            series,
            "segundos log",
            logy=True,
        )

    for bench in BENCHES:
        series = []
        for cls in CLASSES:
            series.append((cls, [speedup(full, "3.15t", bench, cls, 3, t) for t in THREADS_FULL]))
        line_chart(
            FIG_RES_DIR / f"benchmark_{bench.lower()}_speedup_clases.png",
            f"{bench}: speedup por clase",
            [str(t) for t in THREADS_FULL],
            series,
            "speedup",
            logy=False,
        )

    for bench in BENCHES:
        series = []
        for cls in CLASSES:
            series.append((cls, [value(full, "3.15t", bench, cls, 3, t) for t in THREADS_FULL]))
        line_chart(
            FIG_RES_DIR / f"benchmark_{bench.lower()}_tiempos_clases.png",
            f"{bench}: tiempos por clase",
            [str(t) for t in THREADS_FULL],
            series,
            "segundos log",
            logy=True,
        )

    heatmap_values = [[speedup(full, "3.15t", bench, cls, 3, 32) for bench in BENCHES] for cls in CLASSES]
    vmax = max(v for row in heatmap_values for v in row if v is not None)
    canvas_heatmap(
        FIG_RES_DIR / "mode3_speedup32_global.png",
        "Python 3.15t modo 3: speedup a 32 hilos",
        BENCHES,
        [f"Clase {cls}" for cls in CLASSES],
        heatmap_values,
        lambda value_: heat_color_speedup(value_, vmax),
        lambda value_: "---" if value_ is None else f"{value_:.2f}x",
        "Rojo: peor que 1 hilo. Verde: speedup alto.",
    )

    groups = [f"{cls}-{bench}" for cls in ["S", "W"] for bench in BENCHES]
    bars = [(py, [value(versions, py, bench, cls, 3, 32) for cls in ["S", "W"] for bench in BENCHES]) for py in PYS]
    grouped_bar(FIG_VER_DIR / "mode3_t32_versiones.png", "Modo 3: tiempo con 32 hilos por version", groups, bars, "segundos log", logy=True)

    bars = [(py, [speedup(versions, py, bench, cls, 3, 32) for cls in ["S", "W"] for bench in BENCHES]) for py in PYS]
    grouped_bar(FIG_VER_DIR / "mode3_speedup32_versiones.png", "Modo 3: speedup a 32 hilos por version", groups, bars, "speedup", logy=False)

    for cls in ["S", "W"]:
        for bench in BENCHES:
            series = [(py, [speedup(versions, py, bench, cls, 3, t) for t in THREADS_VERSIONS]) for py in PYS]
            line_chart(
                FIG_VER_DIR / f"{bench.lower()}_{cls.lower()}_speedup_versiones.png",
                f"{bench} clase {cls}: speedup modo 3",
                [str(t) for t in THREADS_VERSIONS],
                series,
                "speedup",
                logy=False,
            )
            series = [(py, [value(versions, py, bench, cls, 3, t) for t in THREADS_VERSIONS]) for py in PYS]
            line_chart(
                FIG_VER_DIR / f"{bench.lower()}_{cls.lower()}_tiempos_versiones.png",
                f"{bench} clase {cls}: tiempos modo 3",
                [str(t) for t in THREADS_VERSIONS],
                series,
                "segundos log",
                logy=True,
            )


def main() -> None:
    full_records = load_records(FULL)
    version_records = load_record_sets(VERSIONS)
    full = summarize(full_records)
    versions = summarize(version_records)
    make_tables(full, versions, full_records, version_records)
    make_figures(full, versions)
    print(f"Generated tables in {TABLE_DIR}")
    print(f"Generated figures in {FIG_RES_DIR} and {FIG_VER_DIR}")


if __name__ == "__main__":
    main()
