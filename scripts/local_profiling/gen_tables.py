#!/usr/bin/env python3
"""Generate the two LaTeX tables for the objective-5 (Python 3.15 sampling
profiler) section, from prof_summary.csv."""
import os, csv

OUT = os.path.expanduser("~/prof_local")
TAB = os.path.expanduser(
    "~/Documents/Master/TFM_Separado/omp4py/memoria/Memoria_TFM_MHPC_USC/contenido/tablas")
ORDER = ["EP", "FT", "CG", "MG"]

rows = {}
with open(os.path.join(OUT, "prof_summary.csv")) as f:
    for r in csv.DictReader(f):
        rows[(r["benchmark"], int(r["threads"]))] = r


def g(b, t, key):
    r = rows.get((b, t))
    return float(r[key]) if r else None


# ---- Table 1: synchronization cost (% of parallel execution) vs threads ----
THREADS = [1, 4, 8]
lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\begin{tabular}{l" + "r" * len(THREADS) + "}", r"\hline",
         "Benchmark & " + " & ".join(f"{t} hilos" if t == 1 else str(t) for t in THREADS) + r" \\",
         r"\hline"]
for b in ORDER:
    cells = []
    for t in THREADS:
        v = g(b, t, "sync_of_parallel")
        cells.append(f"{v:.1f}\\%" if v is not None else "---")
    lines.append(f"{b} & " + " & ".join(cells) + r" \\")
lines += [r"\hline", r"\end{tabular}",
          r"\caption{Coste de sincronización medido con el profiler de muestreo de "
          r"Python~3.15 (\texttt{profiling.sampling}, modo \emph{wall}, todos los "
          r"hilos). Porcentaje de las muestras de ejecución paralela que caen en "
          r"primitivas de espera del runtime OMP4Py (barrera y reducción, "
          r"\texttt{threading.Condition.wait}), frente al número de hilos. "
          r"Clase~S, modo~1.}",
          r"\label{tab:profiling-py315-sync}", r"\end{table}"]
with open(os.path.join(TAB, "profiling_py315_sync.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

# ---- Table 2: thread-team churn at 8 threads ----
T = 8
order2 = sorted(ORDER, key=lambda b: g(b, T, "n_workers") or 0)
lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\begin{tabular}{lrr}", r"\hline",
         r"Benchmark & Hilos trabajadores distintos & Sincr./paralelo \\",
         r"\hline"]
for b in order2:
    nw = g(b, T, "n_workers")
    sp = g(b, T, "sync_of_parallel")
    nw_s = f"{int(nw)}" if nw is not None else "---"
    if b == "EP":
        nw_s += " (equipo persistente)"
    lines.append(f"{b} & {nw_s} & {sp:.1f}\\% \\\\")
lines += [r"\hline", r"\end{tabular}",
          r"\caption{Gestión de hilos con 8~hilos según el profiler de muestreo de "
          r"Python~3.15 (clase~S, modo~1). \emph{Hilos trabajadores distintos} es el "
          r"número de identificadores de hilo con muestras de cómputo observados "
          r"durante la ejecución: EP emplea un único equipo persistente, mientras que "
          r"CG, FT y MG crean y destruyen un equipo por región paralela, generando "
          r"cientos o miles de hilos efímeros.}",
          r"\label{tab:profiling-py315-churn}", r"\end{table}"]
with open(os.path.join(TAB, "profiling_py315_churn.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("Escritas:")
print(" ", os.path.join(TAB, "profiling_py315_sync.tex"))
print(" ", os.path.join(TAB, "profiling_py315_churn.tex"))
print("\nResumen para el texto:")
for b in ORDER:
    print(f"  {b}: sync/par 1t={g(b,1,'sync_of_parallel'):.1f}% 4t={g(b,4,'sync_of_parallel'):.1f}% "
          f"8t={g(b,8,'sync_of_parallel'):.1f}%  workers@8={int(g(b,8,'n_workers') or 0)}  "
          f"16t={'%.1f%%'%g(b,16,'sync_of_parallel') if g(b,16,'sync_of_parallel') is not None else 'NA'}")
