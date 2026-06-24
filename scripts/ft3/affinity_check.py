#!/usr/bin/env python3
"""Sonda de afinidad para FT3.

Mide el eslabon que la campana nunca registro: la mascara de CPU real del
proceso y de los hilos, mas el entorno y las librerias nativas cargadas.
La hipotesis de la memoria es que una variable de afinidad del entorno
restringe la mascara y que los hilos free-threaded la heredan. Esto lo
comprueba directamente.

Uso:
    python affinity_check.py            # imprime un JSON con el diagnostico

Ejecutar bajo cada condicion (sin tocar nada / nopin / pin) y comparar.
"""
import json
import os
import sys
import threading


def mask():
    return sorted(os.sched_getaffinity(0))


info = {}
info["python"] = sys.version.split()[0]
info["gil_off"] = (not sys._is_gil_enabled()) if hasattr(sys, "_is_gil_enabled") else None
info["nproc_visible"] = os.cpu_count()

# 1. Que variables de afinidad ve el proceso (las pongas tu o las ponga SLURM/modulos)
info["env_affinity"] = {
    k: os.environ.get(k)
    for k in ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY",
              "OMP_NUM_THREADS", "OMP_DYNAMIC", "KMP_AFFINITY", "OPENBLAS_NUM_THREADS")
}
info["env_slurm_affinity"] = {
    k: v for k, v in os.environ.items()
    if k.startswith("SLURM") and any(t in k for t in ("CPU", "BIND", "AFFINITY", "DISTRIBUTION", "TASK"))
}

# 2. La mascara, en cada momento clave
info["mask_start"] = mask()
import numpy as np
info["mask_after_numpy_import"] = mask()
a = np.random.rand(1024, 1024)
_ = float((a @ a).sum())          # fuerza una llamada BLAS
info["mask_after_blas_call"] = mask()

# 3. Que runtime OpenMP/BLAS hay cargado (lo que de verdad honra las variables)
libs = [ln.split()[-1] for ln in open("/proc/self/maps") if ".so" in ln]
info["native_libs"] = sorted({
    os.path.basename(x) for x in libs
    if any(t in x for t in ("gomp", "iomp", "mkl", "openblas", "blas", "lapack"))
})

# 4. Mascara por hilo con hilos Python planos (heredan la del proceso)
plain = {}
def _w(i):
    plain[i] = len(os.sched_getaffinity(0))
ts = [threading.Thread(target=_w, args=(i,)) for i in range(8)]
[t.start() for t in ts]
[t.join() for t in ts]
info["plain_thread_mask_sizes"] = sorted(set(plain.values()))

# 5. Mascara por hilo dentro de una region paralela real de OMP4Py
try:
    from omp4py import omp, omp_get_thread_num
    omp_sizes = {}

    @omp
    def _region():
        with omp("parallel num_threads(8)"):
            omp_sizes[omp_get_thread_num()] = len(os.sched_getaffinity(0))

    _region()
    info["omp4py_thread_mask_sizes"] = sorted(set(omp_sizes.values()))
except Exception as exc:  # pragma: no cover - solo diagnostico
    info["omp4py_thread_mask_sizes"] = "ERR: %r" % exc

print(json.dumps(info, indent=2))
