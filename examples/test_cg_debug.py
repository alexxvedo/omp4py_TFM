from omputils import omp, set_omp_mode, set_omp_threads
import numpy

set_omp_mode(1)
set_omp_threads(2)

@omp
def test_reduction(n):
    x = numpy.ones(n, dtype=numpy.float64)
    total = 0.0
    with omp("parallel for reduction(+:total)"):
        for i in range(n):
            total = total + x[i]
    return total

result = test_reduction(1000)
print(f"Reduction: {result} (expected 1000.0) {'OK' if result == 1000.0 else 'FAIL'}")

@omp
def test_multiple_parallel_for(n):
    x = numpy.ones(n, dtype=numpy.float64)
    q = numpy.empty(n, dtype=numpy.float64)

    with omp("parallel for"):
        for j in range(n):
            q[j] = x[j] * 2.0

    d = 0.0
    with omp("parallel for reduction(+:d)"):
        for j in range(n):
            d = d + q[j]
    return d

result = test_multiple_parallel_for(1000)
print(f"Multi parallel for: {result} (expected 2000.0) {'OK' if result == 2000.0 else 'FAIL'}")

@omp
def test_cg_pattern(n):
    """Simulates the CG iteration pattern with separate parallel for regions"""
    x = numpy.ones(n, dtype=numpy.float64)
    p = numpy.ones(n, dtype=numpy.float64) * 0.5
    q = numpy.empty(n, dtype=numpy.float64)

    rho = 0.0
    with omp("parallel for reduction(+:rho)"):
        for i in range(n):
            rho = rho + x[i] * x[i]

    for iteration in range(3):
        d = 0.0
        rho0 = rho
        rho = 0.0

        with omp("parallel for"):
            for j in range(n):
                q[j] = p[j] * 2.0

        with omp("parallel for reduction(+:d)"):
            for j in range(n):
                d = d + p[j] * q[j]

        alpha = rho0 / d

        with omp("parallel for reduction(+:rho)"):
            for j in range(n):
                x[j] = x[j] + alpha * p[j]
                rho = rho + x[j] * x[j]

        beta = rho / rho0

        with omp("parallel for"):
            for j in range(n):
                p[j] = x[j] + beta * p[j]

    return rho, rho0, d

rho, rho0, d = test_cg_pattern(100)
print(f"CG pattern: rho={rho:.6f} rho0={rho0:.6f} d={d:.6f}")

set_omp_threads(1)
rho1, rho01, d1 = test_cg_pattern(100)
print(f"CG 1-thread: rho={rho1:.6f} rho0={rho01:.6f} d={d1:.6f}")
print(f"Match: {'OK' if abs(rho - rho1) < 1e-10 else 'FAIL rho=' + str(abs(rho-rho1))}")
