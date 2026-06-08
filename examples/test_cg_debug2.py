import sys, os, math, numpy
from omputils import omp, set_omp_mode, set_omp_threads

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common'))
import npbparams
from c_randdp import randlc

set_omp_mode(1)
npbparams.set_cg_info('S')

NA = npbparams.NA
NZ = NA * (npbparams.NONZER + 1) * (npbparams.NONZER + 1)
NAZ = NA * (npbparams.NONZER + 1)

colidx = numpy.repeat(0, NZ)
rowstr = numpy.repeat(0, NA + 1)
iv = numpy.repeat(0, NA)
arow = numpy.repeat(0, NA)
acol = numpy.repeat(0, NAZ)
aelt = numpy.repeat(0.0, NAZ)
a = numpy.repeat(0.0, NZ)
x = numpy.repeat(1.0, NA + 2)
z = numpy.repeat(0.0, NA + 2)
p = numpy.repeat(0.0, NA + 2)
q = numpy.repeat(0.0, NA + 2)
r = numpy.repeat(0.0, NA + 2)

firstrow = 0
lastrow = NA - 1
firstcol = 0
lastcol = NA - 1
naa = NA

from CG_Python import sparse, makea, vecset, sprnvc, icnvrt, conj_grad

tran = 314159265.0
amult = 1220703125.0
zeta, tran = randlc(tran, amult)
tran = makea(naa, NZ, a, colidx, rowstr, firstrow, lastrow, arow, acol, aelt, iv, tran)

for j in range(lastrow - firstrow + 1):
    for k in range(rowstr[j], rowstr[j + 1]):
        colidx[k] = colidx[k] - firstcol

def conj_grad_serial(colidx, rowstr, x, z, a, p, q, r, naa, fr, lr, fc, lc):
    cgitmax = 25
    rho = 0.0
    end_r = lr - fr + 1
    end_c = lc - fc + 1

    for j in range(naa + 1):
        q[j] = 0.0; z[j] = 0.0; r[j] = x[j]; p[j] = r[j]

    for j in range(end_c):
        rho += r[j] * r[j]

    for cgit in range(1, cgitmax + 1):
        d = 0.0; rho0 = rho; rho = 0.0
        for j in range(end_r):
            q[j] = 0.0
            for k in range(rowstr[j], rowstr[j + 1]):
                q[j] += a[k] * p[colidx[k]]
        for j in range(end_c):
            d += p[j] * q[j]
        alpha = rho0 / d
        for j in range(end_c):
            z[j] += alpha * p[j]
            r[j] -= alpha * q[j]
            rho += r[j] * r[j]
        beta = rho / rho0
        for j in range(end_c):
            p[j] = r[j] + beta * p[j]

    sum_r = 0.0
    for j in range(end_r):
        r[j] = 0.0
        for k in range(rowstr[j], rowstr[j + 1]):
            r[j] += a[k] * z[colidx[k]]
    for j in range(end_c):
        sum_r += (x[j] - r[j]) ** 2
    return math.sqrt(sum_r)

print("=== Serial ===")
x1 = x.copy(); z1 = z.copy(); p1 = p.copy(); q1 = q.copy(); r1 = r.copy()
rnorm_serial = conj_grad_serial(colidx, rowstr, x1, z1, a, p1, q1, r1, naa, firstrow, lastrow, firstcol, lastcol)
print(f"rnorm: {rnorm_serial}")

print("\n=== OMP 1-thread ===")
set_omp_threads(1)
x2 = x.copy(); z2 = z.copy(); p2 = p.copy(); q2 = q.copy(); r2 = r.copy()
rnorm_1t = conj_grad(colidx, rowstr, x2, z2, a, p2, q2, r2, naa, firstrow, lastrow, firstcol, lastcol)
print(f"rnorm: {rnorm_1t}")
print(f"z diff vs serial: {numpy.max(numpy.abs(z1[:naa] - z2[:naa]))}")

print("\n=== OMP 2-thread ===")
set_omp_threads(2)
x3 = x.copy(); z3 = z.copy(); p3 = p.copy(); q3 = q.copy(); r3 = r.copy()
rnorm_2t = conj_grad(colidx, rowstr, x3, z3, a, p3, q3, r3, naa, firstrow, lastrow, firstcol, lastcol)
print(f"rnorm: {rnorm_2t}")
print(f"z diff vs serial: {numpy.max(numpy.abs(z1[:naa] - z3[:naa]))}")
print(f"z diff 1t vs 2t:  {numpy.max(numpy.abs(z2[:naa] - z3[:naa]))}")
