# ------------------------------------------------------------------------------
#
# The original NPB 3.4.1 version was written in Fortran and belongs to:
# 	http://www.nas.nasa.gov/Software/NPB/
#
# Authors of the Fortran code:
#	M. Yarrow
#	C. Kuszmaul
#
# ------------------------------------------------------------------------------
#
# The serial C++ version is a translation of the original NPB 3.4.1
# Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
#
# Authors of the C++ code:
# 	Dalvan Griebler <dalvangriebler@gmail.com>
# 	Gabriell Araujo <hexenoften@gmail.com>
# 	Junior Loff <loffjh@gmail.com>
#
# ------------------------------------------------------------------------------
#
# The serial Python version is a translation of the NPB serial C++ version
# Serial Python version: https://github.com/danidomenico/NPB-PYTHON/tree/master/NPB-SER
#
# Authors of the Python code:
#	LUPS (Laboratory of Ubiquitous and Parallel Systems)
#	UFPEL (Federal University of Pelotas)
#	Pelotas, Rio Grande do Sul, Brazil
#
# ------------------------------------------------------------------------------

import argparse
import sys
import os
import numpy
import math

from omputils import omp, omp_pure, use_pure, use_compiled, use_compiled_types, set_omp_threads, set_omp_mode

if use_pure():
    omp = omp_pure

try:
	import cython
except ImportError:
	class cython:
		long = []
		double = []

# Local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common'))
import npbparams
from c_randdp import randlc
import c_timers
import c_print_results


# Global variables
NZ = 0
NAZ = 0
T_INIT = 0
T_BENCH = 1
T_CONJ_GRAD = 2
T_LAST = 3

colidx = None
rowstr = None
iv = None
arow = None
acol = None
aelt = None
a = None
x = None
z = None
p = None
q = None
r = None

naa = 0
nzz = 0
firstrow = 0
lastrow = 0
firstcol = 0
lastcol = 0
tran = 0.0
amult = 0.0


def set_global_variables():
	global NZ, NAZ
	global colidx, rowstr, iv, arow, acol, aelt
	global a, x, z, p, q, r

	NZ = npbparams.NA * (npbparams.NONZER + 1) * (npbparams.NONZER + 1)
	NAZ = npbparams.NA * (npbparams.NONZER + 1)

	colidx = numpy.repeat(0, NZ)
	rowstr = numpy.repeat(0, npbparams.NA + 1)
	iv = numpy.repeat(0, npbparams.NA)
	arow = numpy.repeat(0, npbparams.NA)
	acol = numpy.repeat(0, NAZ)
	aelt = numpy.repeat(0.0, NAZ)
	a = numpy.repeat(0.0, NZ)
	x = numpy.repeat(1.0, npbparams.NA + 2)
	z = numpy.repeat(0.0, npbparams.NA + 2)
	p = numpy.repeat(0.0, npbparams.NA + 2)
	q = numpy.repeat(0.0, npbparams.NA + 2)
	r = numpy.repeat(0.0, npbparams.NA + 2)
#END set_global_variables()


def create_zeta_verify_value():
	zeta_verify_value = 0.0
	if npbparams.CLASS == 'S':
		zeta_verify_value = 8.5971775078648
	elif npbparams.CLASS == 'W':
		zeta_verify_value = 10.362595087124
	elif npbparams.CLASS == 'A':
		zeta_verify_value = 17.130235054029
	elif npbparams.CLASS == 'B':
		zeta_verify_value = 22.712745482631
	elif npbparams.CLASS == 'C':
		zeta_verify_value = 28.973605592845
	elif npbparams.CLASS == 'D':
		zeta_verify_value = 52.514532105794
	elif npbparams.CLASS == 'E':
		zeta_verify_value = 77.522164599383

	return zeta_verify_value
#END create_zeta_verify_value


@omp(compile=use_compiled())
def shift_column_indices(colidx, rowstr, end, firstcol_arg):
	with omp("parallel for"):
		for j in range(end):
			for k in range(rowstr[j], rowstr[j + 1]):
				colidx[k] = colidx[k] - firstcol_arg
#END shift_column_indices()


@omp(compile=use_compiled())
def shift_column_indices_types(colidx_arg, rowstr_arg, end: int, firstcol_arg: int):
	colidx_view: cython.long[:] = colidx_arg
	rowstr_view: cython.long[:] = rowstr_arg
	j: int
	k: int

	with omp("parallel for"):
		for j in range(end):
			for k in range(rowstr_view[j], rowstr_view[j + 1]):
				colidx_view[k] = colidx_view[k] - firstcol_arg
#END shift_column_indices_types()


@omp(compile=use_compiled())
def fill_vector(x, value, end):
	with omp("parallel for"):
		for i in range(end):
			x[i] = value
#END fill_vector()


@omp(compile=use_compiled())
def fill_vector_types(x_arg, value: float, end: int):
	x_view: cython.double[:] = x_arg
	i: int

	with omp("parallel for"):
		for i in range(end):
			x_view[i] = value
#END fill_vector_types()


@omp(compile=use_compiled())
def normalize_z(x, z, compute_zeta, end, shift):
	norm_temp1 = 0.0
	norm_temp2 = 0.0
	zeta = 0.0

	with omp("parallel"):
		with omp("for reduction(+:norm_temp1,norm_temp2)"):
			for j in range(end):
				norm_temp1 = norm_temp1 + x[j] * z[j]
				norm_temp2 = norm_temp2 + z[j] * z[j]

		with omp("single"):
			norm_temp2 = 1.0 / math.sqrt(norm_temp2)
			if compute_zeta:
				zeta = shift + 1.0 / norm_temp1

		with omp("for"):
			for j in range(end):
				x[j] = norm_temp2 * z[j]

	return norm_temp1, norm_temp2, zeta
#END normalize_z()


@omp(compile=use_compiled())
def normalize_z_types(x_arg, z_arg, compute_zeta: int, end: int, shift: float):
	x: cython.double[:] = x_arg
	z: cython.double[:] = z_arg
	norm_temp1: float = 0.0
	norm_temp2: float = 0.0
	zeta: float = 0.0
	j: int

	with omp("parallel"):
		with omp("for reduction(+:norm_temp1,norm_temp2)"):
			for j in range(end):
				norm_temp1 = norm_temp1 + x[j] * z[j]
				norm_temp2 = norm_temp2 + z[j] * z[j]

		with omp("single"):
			norm_temp2 = 1.0 / math.sqrt(norm_temp2)
			if compute_zeta:
				zeta = shift + 1.0 / norm_temp1

		with omp("for"):
			for j in range(end):
				x[j] = norm_temp2 * z[j]

	return norm_temp1, norm_temp2, zeta
#END normalize_z_types()


@omp(compile=use_compiled())
def conj_grad(colidx,
			rowstr,
			x,
			z,
			a,
			p,
			q,
			r,
			naa_arg,
			firstrow_arg,
			lastrow_arg,
			firstcol_arg,
			lastcol_arg):
	cgitmax = 25
	rho = 0.0
	rho0 = 0.0
	d = 0.0
	sum_r = 0.0
	rnorm = 0.0
	alpha = 0.0
	beta = 0.0

	with omp("parallel"):
		
		with omp("for"):
			for j in range(naa_arg + 1):
				q[j] = 0.0
				z[j] = 0.0
				r[j] = x[j]
				p[j] = r[j]

	end = lastcol_arg - firstcol_arg + 1
	with omp("parallel"):
		with omp("for reduction(+:rho)"):
			for j in range(end):
				rho = rho + r[j] * r[j]

	for cgit in range(1, cgitmax + 1):
		d = 0.0
		rho0 = rho
		rho = 0.0

		end = lastrow_arg - firstrow_arg + 1
		with omp("parallel"):
			with omp("for"):
				for j in range(end):
					q[j] = 0.0
					for k in range(rowstr[j], rowstr[j + 1]):
						q[j] = q[j] + a[k] * p[colidx[k]]

		end = lastcol_arg - firstcol_arg + 1
		with omp("parallel"):
			with omp("for reduction(+:d)"):
				for j in range(end):
					d = d + p[j] * q[j]

		alpha = rho0 / d

		with omp("parallel"):
			with omp("for reduction(+:rho)"):
				for j in range(end):
					z[j] = z[j] + alpha * p[j]
					r[j] = r[j] - alpha * q[j]
					rho = rho + r[j] * r[j]

		beta = rho / rho0

		with omp("parallel"):
			with omp("for"):
				for j in range(end):
					p[j] = r[j] + beta * p[j]

	sum_r = 0.0
	end = lastrow_arg - firstrow_arg + 1
	with omp("parallel"):
		with omp("for"):
			for j in range(end):
				r[j] = 0.0
				for k in range(rowstr[j], rowstr[j + 1]):
					r[j] = r[j] + a[k] * z[colidx[k]]

	end = lastcol_arg - firstcol_arg + 1
	with omp("parallel"):
		with omp("for reduction(+:sum_r)"):
			for j in range(end):
				sum_r = sum_r + (x[j] - r[j]) * (x[j] - r[j])

	rnorm = math.sqrt(sum_r)

	return rnorm
#END conj_grad()


@omp(compile=use_compiled())
def conj_grad_types(colidx_arg,
			rowstr_arg,
			x_arg,
			z_arg,
			a_arg,
			p_arg,
			q_arg,
			r_arg,
			naa_arg: int,
			firstrow_arg: int,
			lastrow_arg: int,
			firstcol_arg: int,
			lastcol_arg: int):
	colidx: cython.long[:] = colidx_arg
	rowstr: cython.long[:] = rowstr_arg
	x: cython.double[:] = x_arg
	z: cython.double[:] = z_arg
	a: cython.double[:] = a_arg
	p: cython.double[:] = p_arg
	q: cython.double[:] = q_arg
	r: cython.double[:] = r_arg

	cgitmax: int = 25
	rho: float = 0.0
	rho0: float = 0.0
	d: float = 0.0
	sum_r: float = 0.0
	rnorm: float = 0.0
	end: int = 0
	alpha: float = 0.0
	beta: float = 0.0
	j: int
	k: int
	cgit: int
	naa_local: int = naa_arg
	firstrow_local: int = firstrow_arg
	lastrow_local: int = lastrow_arg
	firstcol_local: int = firstcol_arg
	lastcol_local: int = lastcol_arg

	with omp("parallel"):
		with omp("single nowait"):
			rho = 0.0
			sum_r = 0.0

		with omp("for"):
			for j in range(naa_local + 1):
				q[j] = 0.0
				z[j] = 0.0
				r[j] = x[j]
				p[j] = r[j]

		end = lastcol_local - firstcol_local + 1
		with omp("for reduction(+:rho)"):
			for j in range(end):
				rho = rho + r[j] * r[j]

		for cgit in range(1, cgitmax + 1):
			with omp("single"):
				d = 0.0
				rho0 = rho
				rho = 0.0

			end = lastrow_local - firstrow_local + 1
			with omp("for schedule(static)"):
				for j in range(end):
					q[j] = 0.0
					for k in range(rowstr[j], rowstr[j + 1]):
						q[j] = q[j] + a[k] * p[colidx[k]]

			end = lastcol_local - firstcol_local + 1
			with omp("for schedule(static) reduction(+:d)"):
				for j in range(end):
					d = d + p[j] * q[j]

			alpha = rho0 / d

			with omp("for reduction(+:rho)"):
				for j in range(end):
					z[j] = z[j] + alpha * p[j]
					r[j] = r[j] - alpha * q[j]
					rho = rho + r[j] * r[j]

			beta = rho / rho0

			with omp("for"):
				for j in range(end):
					p[j] = r[j] + beta * p[j]

		end = lastrow_local - firstrow_local + 1
		with omp("for schedule(static) nowait"):
			for j in range(end):
				r[j] = 0.0
				for k in range(rowstr[j], rowstr[j + 1]):
					r[j] = r[j] + a[k] * z[colidx[k]]

		end = lastcol_local - firstcol_local + 1
		with omp("for schedule(static) reduction(+:sum_r)"):
			for j in range(end):
				sum_r = sum_r + (x[j] - r[j]) * (x[j] - r[j])

		with omp("single"):
			rnorm = math.sqrt(sum_r)
	return rnorm
#END conj_grad_types()


def sparse(a,
		colidx,
		rowstr,
		n,
		nz,
		nozer,
		arow,
		acol,
		aelt,
		firstrow,
		lastrow,
		nzloc,
		rcond,
		shift):

	NONZER_aux = npbparams.NONZER + 1
	nrows = lastrow - firstrow + 1

	for j in range(nrows + 1):
		rowstr[j] = 0

	for i in range(n):
		for nza in range(arow[i]):
			j = acol[i * NONZER_aux + nza] + 1
			rowstr[j] = rowstr[j] + arow[i]

	rowstr[0] = 0
	for j in range(1, nrows + 1):
		rowstr[j] = rowstr[j] + rowstr[j - 1]
	nza = rowstr[nrows] - 1

	if nza > nz:
		print("Space for matrix elements exceeded in sparse")
		print("nza, nzmax = ", nza, ", ", nz)

	for j in range(nrows):
		for k in range(rowstr[j], rowstr[j + 1]):
			a[k] = 0.0
			colidx[k] = -1
		nzloc[j] = 0

	size = 1.0
	ratio = pow(rcond, (1.0 / n))
	for i in range(n):
		for nza in range(arow[i]):
			j = acol[i * NONZER_aux + nza]

			scale = size * aelt[i * NONZER_aux + nza]
			for nzrow in range(arow[i]):
				jcol = acol[i * NONZER_aux + nzrow]
				va = aelt[i * NONZER_aux + nzrow] * scale

				if jcol == j and j == i:
					va = va + rcond - shift

				goto_40 = False
				for k in range(rowstr[j], rowstr[j + 1]):
					if colidx[k] > jcol:
						start = rowstr[j + 1] - 2
						for kk in range(start, k - 1, -1):
							if colidx[kk] > -1:
								a[kk + 1] = a[kk]
								colidx[kk + 1] = colidx[kk]

						colidx[k] = jcol
						a[k] = 0.0
						goto_40 = True
						break
					elif colidx[k] == -1:
						colidx[k] = jcol
						goto_40 = True
						break
					elif colidx[k] == jcol:
						nzloc[j] = nzloc[j] + 1
						goto_40 = True
						break
				if not goto_40:
					print("internal error in sparse: i=", i)
				a[k] = a[k] + va
		size = size * ratio

	for j in range(1, nrows):
		nzloc[j] = nzloc[j] + nzloc[j - 1]

	for j in range(nrows):
		if j > 0:
			j1 = rowstr[j] - nzloc[j - 1]
		else:
			j1 = 0
		j2 = rowstr[j + 1] - nzloc[j]
		nza = rowstr[j]
		for k in range(j1, j2):
			a[k] = a[nza]
			colidx[k] = colidx[nza]
			nza = nza + 1

	for j in range(1, nrows + 1):
		rowstr[j] = rowstr[j] - nzloc[j - 1]
#END sparse()


def icnvrt(x, ipwr2):
	return int(ipwr2 * x)
#END icnvrt()


def vecset(n, v, iv, nzv, i, val):
	sett = False
	for k in range(nzv):
		if iv[k] == i:
			v[k] = val
			sett = True

	if not sett:
		v[nzv] = val
		iv[nzv] = i
		nzv = nzv + 1

	return nzv
#END vecset()


def sprnvc(n, nz, nn1, v, iv, tran_aux):
	nzv = 0

	while nzv < nz:
		vecelt, tran_aux = randlc(tran_aux, amult)
		vecloc, tran_aux = randlc(tran_aux, amult)
		i = icnvrt(vecloc, nn1) + 1
		if i > n:
			continue

		was_gen = False
		for ii in range(nzv):
			if iv[ii] == i:
				was_gen = True
				break

		if was_gen:
			continue
		v[nzv] = vecelt
		iv[nzv] = i
		nzv = nzv + 1

	return tran_aux
#END sprnvc()


def makea(n,
		nz,
		a,
		colidx,
		rowstr,
		firstrow,
		lastrow,
		arow,
		acol,
		aelt,
		iv,
		tran_aux):

	NONZER_aux = npbparams.NONZER + 1
	ivc = numpy.empty(NONZER_aux, dtype=numpy.int32)
	vc = numpy.empty(NONZER_aux, dtype=numpy.float64)

	nn1 = 1
	while True:
		nn1 = 2 * nn1
		if nn1 >= n:
			break

	for iouter in range(n):
		nzv = npbparams.NONZER
		tran_aux = sprnvc(n, nzv, nn1, vc, ivc, tran_aux)
		nzv = vecset(n, vc, ivc, nzv, iouter + 1, 0.5)
		arow[iouter] = nzv
		for ivelt in range(nzv):
			acol[iouter * NONZER_aux + ivelt] = ivc[ivelt] - 1
			aelt[iouter * NONZER_aux + ivelt] = vc[ivelt]

	sparse(a,
		colidx,
		rowstr,
		n,
		nz,
		npbparams.NONZER,
		arow,
		acol,
		aelt,
		firstrow,
		lastrow,
		iv,
		npbparams.RCOND,
		npbparams.SHIFT)

	return tran_aux
#END makea()


def main():
	global naa, nzz, firstrow, lastrow, firstcol, lastcol
	global tran, amult
	global colidx, rowstr, iv, arow, acol, aelt
	global a, x, z, p, q, r

	for i in range(T_LAST):
		c_timers.timer_clear(i)

	t_names = numpy.empty(T_LAST, dtype=object)

	timeron = os.path.isfile("timer.flag")
	if timeron:
		t_names[T_INIT] = "init"
		t_names[T_BENCH] = "benchmk"
		t_names[T_CONJ_GRAD] = "conjgd"

	c_timers.timer_start(T_INIT)

	firstrow = 0
	lastrow = npbparams.NA - 1
	firstcol = 0
	lastcol = npbparams.NA - 1

	zeta_verify_value = create_zeta_verify_value()

	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - CG Benchmark\n")
	print(" Size: %11d" % (npbparams.NA))
	print(" Iterations: %5d" % (npbparams.NITER))

	naa = npbparams.NA
	nzz = NZ

	tran = 314159265.0
	amult = 1220703125.0
	zeta, tran = randlc(tran, amult)

	tran = makea(naa,
				nzz,
				a,
				colidx,
				rowstr,
				firstrow,
				lastrow,
				arow,
				acol,
				aelt,
				iv,
				tran)

	if use_compiled_types():
		shift_column_indices_types(colidx, rowstr, lastrow - firstrow + 1, firstcol)
	else:
		shift_column_indices(colidx, rowstr, lastrow - firstrow + 1, firstcol)

	if use_compiled_types():
		rnorm = conj_grad_types(colidx, rowstr, x, z, a, p, q, r, naa, firstrow, lastrow, firstcol, lastcol)
		norm_temp1, norm_temp2, zeta = normalize_z_types(x, z, 0, lastcol - firstcol + 1, npbparams.SHIFT)
	else:
		rnorm = conj_grad(colidx, rowstr, x, z, a, p, q, r, naa, firstrow, lastrow, firstcol, lastcol)
		norm_temp1, norm_temp2, zeta = normalize_z(x, z, False, lastcol - firstcol + 1, npbparams.SHIFT)

	if use_compiled_types():
		fill_vector_types(x, 1.0, npbparams.NA + 1)
	else:
		fill_vector(x, 1.0, npbparams.NA + 1)
	zeta = 0.0

	c_timers.timer_stop(T_INIT)
	print(" Initialization time = %15.3f seconds" % (c_timers.timer_read(T_INIT)))

	c_timers.timer_start(T_BENCH)

	for it in range(1, npbparams.NITER + 1):
		if timeron:
			c_timers.timer_start(T_CONJ_GRAD)
		if use_compiled_types():
			rnorm = conj_grad_types(colidx, rowstr, x, z, a, p, q, r, naa, firstrow, lastrow, firstcol, lastcol)
		else:
			rnorm = conj_grad(colidx, rowstr, x, z, a, p, q, r, naa, firstrow, lastrow, firstcol, lastcol)
		if timeron:
			c_timers.timer_stop(T_CONJ_GRAD)

		if use_compiled_types():
			norm_temp1, norm_temp2, zeta = normalize_z_types(x, z, 1, lastcol - firstcol + 1, npbparams.SHIFT)
		else:
			norm_temp1, norm_temp2, zeta = normalize_z(x, z, True, lastcol - firstcol + 1, npbparams.SHIFT)
		if it == 1:
			print("\n   iteration           ||r||                 zeta")
		print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

	c_timers.timer_stop(T_BENCH)

	t = c_timers.timer_read(T_BENCH)
	print(" Benchmark completed")

	verified = False
	epsilon = 1.0e-10
	err = abs(zeta - zeta_verify_value) / zeta_verify_value
	if err <= epsilon:
		verified = True
		print(" VERIFICATION SUCCESSFUL")
		print(" Zeta is    %20.13e" % (zeta))
		print(" Error is   %20.13e" % (err))
	else:
		print(" VERIFICATION FAILED")
		print(" Zeta                %20.13e" % (zeta))
		print(" The correct zeta is %20.13e" % (zeta_verify_value))

	mflops = 0.0
	if t != 0.0:
		mflops = ((2.0 * npbparams.NITER * npbparams.NA)
			* (3.0 + (npbparams.NONZER * (npbparams.NONZER + 1))
				+ 25.0
				* (5.0 + (npbparams.NONZER * (npbparams.NONZER + 1))) + 3.0)
			/ t / 1000000.0)

	c_print_results.c_print_results("CG",
			npbparams.CLASS,
			npbparams.NA,
			0,
			0,
			npbparams.NITER,
			t,
			mflops,
			"          floating point",
			verified)

	if timeron:
		tmax = c_timers.timer_read(T_BENCH)
		if tmax == 0.0:
			tmax = 1.0
		print("  SECTION   Time (secs)")
		for i in range(T_LAST):
			t = c_timers.timer_read(i)
			if i == T_INIT:
				print("  %8s:%9.3f" % (t_names[i], t))
			else:
				print("  %8s:%9.3f  (%6.2f%%)" % (t_names[i], t, t * 100.0 / tmax))
				if i == T_CONJ_GRAD:
					t = tmax - t
					print("    --> %8s:%9.3f  (%6.2f%%)" % ("rest", t, t * 100.0 / tmax))
#END main()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-SER CG')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
	parser.add_argument("-m", "--mode", type=int, default=1, help="Mode: 0=pure, 1=hybrid, 2=compiled, 3=compiled with types")
	args = parser.parse_args()

	set_omp_mode(args.mode)
	set_omp_threads(args.threads)
	npbparams.set_cg_info(args.CLASS)
	set_global_variables()

	main()
