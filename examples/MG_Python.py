# ------------------------------------------------------------------------------
#
# The original NPB 3.4.1 version was written in Fortran and belongs to:
# 	http://www.nas.nasa.gov/Software/NPB/
#
# Authors of the Fortran code:
#	E. Barszcz
#	P. Frederickson
#	A. Woo
#	M. Yarrow
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
import math
import numpy

from omputils import omp, omp_pure, use_pure, use_compiled, use_compiled_types, set_omp_threads, set_omp_mode

if use_pure():
    omp = omp_pure

try:
	import cython
except ImportError:
	class cython:
		double = []
		int = []

# Local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import npbparams
import c_timers
import c_print_results


NM = 0
NV = 0
NR = 0
MAXLEVEL = 0
M = 0
MM = 10
A = pow(5.0, 13.0)
X = 314159265.0
T_INIT = 0
T_BENCH = 1
T_MG3P = 2
T_PSINV = 3
T_RESID = 4
T_RESID2 = 5
T_RPRJ3 = 6
T_INTERP = 7
T_NORM2 = 8
T_COMM3 = 9
T_LAST = 10

nx = None
ny = None
nz = None
m1 = None
m2 = None
m3 = None
ir = None
debug_vec = numpy.repeat(npbparams.DEBUG_DEFAULT, 8)
u = None
v = None
r = None

is1 = 0
is2 = 0
is3 = 0
ie1 = 0
ie2 = 0
ie3 = 0
lt = 0
lb = 0

timeron = False

def set_global_variables():
	global NM, NV, NR, MAXLEVEL, M
	global nx, ny, nz, m1, m2, m3, ir
	global u, v, r

	NM = 2 + (1 << npbparams.LM)
	NV = npbparams.ONE * (2 + (1 << npbparams.NDIM1)) * (2 + (1 << npbparams.NDIM2)) * (2 + (1 << npbparams.NDIM3))
	NR = int((NV + NM * NM + 5 * NM + 7 * npbparams.LM + 6) / 7) * 8
	MAXLEVEL = npbparams.LT_DEFAULT + 1
	M = NM + 1

	nx = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
	ny = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
	nz = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
	m1 = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
	m2 = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
	m3 = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)
	ir = numpy.empty(MAXLEVEL + 1, dtype=numpy.int32)

	u = numpy.empty(NR, dtype=numpy.float64)
	v = numpy.empty(NV, dtype=numpy.float64)
	r = numpy.empty(NR, dtype=numpy.float64)
# END set_global_variables()


@omp(compile=use_compiled())
def bubble(ten, j1, j2, j3, m, ind):
	if ind == 1:
		for i in range(m - 1):
			idx = ind * m + i
			next_idx = idx + 1
			if ten[idx] > ten[next_idx]:
				temp = ten[next_idx]
				ten[next_idx] = ten[idx]
				ten[idx] = temp

				j_temp = j1[next_idx]
				j1[next_idx] = j1[idx]
				j1[idx] = j_temp

				j_temp = j2[next_idx]
				j2[next_idx] = j2[idx]
				j2[idx] = j_temp

				j_temp = j3[next_idx]
				j3[next_idx] = j3[idx]
				j3[idx] = j_temp
			else:
				return
	else:
		for i in range(m - 1):
			idx = ind * m + i
			next_idx = idx + 1
			if ten[idx] < ten[next_idx]:
				temp = ten[next_idx]
				ten[next_idx] = ten[idx]
				ten[idx] = temp

				j_temp = j1[next_idx]
				j1[next_idx] = j1[idx]
				j1[idx] = j_temp

				j_temp = j2[next_idx]
				j2[next_idx] = j2[idx]
				j2[idx] = j_temp

				j_temp = j3[next_idx]
				j3[next_idx] = j3[idx]
				j3[idx] = j_temp
			else:
				return
# END bubble()


@omp(compile=use_compiled_types())
def bubble_types(ten_arg, j1_arg, j2_arg, j3_arg, m: int, ind: int):
	ten: cython.double[:] = ten_arg
	j1: cython.int[:] = j1_arg
	j2: cython.int[:] = j2_arg
	j3: cython.int[:] = j3_arg
	i: int
	idx: int
	next_idx: int
	temp: float
	j_temp: int

	if ind == 1:
		for i in range(m - 1):
			idx = ind * m + i
			next_idx = idx + 1
			if ten[idx] > ten[next_idx]:
				temp = ten[next_idx]
				ten[next_idx] = ten[idx]
				ten[idx] = temp

				j_temp = j1[next_idx]
				j1[next_idx] = j1[idx]
				j1[idx] = j_temp

				j_temp = j2[next_idx]
				j2[next_idx] = j2[idx]
				j2[idx] = j_temp

				j_temp = j3[next_idx]
				j3[next_idx] = j3[idx]
				j3[idx] = j_temp
			else:
				return
	else:
		for i in range(m - 1):
			idx = ind * m + i
			next_idx = idx + 1
			if ten[idx] < ten[next_idx]:
				temp = ten[next_idx]
				ten[next_idx] = ten[idx]
				ten[idx] = temp

				j_temp = j1[next_idx]
				j1[next_idx] = j1[idx]
				j1[idx] = j_temp

				j_temp = j2[next_idx]
				j2[next_idx] = j2[idx]
				j2[idx] = j_temp

				j_temp = j3[next_idx]
				j3[next_idx] = j3[idx]
				j3[idx] = j_temp
			else:
				return
# END bubble_types()


@omp(compile=use_compiled())
def randlc_seed(x, a):
	r23 = pow(0.5, 23.0)
	r46 = pow(0.5, 46.0)
	t23 = pow(2.0, 23.0)
	t46 = pow(2.0, 46.0)

	t1 = r23 * a
	a1 = int(t1)
	a2 = a - t23 * a1
	t1 = r23 * x
	x1 = int(t1)
	x2 = x - t23 * x1
	t1 = a1 * x2 + a2 * x1
	t2 = int(r23 * t1)
	z = t1 - t23 * t2
	t3 = t23 * z + a2 * x2
	t4 = int(r46 * t3)
	return t3 - t46 * t4
# END randlc_seed()


@omp(compile=use_compiled_types())
def randlc_seed_types(x: float, a: float):
	r23: float = pow(0.5, 23.0)
	r46: float = pow(0.5, 46.0)
	t23: float = pow(2.0, 23.0)
	t46: float = pow(2.0, 46.0)
	t1: float
	t2: int
	t3: float
	t4: int
	a1: int
	a2: float
	x1: int
	x2: float
	z: float

	t1 = r23 * a
	a1 = int(t1)
	a2 = a - t23 * a1
	t1 = r23 * x
	x1 = int(t1)
	x2 = x - t23 * x1
	t1 = a1 * x2 + a2 * x1
	t2 = int(r23 * t1)
	z = t1 - t23 * t2
	t3 = t23 * z + a2 * x2
	t4 = int(r46 * t3)
	return t3 - t46 * t4
# END randlc_seed_types()


@omp(compile=use_compiled())
def vranlc_mg(n, x_seed, a, y):
	r23 = pow(0.5, 23.0)
	r46 = pow(0.5, 46.0)
	t23 = pow(2.0, 23.0)
	t46 = pow(2.0, 46.0)

	t1 = r23 * a
	a1 = int(t1)
	a2 = a - t23 * a1
	x = x_seed

	for i in range(n):
		t1 = r23 * x
		x1 = int(t1)
		x2 = x - t23 * x1
		t1 = a1 * x2 + a2 * x1
		t2 = int(r23 * t1)
		z = t1 - t23 * t2
		t3 = t23 * z + a2 * x2
		t4 = int(r46 * t3)
		x = t3 - t46 * t4
		y[i] = r46 * x

	return x
# END vranlc_mg()


@omp(compile=use_compiled_types())
def vranlc_mg_types(n: int, x_seed: float, a: float, y_arg):
	y: cython.double[:] = y_arg
	r23: float = pow(0.5, 23.0)
	r46: float = pow(0.5, 46.0)
	t23: float = pow(2.0, 23.0)
	t46: float = pow(2.0, 46.0)
	t1: float
	t2: int
	t3: float
	t4: int
	a1: int
	a2: float
	x1: int
	x2: float
	z: float
	x: float = x_seed
	i: int

	t1 = r23 * a
	a1 = int(t1)
	a2 = a - t23 * a1

	for i in range(n):
		t1 = r23 * x
		x1 = int(t1)
		x2 = x - t23 * x1
		t1 = a1 * x2 + a2 * x1
		t2 = int(r23 * t1)
		z = t1 - t23 * t2
		t3 = t23 * z + a2 * x2
		t4 = int(r46 * t3)
		x = t3 - t46 * t4
		y[i] = r46 * x

	return x
# END vranlc_mg_types()


@omp(compile=use_compiled())
def power(a, n):
	power_value = 1.0
	nj = n
	aj = a

	while nj != 0:
		if (nj % 2) == 1:
			power_value = randlc_seed(power_value, aj)
		aj = randlc_seed(aj, aj)
		nj = nj // 2

	return power_value
# END power()


@omp(compile=use_compiled_types())
def power_types(a: float, n: int):
	power_value: float = 1.0
	nj: int = n
	aj: float = a

	while nj != 0:
		if (nj % 2) == 1:
			power_value = randlc_seed_types(power_value, aj)
		aj = randlc_seed_types(aj, aj)
		nj = nj // 2

	return power_value
# END power_types()


def setup(k):
	global nx, ny, nz, m1, m2, m3, ir
	global is1, is2, is3, ie1, ie2, ie3

	mi = numpy.empty((MAXLEVEL + 1, 3), dtype=numpy.int32)
	ng = numpy.empty((MAXLEVEL + 1, 3), dtype=numpy.int32)

	ng[lt][0] = nx[lt]
	ng[lt][1] = ny[lt]
	ng[lt][2] = nz[lt]
	for ax in range(3):
		for level in range(lt - 1, 0, -1):
			ng[level][ax] = ng[level + 1][ax] // 2

	for level in range(lt, 0, -1):
		nx[level] = ng[level][0]
		ny[level] = ng[level][1]
		nz[level] = ng[level][2]

	for level in range(lt, 0, -1):
		for ax in range(3):
			mi[level][ax] = 2 + ng[level][ax]
		m1[level] = mi[level][0]
		m2[level] = mi[level][1]
		m3[level] = mi[level][2]

	level = lt
	is1 = 2 + ng[level][0] - ng[lt][0]
	ie1 = 1 + ng[level][0]
	n1 = 3 + ie1 - is1
	is2 = 2 + ng[level][1] - ng[lt][1]
	ie2 = 1 + ng[level][1]
	n2 = 3 + ie2 - is2
	is3 = 2 + ng[level][2] - ng[lt][2]
	ie3 = 1 + ng[level][2]
	n3 = 3 + ie3 - is3

	ir[lt] = 0
	for j in range(lt - 1, 0, -1):
		ir[j] = ir[j + 1] + npbparams.ONE * m1[j + 1] * m2[j + 1] * m3[j + 1]

	if debug_vec[1] >= 1:
		print(" in setup")
		print("   k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3")
		print("%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d"
			% (level, lt, ng[level][0], ng[level][1], ng[level][2], n1, n2, n3, is1, is2, is3, ie1, ie2, ie3))

	return n1, n2, n3
# END setup()


def norm2u3_serial(pointer_r, n1, n2, n3, nx_aux, ny_aux, nz_aux):
	dn = 1.0 * nx_aux * ny_aux * nz_aux
	s = 0.0
	rnmu = 0.0
	for i3 in range(1, n3 - 1):
		for i2 in range(1, n2 - 1):
			for i1 in range(1, n1 - 1):
				value = pointer_r[(i3 * n2 + i2) * n1 + i1]
				s = s + value * value
				abs_value = abs(value)
				if abs_value > rnmu:
					rnmu = abs_value

	return math.sqrt(s / dn), rnmu
# END norm2u3_serial()


def rep_nrm(pointer_u, n1, n2, n3, title, kk):
	rnm2, rnmu = norm2u3_serial(pointer_u, n1, n2, n3, nx[kk], ny[kk], nz[kk])
	print(" Level%2d in %8s: norms =%21.14e%21.14e" % (kk, title, rnm2, rnmu))
# END rep_nrm()


def showall(pointer_z, n1, n2, n3):
	m1_show = min(n1, 18)
	m2_show = min(n2, 14)
	m3_show = min(n3, 18)

	print()
	for i3 in range(m3_show):
		for i2 in range(m2_show):
			for i1 in range(m1_show):
				print("%6.3f" % pointer_z[(i3 * n2 + i2) * n1 + i1], end="")
			print()
		print(" - - - - - - - ")
	print()
# END showall()


@omp(compile=use_compiled())
def comm3(pointer_u, n1, n2, n3, kk):
	u_local = pointer_u

	if timeron:
		with omp("single"):
			c_timers.timer_start(T_COMM3)

	with omp("for"):
		for i3 in range(1, n3 - 1):
			for i2 in range(1, n2 - 1):
				u_local[(i3 * n2 + i2) * n1 + 0] = u_local[(i3 * n2 + i2) * n1 + (n1 - 2)]
				u_local[(i3 * n2 + i2) * n1 + (n1 - 1)] = u_local[(i3 * n2 + i2) * n1 + 1]
			for i1 in range(n1):
				u_local[(i3 * n2 + 0) * n1 + i1] = u_local[(i3 * n2 + (n2 - 2)) * n1 + i1]
				u_local[(i3 * n2 + (n2 - 1)) * n1 + i1] = u_local[(i3 * n2 + 1) * n1 + i1]

	with omp("for"):
		for i2 in range(n2):
			for i1 in range(n1):
				u_local[(0 * n2 + i2) * n1 + i1] = u_local[((n3 - 2) * n2 + i2) * n1 + i1]
				u_local[((n3 - 1) * n2 + i2) * n1 + i1] = u_local[(1 * n2 + i2) * n1 + i1]

	if timeron:
		with omp("single"):
			c_timers.timer_stop(T_COMM3)
# END comm3()


@omp(compile=use_compiled_types())
def comm3_types(pointer_u, n1: int, n2: int, n3: int, kk: int):
	u_local: cython.double[:] = pointer_u
	i1: int
	i2: int
	i3: int

	if timeron:
		with omp("single"):
			c_timers.timer_start(T_COMM3)

	with omp("for"):
		for i3 in range(1, n3 - 1):
			for i2 in range(1, n2 - 1):
				u_local[(i3 * n2 + i2) * n1 + 0] = u_local[(i3 * n2 + i2) * n1 + (n1 - 2)]
				u_local[(i3 * n2 + i2) * n1 + (n1 - 1)] = u_local[(i3 * n2 + i2) * n1 + 1]
			for i1 in range(n1):
				u_local[(i3 * n2 + 0) * n1 + i1] = u_local[(i3 * n2 + (n2 - 2)) * n1 + i1]
				u_local[(i3 * n2 + (n2 - 1)) * n1 + i1] = u_local[(i3 * n2 + 1) * n1 + i1]

	with omp("for"):
		for i2 in range(n2):
			for i1 in range(n1):
				u_local[(0 * n2 + i2) * n1 + i1] = u_local[((n3 - 2) * n2 + i2) * n1 + i1]
				u_local[((n3 - 1) * n2 + i2) * n1 + i1] = u_local[(1 * n2 + i2) * n1 + i1]

	if timeron:
		with omp("single"):
			c_timers.timer_stop(T_COMM3)
# END comm3_types()


@omp(compile=use_compiled())
def zero3(pointer_z, n1, n2, n3):
	z_local = pointer_z

	with omp("parallel"):
		with omp("for"):
			for i3 in range(n3):
				for i2 in range(n2):
					for i1 in range(n1):
						z_local[(i3 * n2 + i2) * n1 + i1] = 0.0
# END zero3()


@omp(compile=use_compiled_types())
def zero3_types(pointer_z, n1: int, n2: int, n3: int):
	z_local: cython.double[:] = pointer_z
	i1: int
	i2: int
	i3: int

	with omp("parallel"):
		with omp("for"):
			for i3 in range(n3):
				for i2 in range(n2):
					for i1 in range(n1):
						z_local[(i3 * n2 + i2) * n1 + i1] = 0.0
# END zero3_types()


@omp(compile=use_compiled())
def norm2u3(pointer_r, n1, n2, n3, nx_aux, ny_aux, nz_aux):
	r_local = pointer_r
	dn = 1.0 * nx_aux * ny_aux * nz_aux
	s = 0.0
	rnmu_values = numpy.zeros(1, dtype=numpy.float64)

	if timeron:
		c_timers.timer_start(T_NORM2)
	with omp("parallel"):
		local_rnmu = 0.0

		with omp("for reduction(+:s)"):
			for i3 in range(1, n3 - 1):
				for i2 in range(1, n2 - 1):
					for i1 in range(1, n1 - 1):
						value = r_local[(i3 * n2 + i2) * n1 + i1]
						s = s + value * value
						abs_value = abs(value)
						if abs_value > local_rnmu:
							local_rnmu = abs_value

		with omp("critical"):
			if local_rnmu > rnmu_values[0]:
				rnmu_values[0] = local_rnmu

	rnm2 = math.sqrt(s / dn)
	rnmu = rnmu_values[0]

	if timeron:
		c_timers.timer_stop(T_NORM2)

	return rnm2, rnmu
# END norm2u3()


@omp(compile=use_compiled_types())
def norm2u3_types(pointer_r, n1: int, n2: int, n3: int, nx_aux: int, ny_aux: int, nz_aux: int):
	r_local: cython.double[:] = pointer_r
	dn: float = 1.0 * nx_aux * ny_aux * nz_aux
	s: float = 0.0
	rnmu_values = numpy.zeros(1, dtype=numpy.float64)
	rnmu_view: cython.double[:] = rnmu_values
	rnm2: float
	rnmu: float
	i1: int
	i2: int
	i3: int
	value: float
	abs_value: float
	local_rnmu: float

	if timeron:
		c_timers.timer_start(T_NORM2)
	with omp("parallel private(local_rnmu,value,abs_value,i1,i2,i3)"):
		local_rnmu = 0.0

		with omp("for reduction(+:s)"):
			for i3 in range(1, n3 - 1):
				for i2 in range(1, n2 - 1):
					for i1 in range(1, n1 - 1):
						value = r_local[(i3 * n2 + i2) * n1 + i1]
						s = s + value * value
						abs_value = abs(value)
						if abs_value > local_rnmu:
							local_rnmu = abs_value

		with omp("critical"):
			if local_rnmu > rnmu_view[0]:
				rnmu_view[0] = local_rnmu

	rnm2 = math.sqrt(s / dn)
	rnmu = rnmu_view[0]

	if timeron:
		c_timers.timer_stop(T_NORM2)

	return rnm2, rnmu
# END norm2u3_types()


@omp(compile=use_compiled())
def interp(pointer_z, mm1, mm2, mm3, pointer_u, n1, n2, n3, k):
	z_local = pointer_z
	u_local = pointer_u

	with omp("parallel"):
		z1 = numpy.empty(mm1, dtype=numpy.float64)
		z2 = numpy.empty(mm1, dtype=numpy.float64)
		z3 = numpy.empty(mm1, dtype=numpy.float64)
		z1_view = z1
		z2_view = z2
		z3_view = z3

		if timeron:
			with omp("single"):
				c_timers.timer_start(T_INTERP)

		if n1 != 3 and n2 != 3 and n3 != 3:
			with omp("for"):
				for i3 in range(mm3 - 1):
					for i2 in range(mm2 - 1):
						for i1 in range(mm1):
							z1_view[i1] = z_local[(i3 * mm2 + (i2 + 1)) * mm1 + i1] + z_local[(i3 * mm2 + i2) * mm1 + i1]
							z2_view[i1] = z_local[((i3 + 1) * mm2 + i2) * mm1 + i1] + z_local[(i3 * mm2 + i2) * mm1 + i1]
							z3_view[i1] = (z_local[((i3 + 1) * mm2 + (i2 + 1)) * mm1 + i1]
								+ z_local[((i3 + 1) * mm2 + i2) * mm1 + i1]
								+ z1_view[i1])
						for i1 in range(mm1 - 1):
							u_local[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1)] = (
								u_local[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1)]
								+ z_local[(i3 * mm2 + i2) * mm1 + i1]
							)
							u_local[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1 + 1)] = (
								u_local[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1 + 1)]
								+ 0.5 * (z_local[(i3 * mm2 + i2) * mm1 + (i1 + 1)] + z_local[(i3 * mm2 + i2) * mm1 + i1])
							)
						for i1 in range(mm1 - 1):
							u_local[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] = (
								u_local[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] + 0.5 * z1_view[i1]
							)
							u_local[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1 + 1)] = (
								u_local[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1 + 1)] + 0.25 * (z1_view[i1] + z1_view[i1 + 1])
							)
						for i1 in range(mm1 - 1):
							u_local[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1)] = (
								u_local[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1)] + 0.5 * z2_view[i1]
							)
							u_local[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1 + 1)] = (
								u_local[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1 + 1)] + 0.25 * (z2_view[i1] + z2_view[i1 + 1])
							)
						for i1 in range(mm1 - 1):
							u_local[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] = (
								u_local[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] + 0.25 * z3_view[i1]
							)
							u_local[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1 + 1)] = (
								u_local[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1 + 1)] + 0.125 * (z3_view[i1] + z3_view[i1 + 1])
							)
		else:
			if n1 == 3:
				d1 = 2
				t1 = 1
			else:
				d1 = 1
				t1 = 0

			if n2 == 3:
				d2 = 2
				t2 = 1
			else:
				d2 = 1
				t2 = 0

			if n3 == 3:
				d3 = 2
				t3 = 1
			else:
				d3 = 1
				t3 = 0

			with omp("for"):
				for i3 in range(d3, mm3):
					for i2 in range(d2, mm2):
						for i1 in range(d1, mm1):
							u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)] = (
								u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)]
								+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
							)
						for i1 in range(1, mm1):
							u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
								u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)]
								+ 0.5 * (z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1] + z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)])
							)
					for i2 in range(1, mm2):
						for i1 in range(d1, mm1):
							u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)] = (
								u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)]
								+ 0.5 * (z_local[((i3 - 1) * mm2 + i2) * mm1 + (i1 - 1)] + z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)])
							)
						for i1 in range(1, mm1):
							u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
								u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)]
								+ 0.25 * (
									z_local[((i3 - 1) * mm2 + i2) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
								)
							)

			with omp("for"):
				for i3 in range(1, mm3):
					for i2 in range(d2, mm2):
						for i1 in range(d1, mm1):
							u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)] = (
								u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)]
								+ 0.5 * (z_local[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)] + z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)])
							)
						for i1 in range(1, mm1):
							u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
								u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)]
								+ 0.25 * (
									z_local[(i3 * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
								)
							)
					for i2 in range(1, mm2):
						for i1 in range(d1, mm1):
							u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)] = (
								u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)]
								+ 0.25 * (
									z_local[(i3 * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
								)
							)
						for i1 in range(1, mm1):
							u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
								u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)]
								+ 0.125 * (
									z_local[(i3 * mm2 + i2) * mm1 + i1]
									+ z_local[(i3 * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[(i3 * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + i2) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
								)
							)

		if timeron:
			with omp("single"):
				c_timers.timer_stop(T_INTERP)

		with omp("single"):
			if debug_vec[0] >= 1:
				rep_nrm(z_local, mm1, mm2, mm3, "z: inter", k - 1)
				rep_nrm(u_local, n1, n2, n3, "u: inter", k)
			if debug_vec[5] >= k:
				showall(z_local, mm1, mm2, mm3)
				showall(u_local, n1, n2, n3)
# END interp()


@omp(compile=use_compiled_types())
def interp_types(pointer_z, mm1: int, mm2: int, mm3: int, pointer_u, n1: int, n2: int, n3: int, k: int):
	z_local: cython.double[:] = pointer_z
	u_local: cython.double[:] = pointer_u
	i1: int
	i2: int
	i3: int
	d1: int
	d2: int
	d3: int
	t1: int
	t2: int
	t3: int

	with omp("parallel"):
		z1 = numpy.empty(mm1, dtype=numpy.float64)
		z2 = numpy.empty(mm1, dtype=numpy.float64)
		z3 = numpy.empty(mm1, dtype=numpy.float64)
		z1_view: cython.double[:] = z1
		z2_view: cython.double[:] = z2
		z3_view: cython.double[:] = z3

		if timeron:
			with omp("single"):
				c_timers.timer_start(T_INTERP)

		if n1 != 3 and n2 != 3 and n3 != 3:
			with omp("for"):
				for i3 in range(mm3 - 1):
					for i2 in range(mm2 - 1):
						for i1 in range(mm1):
							z1_view[i1] = z_local[(i3 * mm2 + (i2 + 1)) * mm1 + i1] + z_local[(i3 * mm2 + i2) * mm1 + i1]
							z2_view[i1] = z_local[((i3 + 1) * mm2 + i2) * mm1 + i1] + z_local[(i3 * mm2 + i2) * mm1 + i1]
							z3_view[i1] = (z_local[((i3 + 1) * mm2 + (i2 + 1)) * mm1 + i1]
								+ z_local[((i3 + 1) * mm2 + i2) * mm1 + i1]
								+ z1_view[i1])
						for i1 in range(mm1 - 1):
							u_local[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1)] = (
								u_local[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1)]
								+ z_local[(i3 * mm2 + i2) * mm1 + i1]
							)
							u_local[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1 + 1)] = (
								u_local[((2 * i3) * n2 + (2 * i2)) * n1 + (2 * i1 + 1)]
								+ 0.5 * (z_local[(i3 * mm2 + i2) * mm1 + (i1 + 1)] + z_local[(i3 * mm2 + i2) * mm1 + i1])
							)
						for i1 in range(mm1 - 1):
							u_local[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] = (
								u_local[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] + 0.5 * z1_view[i1]
							)
							u_local[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1 + 1)] = (
								u_local[((2 * i3) * n2 + (2 * i2 + 1)) * n1 + (2 * i1 + 1)] + 0.25 * (z1_view[i1] + z1_view[i1 + 1])
							)
						for i1 in range(mm1 - 1):
							u_local[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1)] = (
								u_local[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1)] + 0.5 * z2_view[i1]
							)
							u_local[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1 + 1)] = (
								u_local[((2 * i3 + 1) * n2 + (2 * i2)) * n1 + (2 * i1 + 1)] + 0.25 * (z2_view[i1] + z2_view[i1 + 1])
							)
						for i1 in range(mm1 - 1):
							u_local[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] = (
								u_local[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1)] + 0.25 * z3_view[i1]
							)
							u_local[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1 + 1)] = (
								u_local[((2 * i3 + 1) * n2 + (2 * i2 + 1)) * n1 + (2 * i1 + 1)] + 0.125 * (z3_view[i1] + z3_view[i1 + 1])
							)
		else:
			if n1 == 3:
				d1 = 2
				t1 = 1
			else:
				d1 = 1
				t1 = 0

			if n2 == 3:
				d2 = 2
				t2 = 1
			else:
				d2 = 1
				t2 = 0

			if n3 == 3:
				d3 = 2
				t3 = 1
			else:
				d3 = 1
				t3 = 0

			with omp("for"):
				for i3 in range(d3, mm3):
					for i2 in range(d2, mm2):
						for i1 in range(d1, mm1):
							u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)] = (
								u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)]
								+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
							)
						for i1 in range(1, mm1):
							u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
								u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)]
								+ 0.5 * (z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1] + z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)])
							)
					for i2 in range(1, mm2):
						for i1 in range(d1, mm1):
							u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)] = (
								u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)]
								+ 0.5 * (z_local[((i3 - 1) * mm2 + i2) * mm1 + (i1 - 1)] + z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)])
							)
						for i1 in range(1, mm1):
							u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
								u_local[((2 * i3 - d3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)]
								+ 0.25 * (
									z_local[((i3 - 1) * mm2 + i2) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
								)
							)

			with omp("for"):
				for i3 in range(1, mm3):
					for i2 in range(d2, mm2):
						for i1 in range(d1, mm1):
							u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)] = (
								u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - d1 - 1)]
								+ 0.5 * (z_local[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)] + z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)])
							)
						for i1 in range(1, mm1):
							u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
								u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - d2 - 1)) * n1 + (2 * i1 - t1 - 1)]
								+ 0.25 * (
									z_local[(i3 * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
								)
							)
					for i2 in range(1, mm2):
						for i1 in range(d1, mm1):
							u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)] = (
								u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - d1 - 1)]
								+ 0.25 * (
									z_local[(i3 * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
								)
							)
						for i1 in range(1, mm1):
							u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)] = (
								u_local[((2 * i3 - t3 - 1) * n2 + (2 * i2 - t2 - 1)) * n1 + (2 * i1 - t1 - 1)]
								+ 0.125 * (
									z_local[(i3 * mm2 + i2) * mm1 + i1]
									+ z_local[(i3 * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[(i3 * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[(i3 * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + i2) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + i1]
									+ z_local[((i3 - 1) * mm2 + i2) * mm1 + (i1 - 1)]
									+ z_local[((i3 - 1) * mm2 + (i2 - 1)) * mm1 + (i1 - 1)]
								)
							)

		if timeron:
			with omp("single"):
				c_timers.timer_stop(T_INTERP)

		with omp("single"):
			if debug_vec[0] >= 1:
				rep_nrm(z_local, mm1, mm2, mm3, "z: inter", k - 1)
				rep_nrm(u_local, n1, n2, n3, "u: inter", k)
			if debug_vec[5] >= k:
				showall(z_local, mm1, mm2, mm3)
				showall(u_local, n1, n2, n3)
# END interp_types()


@omp(compile=use_compiled())
def psinv(pointer_r, pointer_u, n1, n2, n3, c, k):
	r_local = pointer_r
	u_local = pointer_u
	c_local = c

	with omp("parallel"):
		r1 = numpy.empty(n1, dtype=numpy.float64)
		r2 = numpy.empty(n1, dtype=numpy.float64)
		r1_view = r1
		r2_view = r2

		if timeron:
			with omp("single"):
				c_timers.timer_start(T_PSINV)

		with omp("for"):
			for i3 in range(1, n3 - 1):
				for i2 in range(1, n2 - 1):
					for i1 in range(n1):
						r1_view[i1] = (r_local[(i3 * n2 + (i2 - 1)) * n1 + i1]
							+ r_local[(i3 * n2 + (i2 + 1)) * n1 + i1]
							+ r_local[((i3 - 1) * n2 + i2) * n1 + i1]
							+ r_local[((i3 + 1) * n2 + i2) * n1 + i1])
						r2_view[i1] = (r_local[((i3 - 1) * n2 + (i2 - 1)) * n1 + i1]
							+ r_local[((i3 - 1) * n2 + (i2 + 1)) * n1 + i1]
							+ r_local[((i3 + 1) * n2 + (i2 - 1)) * n1 + i1]
							+ r_local[((i3 + 1) * n2 + (i2 + 1)) * n1 + i1])
					for i1 in range(1, n1 - 1):
						u_local[(i3 * n2 + i2) * n1 + i1] = (
							u_local[(i3 * n2 + i2) * n1 + i1]
							+ c_local[0] * r_local[(i3 * n2 + i2) * n1 + i1]
							+ c_local[1] * (r_local[(i3 * n2 + i2) * n1 + (i1 - 1)] + r_local[(i3 * n2 + i2) * n1 + (i1 + 1)] + r1_view[i1])
							+ c_local[2] * (r2_view[i1] + r1_view[i1 - 1] + r1_view[i1 + 1])
						)

		if timeron:
			with omp("single"):
				c_timers.timer_stop(T_PSINV)

		comm3(u_local, n1, n2, n3, k)

		with omp("single"):
			if debug_vec[0] >= 1:
				rep_nrm(u_local, n1, n2, n3, "   psinv", k)
			if debug_vec[3] >= k:
				showall(u_local, n1, n2, n3)
# END psinv()


@omp(compile=use_compiled_types())
def psinv_types(pointer_r, pointer_u, n1: int, n2: int, n3: int, c, k: int):
	r_local: cython.double[:] = pointer_r
	u_local: cython.double[:] = pointer_u
	c_local: cython.double[:] = c
	i1: int
	i2: int
	i3: int

	with omp("parallel"):
		r1 = numpy.empty(n1, dtype=numpy.float64)
		r2 = numpy.empty(n1, dtype=numpy.float64)
		r1_view: cython.double[:] = r1
		r2_view: cython.double[:] = r2

		if timeron:
			with omp("single"):
				c_timers.timer_start(T_PSINV)

		with omp("for"):
			for i3 in range(1, n3 - 1):
				for i2 in range(1, n2 - 1):
					for i1 in range(n1):
						r1_view[i1] = (r_local[(i3 * n2 + (i2 - 1)) * n1 + i1]
							+ r_local[(i3 * n2 + (i2 + 1)) * n1 + i1]
							+ r_local[((i3 - 1) * n2 + i2) * n1 + i1]
							+ r_local[((i3 + 1) * n2 + i2) * n1 + i1])
						r2_view[i1] = (r_local[((i3 - 1) * n2 + (i2 - 1)) * n1 + i1]
							+ r_local[((i3 - 1) * n2 + (i2 + 1)) * n1 + i1]
							+ r_local[((i3 + 1) * n2 + (i2 - 1)) * n1 + i1]
							+ r_local[((i3 + 1) * n2 + (i2 + 1)) * n1 + i1])
					for i1 in range(1, n1 - 1):
						u_local[(i3 * n2 + i2) * n1 + i1] = (
							u_local[(i3 * n2 + i2) * n1 + i1]
							+ c_local[0] * r_local[(i3 * n2 + i2) * n1 + i1]
							+ c_local[1] * (r_local[(i3 * n2 + i2) * n1 + (i1 - 1)] + r_local[(i3 * n2 + i2) * n1 + (i1 + 1)] + r1_view[i1])
							+ c_local[2] * (r2_view[i1] + r1_view[i1 - 1] + r1_view[i1 + 1])
						)

		if timeron:
			with omp("single"):
				c_timers.timer_stop(T_PSINV)

		comm3_types(u_local, n1, n2, n3, k)

		with omp("single"):
			if debug_vec[0] >= 1:
				rep_nrm(u_local, n1, n2, n3, "   psinv", k)
			if debug_vec[3] >= k:
				showall(u_local, n1, n2, n3)
# END psinv_types()


@omp(compile=use_compiled())
def resid(pointer_u, pointer_v, pointer_r, n1, n2, n3, a, k):
	u_local = pointer_u
	v_local = pointer_v
	r_local = pointer_r
	a_local = a

	with omp("parallel"):
		u1 = numpy.empty(n1, dtype=numpy.float64)
		u2 = numpy.empty(n1, dtype=numpy.float64)
		u1_view = u1
		u2_view = u2

		if timeron:
			with omp("single"):
				c_timers.timer_start(T_RESID)

		with omp("for"):
			for i3 in range(1, n3 - 1):
				for i2 in range(1, n2 - 1):
					for i1 in range(n1):
						u1_view[i1] = (u_local[(i3 * n2 + (i2 - 1)) * n1 + i1]
							+ u_local[(i3 * n2 + (i2 + 1)) * n1 + i1]
							+ u_local[((i3 - 1) * n2 + i2) * n1 + i1]
							+ u_local[((i3 + 1) * n2 + i2) * n1 + i1])
						u2_view[i1] = (u_local[((i3 - 1) * n2 + (i2 - 1)) * n1 + i1]
							+ u_local[((i3 - 1) * n2 + (i2 + 1)) * n1 + i1]
							+ u_local[((i3 + 1) * n2 + (i2 - 1)) * n1 + i1]
							+ u_local[((i3 + 1) * n2 + (i2 + 1)) * n1 + i1])
					for i1 in range(1, n1 - 1):
						r_local[(i3 * n2 + i2) * n1 + i1] = (
							v_local[(i3 * n2 + i2) * n1 + i1]
							- a_local[0] * u_local[(i3 * n2 + i2) * n1 + i1]
							- a_local[2] * (u2_view[i1] + u1_view[i1 - 1] + u1_view[i1 + 1])
							- a_local[3] * (u2_view[i1 - 1] + u2_view[i1 + 1])
						)

		if timeron:
			with omp("single"):
				c_timers.timer_stop(T_RESID)

		comm3(r_local, n1, n2, n3, k)

		with omp("single"):
			if debug_vec[0] >= 1:
				rep_nrm(r_local, n1, n2, n3, "   resid", k)
			if debug_vec[2] >= k:
				showall(r_local, n1, n2, n3)
# END resid()


@omp(compile=use_compiled_types())
def resid_types(pointer_u, pointer_v, pointer_r, n1: int, n2: int, n3: int, a, k: int):
	u_local: cython.double[:] = pointer_u
	v_local: cython.double[:] = pointer_v
	r_local: cython.double[:] = pointer_r
	a_local: cython.double[:] = a
	i1: int
	i2: int
	i3: int

	with omp("parallel"):
		u1 = numpy.empty(n1, dtype=numpy.float64)
		u2 = numpy.empty(n1, dtype=numpy.float64)
		u1_view: cython.double[:] = u1
		u2_view: cython.double[:] = u2

		if timeron:
			with omp("single"):
				c_timers.timer_start(T_RESID)

		with omp("for"):
			for i3 in range(1, n3 - 1):
				for i2 in range(1, n2 - 1):
					for i1 in range(n1):
						u1_view[i1] = (u_local[(i3 * n2 + (i2 - 1)) * n1 + i1]
							+ u_local[(i3 * n2 + (i2 + 1)) * n1 + i1]
							+ u_local[((i3 - 1) * n2 + i2) * n1 + i1]
							+ u_local[((i3 + 1) * n2 + i2) * n1 + i1])
						u2_view[i1] = (u_local[((i3 - 1) * n2 + (i2 - 1)) * n1 + i1]
							+ u_local[((i3 - 1) * n2 + (i2 + 1)) * n1 + i1]
							+ u_local[((i3 + 1) * n2 + (i2 - 1)) * n1 + i1]
							+ u_local[((i3 + 1) * n2 + (i2 + 1)) * n1 + i1])
					for i1 in range(1, n1 - 1):
						r_local[(i3 * n2 + i2) * n1 + i1] = (
							v_local[(i3 * n2 + i2) * n1 + i1]
							- a_local[0] * u_local[(i3 * n2 + i2) * n1 + i1]
							- a_local[2] * (u2_view[i1] + u1_view[i1 - 1] + u1_view[i1 + 1])
							- a_local[3] * (u2_view[i1 - 1] + u2_view[i1 + 1])
						)

		if timeron:
			with omp("single"):
				c_timers.timer_stop(T_RESID)

		comm3_types(r_local, n1, n2, n3, k)

		with omp("single"):
			if debug_vec[0] >= 1:
				rep_nrm(r_local, n1, n2, n3, "   resid", k)
			if debug_vec[2] >= k:
				showall(r_local, n1, n2, n3)
# END resid_types()


@omp(compile=use_compiled())
def rprj3(pointer_r, m1k, m2k, m3k, pointer_s, m1j, m2j, m3j, k):
	r_local = pointer_r
	s_local = pointer_s

	if m1k == 3:
		d1 = 2
	else:
		d1 = 1

	if m2k == 3:
		d2 = 2
	else:
		d2 = 1

	if m3k == 3:
		d3 = 2
	else:
		d3 = 1

	with omp("parallel"):
		x1 = numpy.empty(m1k, dtype=numpy.float64)
		y1 = numpy.empty(m1k, dtype=numpy.float64)
		x1_view = x1
		y1_view = y1

		if timeron:
			with omp("single"):
				c_timers.timer_start(T_RPRJ3)

		with omp("for"):
			for j3 in range(1, m3j - 1):
				i3 = 2 * j3 - d3
				for j2 in range(1, m2j - 1):
					i2 = 2 * j2 - d2
					for j1 in range(1, m1j):
						i1 = 2 * j1 - d1
						x1_view[i1] = (r_local[((i3 + 1) * m2k + i2) * m1k + i1]
							+ r_local[((i3 + 1) * m2k + (i2 + 2)) * m1k + i1]
							+ r_local[(i3 * m2k + (i2 + 1)) * m1k + i1]
							+ r_local[((i3 + 2) * m2k + (i2 + 1)) * m1k + i1])
						y1_view[i1] = (r_local[(i3 * m2k + i2) * m1k + i1]
							+ r_local[((i3 + 2) * m2k + i2) * m1k + i1]
							+ r_local[(i3 * m2k + (i2 + 2)) * m1k + i1]
							+ r_local[((i3 + 2) * m2k + (i2 + 2)) * m1k + i1])
					for j1 in range(1, m1j - 1):
						i1 = 2 * j1 - d1
						y2 = (r_local[(i3 * m2k + i2) * m1k + (i1 + 1)]
							+ r_local[((i3 + 2) * m2k + i2) * m1k + (i1 + 1)]
							+ r_local[(i3 * m2k + (i2 + 2)) * m1k + (i1 + 1)]
							+ r_local[((i3 + 2) * m2k + (i2 + 2)) * m1k + (i1 + 1)])
						x2 = (r_local[((i3 + 1) * m2k + i2) * m1k + (i1 + 1)]
							+ r_local[((i3 + 1) * m2k + (i2 + 2)) * m1k + (i1 + 1)]
							+ r_local[(i3 * m2k + (i2 + 1)) * m1k + (i1 + 1)]
							+ r_local[((i3 + 2) * m2k + (i2 + 1)) * m1k + (i1 + 1)])
						s_local[(j3 * m2j + j2) * m1j + j1] = (
							0.5 * r_local[((i3 + 1) * m2k + (i2 + 1)) * m1k + (i1 + 1)]
							+ 0.25 * (r_local[((i3 + 1) * m2k + (i2 + 1)) * m1k + i1] + r_local[((i3 + 1) * m2k + (i2 + 1)) * m1k + (i1 + 2)] + x2)
							+ 0.125 * (x1_view[i1] + x1_view[i1 + 2] + y2)
							+ 0.0625 * (y1_view[i1] + y1_view[i1 + 2])
						)

		if timeron:
			with omp("single"):
				c_timers.timer_stop(T_RPRJ3)

		comm3(s_local, m1j, m2j, m3j, k - 1)

		with omp("single"):
			if debug_vec[0] >= 1:
				rep_nrm(s_local, m1j, m2j, m3j, "   rprj3", k - 1)
			if debug_vec[4] >= k:
				showall(s_local, m1j, m2j, m3j)
# END rprj3()


@omp(compile=use_compiled_types())
def rprj3_types(pointer_r, m1k: int, m2k: int, m3k: int, pointer_s, m1j: int, m2j: int, m3j: int, k: int):
	r_local: cython.double[:] = pointer_r
	s_local: cython.double[:] = pointer_s
	d1: int
	d2: int
	d3: int
	i1: int
	i2: int
	i3: int
	j1: int
	j2: int
	j3: int
	x2: float
	y2: float

	if m1k == 3:
		d1 = 2
	else:
		d1 = 1

	if m2k == 3:
		d2 = 2
	else:
		d2 = 1

	if m3k == 3:
		d3 = 2
	else:
		d3 = 1

	with omp("parallel"):
		x1 = numpy.empty(m1k, dtype=numpy.float64)
		y1 = numpy.empty(m1k, dtype=numpy.float64)
		x1_view: cython.double[:] = x1
		y1_view: cython.double[:] = y1

		if timeron:
			with omp("single"):
				c_timers.timer_start(T_RPRJ3)

		with omp("for"):
			for j3 in range(1, m3j - 1):
				i3 = 2 * j3 - d3
				for j2 in range(1, m2j - 1):
					i2 = 2 * j2 - d2
					for j1 in range(1, m1j):
						i1 = 2 * j1 - d1
						x1_view[i1] = (r_local[((i3 + 1) * m2k + i2) * m1k + i1]
							+ r_local[((i3 + 1) * m2k + (i2 + 2)) * m1k + i1]
							+ r_local[(i3 * m2k + (i2 + 1)) * m1k + i1]
							+ r_local[((i3 + 2) * m2k + (i2 + 1)) * m1k + i1])
						y1_view[i1] = (r_local[(i3 * m2k + i2) * m1k + i1]
							+ r_local[((i3 + 2) * m2k + i2) * m1k + i1]
							+ r_local[(i3 * m2k + (i2 + 2)) * m1k + i1]
							+ r_local[((i3 + 2) * m2k + (i2 + 2)) * m1k + i1])
					for j1 in range(1, m1j - 1):
						i1 = 2 * j1 - d1
						y2 = (r_local[(i3 * m2k + i2) * m1k + (i1 + 1)]
							+ r_local[((i3 + 2) * m2k + i2) * m1k + (i1 + 1)]
							+ r_local[(i3 * m2k + (i2 + 2)) * m1k + (i1 + 1)]
							+ r_local[((i3 + 2) * m2k + (i2 + 2)) * m1k + (i1 + 1)])
						x2 = (r_local[((i3 + 1) * m2k + i2) * m1k + (i1 + 1)]
							+ r_local[((i3 + 1) * m2k + (i2 + 2)) * m1k + (i1 + 1)]
							+ r_local[(i3 * m2k + (i2 + 1)) * m1k + (i1 + 1)]
							+ r_local[((i3 + 2) * m2k + (i2 + 1)) * m1k + (i1 + 1)])
						s_local[(j3 * m2j + j2) * m1j + j1] = (
							0.5 * r_local[((i3 + 1) * m2k + (i2 + 1)) * m1k + (i1 + 1)]
							+ 0.25 * (r_local[((i3 + 1) * m2k + (i2 + 1)) * m1k + i1] + r_local[((i3 + 1) * m2k + (i2 + 1)) * m1k + (i1 + 2)] + x2)
							+ 0.125 * (x1_view[i1] + x1_view[i1 + 2] + y2)
							+ 0.0625 * (y1_view[i1] + y1_view[i1 + 2])
						)

		if timeron:
			with omp("single"):
				c_timers.timer_stop(T_RPRJ3)

		comm3_types(s_local, m1j, m2j, m3j, k - 1)

		with omp("single"):
			if debug_vec[0] >= 1:
				rep_nrm(s_local, m1j, m2j, m3j, "   rprj3", k - 1)
			if debug_vec[4] >= k:
				showall(s_local, m1j, m2j, m3j)
# END rprj3_types()


@omp(compile=use_compiled())
def mg3P(u_local, v_local, r_local, a, c, m1_arg, m2_arg, m3_arg, ir_arg, lt_arg, lb_arg, n1, n2, n3, k):
	u_view = u_local
	v_view = v_local
	r_view = r_local
	a_view = a
	c_view = c
	m1_view = m1_arg
	m2_view = m2_arg
	m3_view = m3_arg
	ir_view = ir_arg

	for level in range(lt_arg, lb_arg, -1):
		j = level - 1
		rprj3(r_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], r_view[ir_view[j]:], m1_view[j], m2_view[j], m3_view[j], level)

	level = lb_arg
	zero3(u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level])
	psinv(r_view[ir_view[level]:], u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], c_view, level)

	for level in range(lb_arg + 1, lt_arg):
		j = level - 1
		zero3(u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level])
		interp(u_view[ir_view[j]:], m1_view[j], m2_view[j], m3_view[j], u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], level)
		resid(u_view[ir_view[level]:], r_view[ir_view[level]:], r_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], a_view, level)
		psinv(r_view[ir_view[level]:], u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], c_view, level)

	j = lt_arg - 1
	level = lt_arg
	interp(u_view[ir_view[j]:], m1_view[j], m2_view[j], m3_view[j], u_view, n1, n2, n3, level)
	resid(u_view, v_view, r_view, n1, n2, n3, a_view, level)
	psinv(r_view, u_view, n1, n2, n3, c_view, level)
# END mg3P()


@omp(compile=use_compiled_types())
def mg3P_types(u_local, v_local, r_local, a, c, m1_arg, m2_arg, m3_arg, ir_arg, lt_arg: int, lb_arg: int, n1: int, n2: int, n3: int, k: int):
	u_view: cython.double[:] = u_local
	v_view: cython.double[:] = v_local
	r_view: cython.double[:] = r_local
	a_view: cython.double[:] = a
	c_view: cython.double[:] = c
	m1_view: cython.int[:] = m1_arg
	m2_view: cython.int[:] = m2_arg
	m3_view: cython.int[:] = m3_arg
	ir_view: cython.int[:] = ir_arg
	level: int
	j: int

	for level in range(lt_arg, lb_arg, -1):
		j = level - 1
		rprj3_types(r_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], r_view[ir_view[j]:], m1_view[j], m2_view[j], m3_view[j], level)

	level = lb_arg
	zero3_types(u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level])
	psinv_types(r_view[ir_view[level]:], u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], c_view, level)

	for level in range(lb_arg + 1, lt_arg):
		j = level - 1
		zero3_types(u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level])
		interp_types(u_view[ir_view[j]:], m1_view[j], m2_view[j], m3_view[j], u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], level)
		resid_types(u_view[ir_view[level]:], r_view[ir_view[level]:], r_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], a_view, level)
		psinv_types(r_view[ir_view[level]:], u_view[ir_view[level]:], m1_view[level], m2_view[level], m3_view[level], c_view, level)

	j = lt_arg - 1
	level = lt_arg
	interp_types(u_view[ir_view[j]:], m1_view[j], m2_view[j], m3_view[j], u_view, n1, n2, n3, level)
	resid_types(u_view, v_view, r_view, n1, n2, n3, a_view, level)
	psinv_types(r_view, u_view, n1, n2, n3, c_view, level)
# END mg3P_types()


@omp(compile=use_compiled())
def zran3(pointer_z, n1, n2, n3, nx_aux, ny_aux, k, is1_arg, is2_arg, is3_arg, ie1_arg, ie2_arg, ie3_arg):
	z_local = pointer_z
	mm = MM

	ten = numpy.empty(2 * mm, dtype=numpy.float64)
	j1 = numpy.empty(2 * mm, dtype=numpy.int32)
	j2 = numpy.empty(2 * mm, dtype=numpy.int32)
	j3 = numpy.empty(2 * mm, dtype=numpy.int32)
	jg = numpy.empty(2 * mm * 4, dtype=numpy.int32)

	a1 = power(A, nx_aux)
	a2 = power(A, nx_aux * ny_aux)

	zero3(z_local, n1, n2, n3)

	i = is1_arg - 2 + nx_aux * (is2_arg - 2 + ny_aux * (is3_arg - 2))
	ai = power(A, i)
	d1 = ie1_arg - is1_arg + 1
	e1 = ie1_arg - is1_arg + 2
	e2 = ie2_arg - is2_arg + 2
	e3 = ie3_arg - is3_arg + 2
	x0 = X
	x0 = randlc_seed(x0, ai)

	for i3 in range(1, e3):
		x1 = x0
		for i2 in range(1, e2):
			xx = x1
			start = (i3 * n2 + i2) * n1 + 1
			xx = vranlc_mg(d1, xx, A, z_local[start:])
			x1 = randlc_seed(x1, a1)
		x0 = randlc_seed(x0, a2)

	for i in range(mm):
		top = mm + i
		ten[top] = 0.0
		j1[top] = 0
		j2[top] = 0
		j3[top] = 0
		ten[i] = 1.0
		j1[i] = 0
		j2[i] = 0
		j3[i] = 0

	for i3 in range(1, n3 - 1):
		for i2 in range(1, n2 - 1):
			for i1 in range(1, n1 - 1):
				value = z_local[(i3 * n2 + i2) * n1 + i1]
				if value > ten[mm]:
					ten[mm] = value
					j1[mm] = i1
					j2[mm] = i2
					j3[mm] = i3
					bubble(ten, j1, j2, j3, mm, 1)
				if value < ten[0]:
					ten[0] = value
					j1[0] = i1
					j2[0] = i2
					j3[0] = i3
					bubble(ten, j1, j2, j3, mm, 0)

	i1 = mm - 1
	i0 = mm - 1
	for i in range(mm - 1, -1, -1):
		best = 0.0
		source = mm + i1
		target = (mm + i) * 4
		if best < ten[source]:
			jg[target] = 0
			jg[target + 1] = is1_arg - 2 + j1[source]
			jg[target + 2] = is2_arg - 2 + j2[source]
			jg[target + 3] = is3_arg - 2 + j3[source]
			i1 = i1 - 1
		else:
			jg[target] = 0
			jg[target + 1] = 0
			jg[target + 2] = 0
			jg[target + 3] = 0

		best = 1.0
		source = i0
		target = i * 4
		if best > ten[source]:
			jg[target] = 0
			jg[target + 1] = is1_arg - 2 + j1[source]
			jg[target + 2] = is2_arg - 2 + j2[source]
			jg[target + 3] = is3_arg - 2 + j3[source]
			i0 = i0 - 1
		else:
			jg[target] = 0
			jg[target + 1] = 0
			jg[target + 2] = 0
			jg[target + 3] = 0

	zero3(z_local, n1, n2, n3)

	for i in range(mm - 1, -1, -1):
		target = i * 4
		z_local[(jg[target + 3] * n2 + jg[target + 2]) * n1 + jg[target + 1]] = -1.0

	for i in range(mm - 1, -1, -1):
		target = (mm + i) * 4
		z_local[(jg[target + 3] * n2 + jg[target + 2]) * n1 + jg[target + 1]] = 1.0

	with omp("parallel"):
		comm3(z_local, n1, n2, n3, k)
# END zran3()


@omp(compile=use_compiled_types())
def zran3_types(pointer_z, n1: int, n2: int, n3: int, nx_aux: int, ny_aux: int, k: int, is1_arg: int, is2_arg: int, is3_arg: int, ie1_arg: int, ie2_arg: int, ie3_arg: int):
	z_local: cython.double[:] = pointer_z
	mm: int = MM
	ten_values = numpy.empty(2 * mm, dtype=numpy.float64)
	j1_values = numpy.empty(2 * mm, dtype=numpy.int32)
	j2_values = numpy.empty(2 * mm, dtype=numpy.int32)
	j3_values = numpy.empty(2 * mm, dtype=numpy.int32)
	jg_values = numpy.empty(2 * mm * 4, dtype=numpy.int32)
	ten: cython.double[:] = ten_values
	j1: cython.int[:] = j1_values
	j2: cython.int[:] = j2_values
	j3: cython.int[:] = j3_values
	jg: cython.int[:] = jg_values
	a_const: float = A
	x_const: float = X
	a1: float = power_types(a_const, nx_aux)
	a2: float = power_types(a_const, nx_aux * ny_aux)
	i: int
	i0: int
	i1: int
	i2: int
	i3: int
	ai: float
	d1: int
	e1: int
	e2: int
	e3: int
	x0: float
	x1: float
	xx: float
	start: int
	top: int
	source: int
	target: int
	best: float
	value: float

	zero3_types(z_local, n1, n2, n3)

	i = is1_arg - 2 + nx_aux * (is2_arg - 2 + ny_aux * (is3_arg - 2))
	ai = power_types(a_const, i)
	d1 = ie1_arg - is1_arg + 1
	e1 = ie1_arg - is1_arg + 2
	e2 = ie2_arg - is2_arg + 2
	e3 = ie3_arg - is3_arg + 2
	x0 = x_const
	x0 = randlc_seed_types(x0, ai)

	for i3 in range(1, e3):
		x1 = x0
		for i2 in range(1, e2):
			xx = x1
			start = (i3 * n2 + i2) * n1 + 1
			xx = vranlc_mg_types(d1, xx, a_const, z_local[start:])
			x1 = randlc_seed_types(x1, a1)
		x0 = randlc_seed_types(x0, a2)

	for i in range(mm):
		top = mm + i
		ten[top] = 0.0
		j1[top] = 0
		j2[top] = 0
		j3[top] = 0
		ten[i] = 1.0
		j1[i] = 0
		j2[i] = 0
		j3[i] = 0

	for i3 in range(1, n3 - 1):
		for i2 in range(1, n2 - 1):
			for i1 in range(1, n1 - 1):
				value = z_local[(i3 * n2 + i2) * n1 + i1]
				if value > ten[mm]:
					ten[mm] = value
					j1[mm] = i1
					j2[mm] = i2
					j3[mm] = i3
					bubble_types(ten, j1, j2, j3, mm, 1)
				if value < ten[0]:
					ten[0] = value
					j1[0] = i1
					j2[0] = i2
					j3[0] = i3
					bubble_types(ten, j1, j2, j3, mm, 0)

	i1 = mm - 1
	i0 = mm - 1
	for i in range(mm - 1, -1, -1):
		best = 0.0
		source = mm + i1
		target = (mm + i) * 4
		if best < ten[source]:
			jg[target] = 0
			jg[target + 1] = is1_arg - 2 + j1[source]
			jg[target + 2] = is2_arg - 2 + j2[source]
			jg[target + 3] = is3_arg - 2 + j3[source]
			i1 = i1 - 1
		else:
			jg[target] = 0
			jg[target + 1] = 0
			jg[target + 2] = 0
			jg[target + 3] = 0

		best = 1.0
		source = i0
		target = i * 4
		if best > ten[source]:
			jg[target] = 0
			jg[target + 1] = is1_arg - 2 + j1[source]
			jg[target + 2] = is2_arg - 2 + j2[source]
			jg[target + 3] = is3_arg - 2 + j3[source]
			i0 = i0 - 1
		else:
			jg[target] = 0
			jg[target + 1] = 0
			jg[target + 2] = 0
			jg[target + 3] = 0

	zero3_types(z_local, n1, n2, n3)

	for i in range(mm - 1, -1, -1):
		target = i * 4
		z_local[(jg[target + 3] * n2 + jg[target + 2]) * n1 + jg[target + 1]] = -1.0

	for i in range(mm - 1, -1, -1):
		target = (mm + i) * 4
		z_local[(jg[target + 3] * n2 + jg[target + 2]) * n1 + jg[target + 1]] = 1.0

	with omp("parallel"):
		comm3_types(z_local, n1, n2, n3, k)
# END zran3_types()


def get_verify_value():
	if npbparams.CLASS == "S":
		return 0.5307707005734e-04
	if npbparams.CLASS == "W":
		return 0.6467329375339e-05
	if npbparams.CLASS == "A":
		return 0.2433365309069e-05
	if npbparams.CLASS == "B":
		return 0.1800564401355e-05
	if npbparams.CLASS == "C":
		return 0.5706732285740e-06
	if npbparams.CLASS == "D":
		return 0.1583275060440e-09
	if npbparams.CLASS == "E":
		return 0.8157592357404e-10
	return 0.0
# END get_verify_value()


@omp(compile=False)
def main():
	global debug_vec
	global is1, is2, is3, ie1, ie2, ie3, lt, lb
	global timeron

	for i in range(T_LAST):
		c_timers.timer_clear(i)

	t_names = numpy.empty(T_LAST, dtype=object)
	c_timers.timer_start(T_INIT)

	timeron = os.path.isfile("timer.flag")
	if timeron:
		t_names[T_INIT] = "init"
		t_names[T_BENCH] = "benchmk"
		t_names[T_MG3P] = "mg3P"
		t_names[T_PSINV] = "psinv"
		t_names[T_RESID] = "resid"
		t_names[T_RPRJ3] = "rprj3"
		t_names[T_INTERP] = "interp"
		t_names[T_NORM2] = "norm2"
		t_names[T_COMM3] = "comm3"

	print(" No input file. Using compiled defaults")
	lt = npbparams.LT_DEFAULT
	nit = npbparams.NIT_DEFAULT
	nx[lt] = npbparams.NX_DEFAULT
	ny[lt] = npbparams.NY_DEFAULT
	nz[lt] = npbparams.NZ_DEFAULT

	a = numpy.empty(4, dtype=numpy.float64)
	a[0] = -8.0 / 3.0
	a[1] = 0.0
	a[2] = 1.0 / 6.0
	a[3] = 1.0 / 12.0

	c = numpy.empty(4, dtype=numpy.float64)
	if npbparams.CLASS in ("A", "S", "W"):
		c[0] = -3.0 / 8.0
		c[1] = 1.0 / 32.0
		c[2] = -1.0 / 64.0
		c[3] = 0.0
	else:
		c[0] = -3.0 / 17.0
		c[1] = 1.0 / 33.0
		c[2] = -1.0 / 61.0
		c[3] = 0.0

	lb = 1
	k = lt

	n1, n2, n3 = setup(k)

	if use_compiled_types():
		zero3_types(u, n1, n2, n3)
		zran3_types(v, n1, n2, n3, nx[lt], ny[lt], k, is1, is2, is3, ie1, ie2, ie3)
	else:
		zero3(u, n1, n2, n3)
		zran3(v, n1, n2, n3, nx[lt], ny[lt], k, is1, is2, is3, ie1, ie2, ie3)

	if use_compiled_types():
		norm2u3_types(v, n1, n2, n3, nx[lt], ny[lt], nz[lt])
	else:
		norm2u3(v, n1, n2, n3, nx[lt], ny[lt], nz[lt])

	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - MG Benchmark\n")
	print(" Size: %3dx%3dx%3d (class_npb %1c)" % (nx[lt], ny[lt], nz[lt], npbparams.CLASS))
	print(" Iterations: %3d" % nit)

	if use_compiled_types():
		resid_types(u, v, r, n1, n2, n3, a, k)
		norm2u3_types(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])
		mg3P_types(u, v, r, a, c, m1, m2, m3, ir, lt, lb, n1, n2, n3, k)
		resid_types(u, v, r, n1, n2, n3, a, k)
	else:
		resid(u, v, r, n1, n2, n3, a, k)
		norm2u3(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])
		mg3P(u, v, r, a, c, m1, m2, m3, ir, lt, lb, n1, n2, n3, k)
		resid(u, v, r, n1, n2, n3, a, k)

	n1, n2, n3 = setup(k)

	if use_compiled_types():
		zero3_types(u, n1, n2, n3)
		zran3_types(v, n1, n2, n3, nx[lt], ny[lt], k, is1, is2, is3, ie1, ie2, ie3)
	else:
		zero3(u, n1, n2, n3)
		zran3(v, n1, n2, n3, nx[lt], ny[lt], k, is1, is2, is3, ie1, ie2, ie3)

	c_timers.timer_stop(T_INIT)
	tinit = c_timers.timer_read(T_INIT)
	print(" Initialization time: %15.3f seconds" % tinit)

	for i in range(T_BENCH, T_LAST):
		c_timers.timer_clear(i)
	c_timers.timer_start(T_BENCH)

	rnm2 = 0.0
	rnmu = 0.0
	t = 0.0

	if timeron:
		c_timers.timer_start(T_RESID2)

	if use_compiled_types():
		resid_types(u, v, r, n1, n2, n3, a, k)
	else:
		resid(u, v, r, n1, n2, n3, a, k)

	if timeron:
		c_timers.timer_stop(T_RESID2)

	if use_compiled_types():
		norm2u3_types(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])
	else:
		norm2u3(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])

	for it in range(1, nit + 1):
		if (it == 1) or (it == nit) or ((it % 5) == 0):
			print("  iter %3d" % it)

		if timeron:
			c_timers.timer_start(T_MG3P)

		if use_compiled_types():
			mg3P_types(u, v, r, a, c, m1, m2, m3, ir, lt, lb, n1, n2, n3, k)
		else:
			mg3P(u, v, r, a, c, m1, m2, m3, ir, lt, lb, n1, n2, n3, k)

		if timeron:
			c_timers.timer_stop(T_MG3P)
			c_timers.timer_start(T_RESID2)

		if use_compiled_types():
			resid_types(u, v, r, n1, n2, n3, a, k)
		else:
			resid(u, v, r, n1, n2, n3, a, k)

		if timeron:
			c_timers.timer_stop(T_RESID2)

		if use_compiled_types():
			rnm2, rnmu = norm2u3_types(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])
		else:
			rnm2, rnmu = norm2u3(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])

	c_timers.timer_stop(T_BENCH)
	t = c_timers.timer_read(T_BENCH)
	print(" Benchmark completed")

	verify_value = get_verify_value()
	verified = False
	epsilon = 1.0e-8
	if verify_value != 0.0:
		err = abs(rnm2 - verify_value) / verify_value
		if err <= epsilon:
			verified = True
			print(" VERIFICATION SUCCESSFUL")
			print(" L2 Norm is %20.13e" % rnm2)
			print(" Error is   %20.13e" % err)
		else:
			print(" VERIFICATION FAILED")
			print(" L2 Norm is             %20.13e" % rnm2)
			print(" The correct L2 Norm is %20.13e" % verify_value)
	else:
		print(" Problem size unknown")
		print(" NO VERIFICATION PERFORMED")

	nn = 1.0 * nx[lt] * ny[lt] * nz[lt]
	if t != 0.0:
		mflops = 58.0 * nit * nn * 1.0e-6 / t
	else:
		mflops = 0.0

	c_print_results.c_print_results("MG",
			npbparams.CLASS,
			nx[lt],
			ny[lt],
			nz[lt],
			nit,
			t,
			mflops,
			"          floating point",
			verified)

	if timeron:
		tmax = c_timers.timer_read(T_BENCH)
		if tmax == 0.0:
			tmax = 1.0
		print("  SECTION   Time (secs)")
		for i in range(T_BENCH, T_LAST):
			section_time = c_timers.timer_read(i)
			if i == T_RESID2:
				section_time = c_timers.timer_read(T_RESID) - section_time
				print("    --> %8s:%9.3f  (%6.2f%%)" % ("mg-resid", section_time, section_time * 100.0 / tmax))
			else:
				print("  %-8s:%9.3f  (%6.2f%%)" % (t_names[i], section_time, section_time * 100.0 / tmax))
# END main()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="NPB-PYTHON-SER MG")
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
	parser.add_argument("-m", "--mode", type=int, default=1, help="Mode: 0=pure, 1=hybrid, 2=compiled, 3=compiled with types")
	args = parser.parse_args()

	set_omp_mode(args.mode)
	set_omp_threads(args.threads)
	npbparams.set_mg_info(args.CLASS)
	set_global_variables()

	main()
