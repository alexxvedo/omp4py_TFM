# ------------------------------------------------------------------------------
#
# The original NPB 3.4.1 version was written in Fortran and belongs to:
# 	http://www.nas.nasa.gov/Software/NPB/
#
# Authors of the Fortran code:
#	P. O. Frederickson
#	D. H. Bailey
# 	A. C. Woo
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

from omputils import omp, omp_get_thread_num, omp_pure, use_pure, use_compiled, use_compiled_types, set_omp_threads, set_omp_mode

if use_pure():
    omp = omp_pure

try:
	import cython
except ImportError:
	class cython:
		double = []

# Local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import npbparams
from c_randdp import randlc
import c_timers
import c_print_results


M = 0
MK = 16
MM = 0
NN = 0
NK = 0
NQ = 10
EPSILON = 1.0e-8
A = 1220703125.0
S = 271828183.0
NK_PLUS = 0


def set_global_variables():
	global M, MM, NN, NK, NK_PLUS

	M = npbparams.M
	MM = M - MK
	NN = 1 << MM
	NK = 1 << MK
	NK_PLUS = (2 * NK) + 1
# END set_global_variables()


def find_start_seed_t1(kk, t1, t2):
	for _ in range(1, 101):
		ik = kk // 2
		if (2 * ik) != kk:
			_, t1 = randlc(t1, t2)
		if ik == 0:
			break
		_, t2 = randlc(t2, t2)
		kk = ik

	return t1
# END find_start_seed_t1()


def get_verify_values(m):
	if m == 24:
		return -3.247834652034740e3, -6.958407078382297e3, True
	if m == 25:
		return -2.863319731645753e3, -6.320053679109499e3, True
	if m == 28:
		return -4.295875165629892e3, -1.580732573678431e4, True
	if m == 30:
		return 4.033815542441498e4, -2.660669192809235e4, True
	if m == 32:
		return 4.764367927995374e4, -8.084072988043731e4, True
	if m == 36:
		return 1.982481200946593e5, -1.020596636361769e5, True
	if m == 40:
		return -5.319717441530e5, -3.688834557731e5, True
	return 0.0, 0.0, False
# END get_verify_values()


@omp(compile=use_compiled())
def vranlc_ep(n, x_seed, a, y):
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
# END vranlc_ep()


@omp(compile=use_compiled_types())
def vranlc_ep_types(n: int, x_seed: float, a: float, y_arg):
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
# END vranlc_ep_types()


@omp(compile=use_compiled())
def ep_compute(nn, nk, nk_plus, nq, an, timers_enabled):
	sx = 0.0
	sy = 0.0
	q = numpy.repeat(0.0, nq)
	k_offset = -1

	with omp("parallel reduction(+:sx,sy)"):
		qq = numpy.repeat(0.0, nq)
		x_local = numpy.empty(nk_plus, dtype=numpy.float64)
		thread_id = omp_get_thread_num()

		with omp("for schedule(static)"):
			for k in range(1, nn + 1):
				kk = k_offset + k
				t1_local = S
				t2_local = an

				t1_local = find_start_seed_t1(kk, t1_local, t2_local)

				if timers_enabled and thread_id == 0:
					c_timers.timer_start(2)
				t1_local = vranlc_ep(2 * nk, t1_local, A, x_local)
				if timers_enabled and thread_id == 0:
					c_timers.timer_stop(2)

				if timers_enabled and thread_id == 0:
					c_timers.timer_start(1)
				for i in range(nk):
					x1 = 2.0 * x_local[2 * i] - 1.0
					x2 = 2.0 * x_local[2 * i + 1] - 1.0
					t1_value = x1 * x1 + x2 * x2
					if t1_value <= 1.0:
						t2_value = math.sqrt(-2.0 * math.log(t1_value) / t1_value)
						t3_value = x1 * t2_value
						t4_value = x2 * t2_value
						l = int(max(abs(t3_value), abs(t4_value)))
						qq[l] = qq[l] + 1.0
						sx = sx + t3_value
						sy = sy + t4_value
				if timers_enabled and thread_id == 0:
					c_timers.timer_stop(1)

		with omp("critical"):
			for i in range(nq):
				q[i] = q[i] + qq[i]

	return sx, sy, q
# END ep_compute()


@omp(compile=use_compiled_types())
def ep_compute_types(nn: int, nk: int, nk_plus: int, nq: int, an: float):
	sx: float = 0.0
	sy: float = 0.0
	k_offset: int = -1
	r23: float = pow(0.5, 23.0)
	r46: float = pow(0.5, 46.0)
	t23: float = pow(2.0, 23.0)
	t46: float = pow(2.0, 46.0)
	a: float = A
	s: float = S
	i: int
	k: int
	seed_iter: int
	ik: int
	kk: int
	l: int
	a1: int
	x1_seed: int
	t2_int: int
	t4_int: int
	t1: float
	t3: float
	a2: float
	x2_seed: float
	z: float
	t1_local: float
	t2_local: float
	x1: float
	x2: float
	t1_value: float
	t2_value: float
	t3_value: float
	t4_value: float
	q_values = numpy.repeat(0.0, nq)
	q: cython.double[:] = q_values

	with omp("parallel reduction(+:sx,sy) private(i,k,seed_iter,ik,kk,l,a1,x1_seed,t2_int,t4_int,t1,t3,a2,x2_seed,z,t1_local,t2_local,x1,x2,t1_value,t2_value,t3_value,t4_value)"):
		qq_values = numpy.repeat(0.0, nq)
		x_values = numpy.empty(nk_plus, dtype=numpy.float64)
		qq: cython.double[:] = qq_values
		x_local: cython.double[:] = x_values

		with omp("for schedule(static)"):
			for k in range(1, nn + 1):
				kk = k_offset + k
				t1_local = s
				t2_local = an

				for seed_iter in range(1, 101):
					ik = kk // 2
					if (2 * ik) != kk:
						t1 = r23 * t2_local
						a1 = int(t1)
						a2 = t2_local - t23 * a1
						t1 = r23 * t1_local
						x1_seed = int(t1)
						x2_seed = t1_local - t23 * x1_seed
						t1 = a1 * x2_seed + a2 * x1_seed
						t2_int = int(r23 * t1)
						z = t1 - t23 * t2_int
						t3 = t23 * z + a2 * x2_seed
						t4_int = int(r46 * t3)
						t1_local = t3 - t46 * t4_int
					if ik == 0:
						break

					t1 = r23 * t2_local
					a1 = int(t1)
					a2 = t2_local - t23 * a1
					t1 = r23 * t2_local
					x1_seed = int(t1)
					x2_seed = t2_local - t23 * x1_seed
					t1 = a1 * x2_seed + a2 * x1_seed
					t2_int = int(r23 * t1)
					z = t1 - t23 * t2_int
					t3 = t23 * z + a2 * x2_seed
					t4_int = int(r46 * t3)
					t2_local = t3 - t46 * t4_int
					kk = ik

				t1_local = vranlc_ep_types(2 * nk, t1_local, a, x_values)

				for i in range(nk):
					x1 = 2.0 * x_local[2 * i] - 1.0
					x2 = 2.0 * x_local[2 * i + 1] - 1.0
					t1_value = x1 * x1 + x2 * x2
					if t1_value <= 1.0:
						t2_value = math.sqrt(-2.0 * math.log(t1_value) / t1_value)
						t3_value = x1 * t2_value
						t4_value = x2 * t2_value
						l = int(max(abs(t3_value), abs(t4_value)))
						qq[l] = qq[l] + 1.0
						sx = sx + t3_value
						sy = sy + t4_value

		with omp("critical"):
			for i in range(nq):
				q[i] = q[i] + qq[i]

	return sx, sy, q_values
# END ep_compute_types()


def main(m, mm, nn, nk, nk_plus):
	dum = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float64)
	timers_enabled = os.path.isfile("timer.flag")

	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - EP Benchmark\n")
	print(" Number of random numbers generated:", int(pow(2, m + 1)))

	vranlc_ep(0, dum[0], dum[1], numpy.array([], dtype=numpy.float64))
	dum[0], dum[1] = randlc(dum[1], dum[2])
	x = numpy.repeat(-1.0e99, nk_plus)
	_ = math.log(math.sqrt(abs(max(1.0, 1.0))))

	c_timers.timer_clear(0)
	c_timers.timer_clear(1)
	c_timers.timer_clear(2)
	c_timers.timer_start(0)

	t1 = A
	t1 = vranlc_ep(0, t1, A, x)
	t1 = A
	for _ in range(MK + 1):
		_, t1 = randlc(t1, t1)

	an = t1
	if use_compiled_types():
		sx, sy, q = ep_compute_types(nn, nk, nk_plus, NQ, an)
	else:
		sx, sy, q = ep_compute(nn, nk, nk_plus, NQ, an, timers_enabled)

	gc = float(numpy.sum(q))

	c_timers.timer_stop(0)
	tm = c_timers.timer_read(0)

	sx_verify_value, sy_verify_value, verified = get_verify_values(m)
	if verified:
		sx_err = abs((sx - sx_verify_value) / sx_verify_value)
		sy_err = abs((sy - sy_verify_value) / sy_verify_value)
		verified = (sx_err <= EPSILON) and (sy_err <= EPSILON)

	mops = pow(2.0, m + 1) / tm / 1000000.0

	print("\n EP Benchmark Results:\n")
	print(" CPU Time = {0:10.4f}".format(tm))
	print(" N = 2^{0:5d}".format(m))
	print(" No. Gaussian Pairs = {0:15.0f}".format(gc))
	print(" Sums = {0:25.15e} {1:25.15e}".format(sx, sy))
	print(" Counts: ")
	for i in range(NQ):
		print("{0:3d}{1:15.0f}".format(i, q[i]))

	c_print_results.c_print_results("EP",
			npbparams.CLASS,
			m + 1,
			0,
			0,
			0,
			tm,
			mops,
			"Random numbers generated",
			verified)

	if timers_enabled:
		if tm <= 0.0:
			tm = 1.0
		tt = c_timers.timer_read(0)
		print("\nTotal time:     {0:9.3f} ({1:6.2f})".format(tt, tt * 100.0 / tm))
		tt = c_timers.timer_read(1)
		print("Gaussian pairs: {0:9.3f} ({1:6.2f})".format(tt, tt * 100.0 / tm))
		tt = c_timers.timer_read(2)
		print("Random numbers: {0:9.3f} ({1:6.2f})".format(tt, tt * 100.0 / tm))
# END main()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="NPB-PYTHON-SER EP")
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
	parser.add_argument("-m", "--mode", type=int, default=1, help="Mode: 0=pure, 1=hybrid, 2=compiled, 3=compiled with types")
	args = parser.parse_args()

	set_omp_mode(args.mode)
	set_omp_threads(args.threads)
	npbparams.set_ep_info(args.CLASS)
	set_global_variables()

	main(M, MM, NN, NK, NK_PLUS)
