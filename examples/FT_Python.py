# ------------------------------------------------------------------------------
#
# The original NPB 3.4.1 version was written in Fortran and belongs to:
# 	http://www.nas.nasa.gov/Software/NPB/
#
# Authors of the Fortran code:
#	D. Bailey
#	W. Saphir
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

from omputils import (
	omp,
	omp_get_max_threads,
	omp_get_thread_num,
	omp_pure,
	use_pure,
	use_compiled,
	use_compiled_types,
	set_omp_threads,
	set_omp_mode,
	get_omp_threads,
)

if use_pure():
    omp = omp_pure

try:
	import cython
except ImportError:
	class cython:
		double = []
		doublecomplex = []

# Local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common'))
import npbparams
import c_timers
import c_print_results


# Global variables
FFTBLOCK_DEFAULT = 0
FFTBLOCKPAD_DEFAULT = 0
FFTBLOCK = 0
FFTBLOCKPAD = 0
SEED = 314159265.0
A = 1220703125.0
PI = 3.141592653589793238
ALPHA = 1.0e-6
T_TOTAL = 1
T_SETUP = 2
T_FFT = 3
T_EVOLVE = 4
T_CHECKSUM = 5
T_FFTX = 6
T_FFTY = 7
T_FFTZ = 8
T_MAX = 8

NX = 0
NY = 0
NZ = 0

sums = None
twiddle = None
u = None
u0 = None
u1 = None

dims = numpy.empty(3, dtype=numpy.int32)

niter = 0
timers_enabled = False


def get_worker_slots():
	return max(1, get_omp_threads(), omp_get_max_threads(), os.cpu_count() or 1)
#END get_worker_slots()


def set_global_variables():
	global FFTBLOCK_DEFAULT, FFTBLOCKPAD_DEFAULT, FFTBLOCK, FFTBLOCKPAD
	global NX, NY, NZ
	global sums, twiddle, u, u0, u1

	FFTBLOCK_DEFAULT = npbparams.DEFAULT_BEHAVIOR
	FFTBLOCKPAD_DEFAULT = npbparams.DEFAULT_BEHAVIOR
	FFTBLOCK = FFTBLOCK_DEFAULT
	FFTBLOCKPAD = FFTBLOCKPAD_DEFAULT

	NX = npbparams.NX
	NY = npbparams.NY
	NZ = npbparams.NZ

	sums = numpy.empty(npbparams.NITER_DEFAULT + 1, dtype=numpy.complex128)
	twiddle = numpy.repeat(0.0, npbparams.NTOTAL)
	u = numpy.empty(npbparams.MAXDIM, dtype=numpy.complex128)
	u0 = numpy.repeat(complex(0.0, 0.0), npbparams.NTOTAL)
	u1 = numpy.repeat(complex(0.0, 0.0), npbparams.NTOTAL)
#END set_global_variables()


def print_timers():
	tstrings = numpy.empty(T_MAX + 1, dtype=object)
	tstrings[1] = "          total "
	tstrings[2] = "          setup "
	tstrings[3] = "            fft "
	tstrings[4] = "         evolve "
	tstrings[5] = "       checksum "
	tstrings[6] = "          fftx* "
	tstrings[7] = "          ffty* "
	tstrings[8] = "          fftz* "

	t_m = c_timers.timer_read(T_TOTAL)
	if t_m <= 0.0:
		t_m = 1.0
	for i in range(1, T_MAX + 1):
		t = c_timers.timer_read(i)
		print(" timer %2d(%16s) :%9.4f (%6.2f%%)" % (i, tstrings[i], t, t * 100.0 / t_m))
#END print_timers()


def verify(d1,
		d2,
		d3,
		nt):
	csum_ref = numpy.empty(nt + 1, dtype=numpy.complex128)
	epsilon = 1.0e-12

	if npbparams.CLASS == 'S':
		csum_ref[1] = complex(5.546087004964E+02, 4.845363331978E+02)
		csum_ref[2] = complex(5.546385409189E+02, 4.865304269511E+02)
		csum_ref[3] = complex(5.546148406171E+02, 4.883910722336E+02)
		csum_ref[4] = complex(5.545423607415E+02, 4.901273169046E+02)
		csum_ref[5] = complex(5.544255039624E+02, 4.917475857993E+02)
		csum_ref[6] = complex(5.542683411902E+02, 4.932597244941E+02)
	elif npbparams.CLASS == 'W':
		csum_ref[1] = complex(5.673612178944E+02, 5.293246849175E+02)
		csum_ref[2] = complex(5.631436885271E+02, 5.282149986629E+02)
		csum_ref[3] = complex(5.594024089970E+02, 5.270996558037E+02)
		csum_ref[4] = complex(5.560698047020E+02, 5.260027904925E+02)
		csum_ref[5] = complex(5.530898991250E+02, 5.249400845633E+02)
		csum_ref[6] = complex(5.504159734538E+02, 5.239212247086E+02)
	elif npbparams.CLASS == 'A':
		csum_ref[1] = complex(5.046735008193E+02, 5.114047905510E+02)
		csum_ref[2] = complex(5.059412319734E+02, 5.098809666433E+02)
		csum_ref[3] = complex(5.069376896287E+02, 5.098144042213E+02)
		csum_ref[4] = complex(5.077892868474E+02, 5.101336130759E+02)
		csum_ref[5] = complex(5.085233095391E+02, 5.104914655194E+02)
		csum_ref[6] = complex(5.091487099959E+02, 5.107917842803E+02)
	elif npbparams.CLASS == 'B':
		csum_ref[1] = complex(5.177643571579E+02, 5.077803458597E+02)
		csum_ref[2] = complex(5.154521291263E+02, 5.088249431599E+02)
		csum_ref[3] = complex(5.146409228649E+02, 5.096208912659E+02)
		csum_ref[4] = complex(5.142378756213E+02, 5.101023387619E+02)
		csum_ref[5] = complex(5.139626667737E+02, 5.103976610617E+02)
		csum_ref[6] = complex(5.137423460082E+02, 5.105948019802E+02)
		csum_ref[7] = complex(5.135547056878E+02, 5.107404165783E+02)
		csum_ref[8] = complex(5.133910925466E+02, 5.108576573661E+02)
		csum_ref[9] = complex(5.132470705390E+02, 5.109577278523E+02)
		csum_ref[10] = complex(5.131197729984E+02, 5.110460304483E+02)
		csum_ref[11] = complex(5.130070319283E+02, 5.111252433800E+02)
		csum_ref[12] = complex(5.129070537032E+02, 5.111968077718E+02)
		csum_ref[13] = complex(5.128182883502E+02, 5.112616233064E+02)
		csum_ref[14] = complex(5.127393733383E+02, 5.113203605551E+02)
		csum_ref[15] = complex(5.126691062020E+02, 5.113735928093E+02)
		csum_ref[16] = complex(5.126064276004E+02, 5.114218460548E+02)
		csum_ref[17] = complex(5.125504076570E+02, 5.114656139760E+02)
		csum_ref[18] = complex(5.125002331720E+02, 5.115053595966E+02)
		csum_ref[19] = complex(5.124551951846E+02, 5.115415130407E+02)
		csum_ref[20] = complex(5.124146770029E+02, 5.115744692211E+02)
	elif npbparams.CLASS == 'C':
		csum_ref[1] = complex(5.195078707457E+02, 5.149019699238E+02)
		csum_ref[2] = complex(5.155422171134E+02, 5.127578201997E+02)
		csum_ref[3] = complex(5.144678022222E+02, 5.122251847514E+02)
		csum_ref[4] = complex(5.140150594328E+02, 5.121090289018E+02)
		csum_ref[5] = complex(5.137550426810E+02, 5.121143685824E+02)
		csum_ref[6] = complex(5.135811056728E+02, 5.121496764568E+02)
		csum_ref[7] = complex(5.134569343165E+02, 5.121870921893E+02)
		csum_ref[8] = complex(5.133651975661E+02, 5.122193250322E+02)
		csum_ref[9] = complex(5.132955192805E+02, 5.122454735794E+02)
		csum_ref[10] = complex(5.132410471738E+02, 5.122663649603E+02)
		csum_ref[11] = complex(5.131971141679E+02, 5.122830879827E+02)
		csum_ref[12] = complex(5.131605205716E+02, 5.122965869718E+02)
		csum_ref[13] = complex(5.131290734194E+02, 5.123075927445E+02)
		csum_ref[14] = complex(5.131012720314E+02, 5.123166486553E+02)
		csum_ref[15] = complex(5.130760908195E+02, 5.123241541685E+02)
		csum_ref[16] = complex(5.130528295923E+02, 5.123304037599E+02)
		csum_ref[17] = complex(5.130310107773E+02, 5.123356167976E+02)
		csum_ref[18] = complex(5.130103090133E+02, 5.123399592211E+02)
		csum_ref[19] = complex(5.129905029333E+02, 5.123435588985E+02)
		csum_ref[20] = complex(5.129714421109E+02, 5.123465164008E+02)
	elif npbparams.CLASS == 'D':
		csum_ref[1] = complex(5.122230065252E+02, 5.118534037109E+02)
		csum_ref[2] = complex(5.120463975765E+02, 5.117061181082E+02)
		csum_ref[3] = complex(5.119865766760E+02, 5.117096364601E+02)
		csum_ref[4] = complex(5.119518799488E+02, 5.117373863950E+02)
		csum_ref[5] = complex(5.119269088223E+02, 5.117680347632E+02)
		csum_ref[6] = complex(5.119082416858E+02, 5.117967875532E+02)
		csum_ref[7] = complex(5.118943814638E+02, 5.118225281841E+02)
		csum_ref[8] = complex(5.118842385057E+02, 5.118451629348E+02)
		csum_ref[9] = complex(5.118769435632E+02, 5.118649119387E+02)
		csum_ref[10] = complex(5.118718203448E+02, 5.118820803844E+02)
		csum_ref[11] = complex(5.118683569061E+02, 5.118969781011E+02)
		csum_ref[12] = complex(5.118661708593E+02, 5.119098918835E+02)
		csum_ref[13] = complex(5.118649768950E+02, 5.119210777066E+02)
		csum_ref[14] = complex(5.118645605626E+02, 5.119307604484E+02)
		csum_ref[15] = complex(5.118647586618E+02, 5.119391362671E+02)
		csum_ref[16] = complex(5.118654451572E+02, 5.119463757241E+02)
		csum_ref[17] = complex(5.118665212451E+02, 5.119526269238E+02)
		csum_ref[18] = complex(5.118679083821E+02, 5.119580184108E+02)
		csum_ref[19] = complex(5.118695433664E+02, 5.119626617538E+02)
		csum_ref[20] = complex(5.118713748264E+02, 5.119666538138E+02)
		csum_ref[21] = complex(5.118733606701E+02, 5.119700787219E+02)
		csum_ref[22] = complex(5.118754661974E+02, 5.119730095953E+02)
		csum_ref[23] = complex(5.118776626738E+02, 5.119755100241E+02)
		csum_ref[24] = complex(5.118799262314E+02, 5.119776353561E+02)
		csum_ref[25] = complex(5.118822370068E+02, 5.119794338060E+02)
	elif npbparams.CLASS == 'E':
		csum_ref[1] = complex(5.121601045346E+02, 5.117395998266E+02)
		csum_ref[2] = complex(5.120905403678E+02, 5.118614716182E+02)
		csum_ref[3] = complex(5.120623229306E+02, 5.119074203747E+02)
		csum_ref[4] = complex(5.120438418997E+02, 5.119345900733E+02)
		csum_ref[5] = complex(5.120311521872E+02, 5.119551325550E+02)
		csum_ref[6] = complex(5.120226088809E+02, 5.119720179919E+02)
		csum_ref[7] = complex(5.120169296534E+02, 5.119861371665E+02)
		csum_ref[8] = complex(5.120131225172E+02, 5.119979364402E+02)
		csum_ref[9] = complex(5.120104767108E+02, 5.120077674092E+02)
		csum_ref[10] = complex(5.120085127969E+02, 5.120159443121E+02)
		csum_ref[11] = complex(5.120069224127E+02, 5.120227453670E+02)
		csum_ref[12] = complex(5.120055158164E+02, 5.120284096041E+02)
		csum_ref[13] = complex(5.120041820159E+02, 5.120331373793E+02)
		csum_ref[14] = complex(5.120028605402E+02, 5.120370938679E+02)
		csum_ref[15] = complex(5.120015223011E+02, 5.120404138831E+02)
		csum_ref[16] = complex(5.120001570022E+02, 5.120432068837E+02)
		csum_ref[17] = complex(5.119987650555E+02, 5.120455615860E+02)
		csum_ref[18] = complex(5.119973525091E+02, 5.120475499442E+02)
		csum_ref[19] = complex(5.119959279472E+02, 5.120492304629E+02)
		csum_ref[20] = complex(5.119945006558E+02, 5.120506508902E+02)
		csum_ref[21] = complex(5.119930795911E+02, 5.120518503782E+02)
		csum_ref[22] = complex(5.119916728462E+02, 5.120528612016E+02)
		csum_ref[23] = complex(5.119902874185E+02, 5.120537101195E+02)
		csum_ref[24] = complex(5.119889291565E+02, 5.120544194514E+02)
		csum_ref[25] = complex(5.119876028049E+02, 5.120550079284E+02)

	verified = True
	if d1 != npbparams.NX or d2 != npbparams.NY or d3 != npbparams.NZ or nt != npbparams.NITER_DEFAULT:
		verified = False

	if verified:
		for i in range(1, nt + 1):
			err = abs((sums[i] - csum_ref[i]) / csum_ref[i])
			if err > epsilon:
				verified = False
				break

	if verified:
		print(" Result verification successful")
	else:
		print(" Result verification failed")

	print(" class_npb = %c" % (npbparams.CLASS))
	return verified
#END verify()


@omp(compile=use_compiled())
def randlc_ft(x, a):
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
	x = t3 - t46 * t4
	return r46 * x, x
#END randlc_ft()


@omp(compile=use_compiled())
def randlc_seed_ft(x, a):
	_, x = randlc_ft(x, a)
	return x
#END randlc_seed_ft()


@omp(compile=use_compiled_types())
def randlc_seed_ft_types(x: float, a: float):
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
#END randlc_seed_ft_types()


@omp(compile=use_compiled())
def vranlc_complex_ft(n, x_seed, a, y):
	r23 = pow(0.5, 23.0)
	r46 = pow(0.5, 46.0)
	t23 = pow(2.0, 23.0)
	t46 = pow(2.0, 46.0)

	t1 = r23 * a
	a1 = int(t1)
	a2 = a - t23 * a1
	x = x_seed
	idx = 0

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

		if i % 2 == 0:
			y[idx] = complex(r46 * x, y[idx].imag)
		else:
			y[idx] = complex(y[idx].real, r46 * x)
			idx = idx + 1

	return x
#END vranlc_complex_ft()


@omp(compile=use_compiled())
def checksum(i,
		pointer_u1,
		d1,
		d2,
		d3,
		sums):
	u1 = numpy.reshape(pointer_u1, (d3, d2, d1))
	chk = complex(0.0, 0.0)

	with omp("parallel"):
		chk_worker = complex(0.0, 0.0)

		with omp("for"):
			for j in range(1, 1024 + 1):
				q = j % d1
				r = (3 * j) % d2
				s = (5 * j) % d3
				chk_worker = chk_worker + u1[s][r][q]

		with omp("critical"):
			chk = chk + chk_worker

	chk = chk / npbparams.NTOTAL
	print(" T =%5d     Checksum =%22.12e%22.12e" % (i, chk.real, chk.imag))
	sums[i] = chk
#END checksum()


@omp(compile=use_compiled_types())
def checksum_types(i: int,
		pointer_u1,
		d1: int,
		d2: int,
		d3: int,
		sums):
	u1: cython.doublecomplex[:] = pointer_u1
	sums_view: cython.doublecomplex[:] = sums
	chk_real: float = 0.0
	chk_imag: float = 0.0
	j: int
	q: int
	r: int
	s: int
	idx: int
	chk: cython.doublecomplex

	with omp("parallel reduction(+:chk_real,chk_imag)"):
		with omp("for"):
			for j in range(1, 1024 + 1):
				q = j % d1
				r = (3 * j) % d2
				s = (5 * j) % d3
				idx = (s * d2 + r) * d1 + q
				chk_real = chk_real + u1[idx].real
				chk_imag = chk_imag + u1[idx].imag

	chk = complex(chk_real / npbparams.NTOTAL, chk_imag / npbparams.NTOTAL)
	print(" T =%5d     Checksum =%22.12e%22.12e" % (i, chk.real, chk.imag))
	sums_view[i] = chk
#END checksum_types()


@omp(compile=use_compiled())
def evolve(pointer_u0,
		pointer_u1,
		pointer_twiddle,
		d1,
		d2,
		d3):
	u0 = numpy.reshape(pointer_u0, (d3, d2, d1))
	u1 = numpy.reshape(pointer_u1, (d3, d2, d1))
	twiddle = numpy.reshape(pointer_twiddle, (d3, d2, d1))

	with omp("parallel for"):
		for k in range(d3):
			for j in range(d2):
				for i in range(d1):
					u0[k][j][i] = u0[k][j][i] * twiddle[k][j][i]
					u1[k][j][i] = u0[k][j][i]
#END evolve()


@omp(compile=use_compiled_types())
def evolve_types(pointer_u0,
		pointer_u1,
		pointer_twiddle,
		d1: int,
		d2: int,
		d3: int):
	u0: cython.doublecomplex[:] = pointer_u0
	u1: cython.doublecomplex[:] = pointer_u1
	twiddle_view: cython.double[:] = pointer_twiddle
	total: int = d1 * d2 * d3
	idx: int
	value: cython.doublecomplex

	with omp("parallel for private(idx,value)"):
		for idx in range(total):
			value = u0[idx] * twiddle_view[idx]
			u0[idx] = value
			u1[idx] = value
#END evolve_types()


@omp(compile=use_compiled())
def fftz2(iss,
		l,
		m,
		n,
		ny,
		ny1,
		u,
		x,
		y):
	n1 = int(n / 2)
	lk = 1 << (l - 1)
	li = 1 << (m - l)
	lj = 2 * lk
	ku = li

	cplx_conj = numpy.conj
	for i in range(li):
		i11 = i * lk
		i12 = i11 + n1
		i21 = i * lj
		i22 = i21 + lk
		if iss >= 1:
			u1_local = u[ku + i]
		else:
			u1_local = cplx_conj(u[ku + i])

		for k in range(lk):
			for j in range(ny):
				x11 = x[i11 + k][j]
				x21 = x[i12 + k][j]
				y[i21 + k][j] = x11 + x21
				y[i22 + k][j] = u1_local * (x11 - x21)
#END fftz2()


@omp(compile=use_compiled())
def ilog2(n):
	if n == 1:
		return 0
	lg = 1
	nn = 2
	while nn < n:
		nn *= 2
		lg += 1
	return lg
#END ilog2()


@omp(compile=use_compiled())
def cfftz(iss,
		m,
		n,
		p_u,
		fftblock,
		fftblockpad,
		x,
		y):
	mx = int(p_u[0].real)
	if (iss != 1 and iss != -1) or m < 1 or m > mx:
		print("CFFTZ: Either U has not been initialized, or else\n"
				"one of the input parameters is invalid", iss, m, mx)

	for l in range(1, m + 1, 2):
		fftz2(iss, l, m, n, fftblock, fftblockpad, p_u, x, y)
		if l == m:
			for j in range(n):
				for i in range(fftblock):
					x[j][i] = y[j][i]
			break
		fftz2(iss, l + 1, m, n, fftblock, fftblockpad, p_u, y, x)
#END cfftz()


@omp(compile=use_compiled())
def cffts3(iss,
		d1,
		d2,
		d3,
		pointer_x,
		pointer_xout,
		p_u,
		fftblock,
		fftblockpad):
	x = numpy.reshape(pointer_x, (d3, d2, d1))
	xout = numpy.reshape(pointer_xout, (d3, d2, d1))

	logd3 = ilog2(d3)
	y1_pool = numpy.empty(shape=(get_worker_slots(), d3, fftblockpad), dtype=numpy.complex128)
	y2_pool = numpy.empty(shape=(get_worker_slots(), d3, fftblockpad), dtype=numpy.complex128)

	with omp("parallel"):
		thread_id = omp_get_thread_num()
		y1_local = y1_pool[thread_id]
		y2_local = y2_pool[thread_id]

		if timers_enabled:
			with omp("single"):
				c_timers.timer_start(T_FFTZ)

		with omp("for"):
			for j in range(d2):
				for ii in range(0, d1 - fftblock + 1, fftblock):
					for k in range(d3):
						for i in range(fftblock):
							y1_local[k][i] = x[k][j][i + ii]

					cfftz(iss, logd3, d3, p_u, fftblock, fftblockpad, y1_local, y2_local)
					for k in range(d3):
						for i in range(fftblock):
							xout[k][j][i + ii] = y1_local[k][i]

		if timers_enabled:
			with omp("single"):
				c_timers.timer_stop(T_FFTZ)
#END cffts3()


@omp(compile=use_compiled())
def cffts2(iss,
		d1,
		d2,
		d3,
		pointer_x,
		pointer_xout,
		p_u,
		fftblock,
		fftblockpad):
	x = numpy.reshape(pointer_x, (d3, d2, d1))
	xout = numpy.reshape(pointer_xout, (d3, d2, d1))

	logd2 = ilog2(d2)
	y1_pool = numpy.empty(shape=(get_worker_slots(), d2, fftblockpad), dtype=numpy.complex128)
	y2_pool = numpy.empty(shape=(get_worker_slots(), d2, fftblockpad), dtype=numpy.complex128)

	with omp("parallel"):
		thread_id = omp_get_thread_num()
		y1_local = y1_pool[thread_id]
		y2_local = y2_pool[thread_id]

		if timers_enabled:
			with omp("single"):
				c_timers.timer_start(T_FFTY)

		with omp("for"):
			for k in range(d3):
				for ii in range(0, d1 - fftblock + 1, fftblock):
					for j in range(d2):
						for i in range(fftblock):
							y1_local[j][i] = x[k][j][i + ii]

					cfftz(iss, logd2, d2, p_u, fftblock, fftblockpad, y1_local, y2_local)
					for j in range(d2):
						for i in range(fftblock):
							xout[k][j][i + ii] = y1_local[j][i]

		if timers_enabled:
			with omp("single"):
				c_timers.timer_stop(T_FFTY)
#END cffts2()


@omp(compile=use_compiled())
def cffts1(iss,
		d1,
		d2,
		d3,
		pointer_x,
		pointer_xout,
		p_u,
		fftblock,
		fftblockpad):
	x = numpy.reshape(pointer_x, (d3, d2, d1))
	xout = numpy.reshape(pointer_xout, (d3, d2, d1))

	logd1 = ilog2(d1)
	y1_pool = numpy.empty(shape=(get_worker_slots(), d1, fftblockpad), dtype=numpy.complex128)
	y2_pool = numpy.empty(shape=(get_worker_slots(), d1, fftblockpad), dtype=numpy.complex128)

	with omp("parallel"):
		thread_id = omp_get_thread_num()
		y1_local = y1_pool[thread_id]
		y2_local = y2_pool[thread_id]

		if timers_enabled:
			with omp("single"):
				c_timers.timer_start(T_FFTX)

		with omp("for"):
			for k in range(d3):
				for jj in range(0, d2 - fftblock + 1, fftblock):
					for j in range(fftblock):
						for i in range(d1):
							y1_local[i][j] = x[k][j + jj][i]

					cfftz(iss, logd1, d1, p_u, fftblock, fftblockpad, y1_local, y2_local)
					for j in range(fftblock):
						for i in range(d1):
							xout[k][j + jj][i] = y1_local[i][j]

		if timers_enabled:
			with omp("single"):
				c_timers.timer_stop(T_FFTX)
#END cffts1()


@omp(compile=use_compiled())
def fft(dirr,
		pointer_x1,
		pointer_x2,
		d1,
		d2,
		d3,
		p_u,
		fftblock,
		fftblockpad,
		maxdim):
	if dirr == 1:
		cffts1(1, d1, d2, d3, pointer_x1, pointer_x1, p_u, fftblock, fftblockpad)
		cffts2(1, d1, d2, d3, pointer_x1, pointer_x1, p_u, fftblock, fftblockpad)
		cffts3(1, d1, d2, d3, pointer_x1, pointer_x2, p_u, fftblock, fftblockpad)
	else:
		cffts3(-1, d1, d2, d3, pointer_x1, pointer_x1, p_u, fftblock, fftblockpad)
		cffts2(-1, d1, d2, d3, pointer_x1, pointer_x1, p_u, fftblock, fftblockpad)
		cffts1(-1, d1, d2, d3, pointer_x1, pointer_x2, p_u, fftblock, fftblockpad)
#END fft()


@omp(compile=use_compiled_types())
def ilog2_types(n: int):
	if n == 1:
		return 0
	lg: int = 1
	nn: int = 2
	while nn < n:
		nn *= 2
		lg += 1
	return lg
#END ilog2_types()


@omp(compile=use_compiled_types())
def fftz2_types(iss: int,
		l: int,
		m: int,
		n: int,
		ny: int,
		u_arg,
		x_arg,
		y_arg):
	u_view: cython.doublecomplex[:] = u_arg
	x: cython.doublecomplex[:, :] = x_arg
	y: cython.doublecomplex[:, :] = y_arg
	n1: int = n // 2
	lk: int = 1 << (l - 1)
	li: int = 1 << (m - l)
	lj: int = 2 * lk
	ku: int = li
	i: int
	j: int
	k: int
	i11: int
	i12: int
	i21: int
	i22: int
	u_value: cython.doublecomplex
	u1_local: cython.doublecomplex
	x11: cython.doublecomplex
	x21: cython.doublecomplex

	for i in range(li):
		i11 = i * lk
		i12 = i11 + n1
		i21 = i * lj
		i22 = i21 + lk
		u_value = u_view[ku + i]
		if iss >= 1:
			u1_local = u_value
		else:
			u1_local = complex(u_value.real, -u_value.imag)

		for k in range(lk):
			for j in range(ny):
				x11 = x[i11 + k][j]
				x21 = x[i12 + k][j]
				y[i21 + k][j] = x11 + x21
				y[i22 + k][j] = u1_local * (x11 - x21)
#END fftz2_types()


@omp(compile=use_compiled_types())
def cfftz_types(iss: int,
		m: int,
		n: int,
		p_u,
		fftblock: int,
		x,
		y):
	p_u_view: cython.doublecomplex[:] = p_u
	x_view: cython.doublecomplex[:, :] = x
	y_view: cython.doublecomplex[:, :] = y
	mx: int = int(p_u_view[0].real)
	l: int
	i: int
	j: int

	if (iss != 1 and iss != -1) or m < 1 or m > mx:
		print("CFFTZ: Either U has not been initialized, or else\n"
				"one of the input parameters is invalid", iss, m, mx)

	for l in range(1, m + 1, 2):
		fftz2_types(iss, l, m, n, fftblock, p_u_view, x_view, y_view)
		if l == m:
			for j in range(n):
				for i in range(fftblock):
					x_view[j][i] = y_view[j][i]
			break
		fftz2_types(iss, l + 1, m, n, fftblock, p_u_view, y_view, x_view)
#END cfftz_types()


@omp(compile=use_compiled_types())
def cffts1_types(iss: int,
		d1: int,
		d2: int,
		d3: int,
		pointer_x,
		pointer_xout,
		p_u,
		fftblock: int):
	x: cython.doublecomplex[:] = pointer_x
	xout: cython.doublecomplex[:] = pointer_xout
	y1: cython.doublecomplex[:, :, :] = numpy.empty((get_worker_slots(), d1, fftblock), dtype=numpy.complex128)
	y2: cython.doublecomplex[:, :, :] = numpy.empty((get_worker_slots(), d1, fftblock), dtype=numpy.complex128)
	logd1: int = ilog2_types(d1)
	thread_id: int = 0
	k: int = 0
	jj: int = 0
	j: int = 0
	i: int = 0
	src_idx: int = 0
	dst_idx: int = 0

	with omp("parallel private(thread_id,k,jj,j,i,src_idx,dst_idx)"):
		thread_id = omp_get_thread_num()
		with omp("for"):
			for k in range(d3):
				for jj in range(0, d2 - fftblock + 1, fftblock):
					for j in range(fftblock):
						for i in range(d1):
							src_idx = (k * d2 + (j + jj)) * d1 + i
							y1[thread_id, i, j] = x[src_idx]

					cfftz_types(iss, logd1, d1, p_u, fftblock, y1[thread_id], y2[thread_id])
					for j in range(fftblock):
						for i in range(d1):
							dst_idx = (k * d2 + (j + jj)) * d1 + i
							xout[dst_idx] = y1[thread_id, i, j]
#END cffts1_types()


@omp(compile=use_compiled_types())
def cffts2_types(iss: int,
		d1: int,
		d2: int,
		d3: int,
		pointer_x,
		pointer_xout,
		p_u,
		fftblock: int):
	x: cython.doublecomplex[:] = pointer_x
	xout: cython.doublecomplex[:] = pointer_xout
	y1: cython.doublecomplex[:, :, :] = numpy.empty((get_worker_slots(), d2, fftblock), dtype=numpy.complex128)
	y2: cython.doublecomplex[:, :, :] = numpy.empty((get_worker_slots(), d2, fftblock), dtype=numpy.complex128)
	logd2: int = ilog2_types(d2)
	thread_id: int = 0
	k: int = 0
	ii: int = 0
	j: int = 0
	i: int = 0
	src_idx: int = 0
	dst_idx: int = 0

	with omp("parallel private(thread_id,k,ii,j,i,src_idx,dst_idx)"):
		thread_id = omp_get_thread_num()
		with omp("for"):
			for k in range(d3):
				for ii in range(0, d1 - fftblock + 1, fftblock):
					for j in range(d2):
						for i in range(fftblock):
							src_idx = (k * d2 + j) * d1 + i + ii
							y1[thread_id, j, i] = x[src_idx]

					cfftz_types(iss, logd2, d2, p_u, fftblock, y1[thread_id], y2[thread_id])
					for j in range(d2):
						for i in range(fftblock):
							dst_idx = (k * d2 + j) * d1 + i + ii
							xout[dst_idx] = y1[thread_id, j, i]
#END cffts2_types()


@omp(compile=use_compiled_types())
def cffts3_types(iss: int,
		d1: int,
		d2: int,
		d3: int,
		pointer_x,
		pointer_xout,
		p_u,
		fftblock: int):
	x: cython.doublecomplex[:] = pointer_x
	xout: cython.doublecomplex[:] = pointer_xout
	y1: cython.doublecomplex[:, :, :] = numpy.empty((get_worker_slots(), d3, fftblock), dtype=numpy.complex128)
	y2: cython.doublecomplex[:, :, :] = numpy.empty((get_worker_slots(), d3, fftblock), dtype=numpy.complex128)
	logd3: int = ilog2_types(d3)
	thread_id: int = 0
	j: int = 0
	ii: int = 0
	k: int = 0
	i: int = 0
	src_idx: int = 0
	dst_idx: int = 0

	with omp("parallel private(thread_id,j,ii,k,i,src_idx,dst_idx)"):
		thread_id = omp_get_thread_num()
		with omp("for"):
			for j in range(d2):
				for ii in range(0, d1 - fftblock + 1, fftblock):
					for k in range(d3):
						for i in range(fftblock):
							src_idx = (k * d2 + j) * d1 + i + ii
							y1[thread_id, k, i] = x[src_idx]

					cfftz_types(iss, logd3, d3, p_u, fftblock, y1[thread_id], y2[thread_id])
					for k in range(d3):
						for i in range(fftblock):
							dst_idx = (k * d2 + j) * d1 + i + ii
							xout[dst_idx] = y1[thread_id, k, i]
#END cffts3_types()


@omp(compile=use_compiled_types())
def fft_types(dirr: int,
		pointer_x1,
		pointer_x2,
		d1: int,
		d2: int,
		d3: int,
		p_u,
		fftblock: int,
		fftblockpad: int,
		maxdim: int):
	if dirr == 1:
		cffts1_types(1, d1, d2, d3, pointer_x1, pointer_x1, p_u, fftblock)
		cffts2_types(1, d1, d2, d3, pointer_x1, pointer_x1, p_u, fftblock)
		cffts3_types(1, d1, d2, d3, pointer_x1, pointer_x2, p_u, fftblock)
	else:
		cffts3_types(-1, d1, d2, d3, pointer_x1, pointer_x1, p_u, fftblock)
		cffts2_types(-1, d1, d2, d3, pointer_x1, pointer_x1, p_u, fftblock)
		cffts1_types(-1, d1, d2, d3, pointer_x1, pointer_x2, p_u, fftblock)
#END fft_types()


@omp(compile=use_compiled())
def fft_init(n, u):
	m = ilog2(n)
	u[0] = complex(float(m), 0.0)
	ku = 2
	ln = 1

	m_cos, m_sin = math.cos, math.sin
	for j in range(1, m + 1):
		t = PI / ln
		for i in range(ln):
			ti = i * t
			u[i + ku - 1] = complex(m_cos(ti), m_sin(ti))
		ku = ku + ln
		ln = 2 * ln
#END fft_init()


@omp(compile=use_compiled())
def ipow46(a,
		exponent):
	result = 1
	if exponent == 0:
		return result
	q = a
	r = 1.0
	n = exponent

	while n > 1:
		n2 = int(n / 2)
		if (n2 * 2) == n:
			q = randlc_seed_ft(q, q)
			n = n2
		else:
			r = randlc_seed_ft(r, q)
			n = n - 1

	r = randlc_seed_ft(r, q)
	result = r
	return result
#END ipow46()


@omp(compile=use_compiled_types())
def ipow46_types(a: float,
		exponent: int):
	result: float = 1.0
	q: float
	r: float
	n: int
	n2: int

	if exponent == 0:
		return result

	q = a
	r = 1.0
	n = exponent

	while n > 1:
		n2 = int(n / 2)
		if (n2 * 2) == n:
			q = randlc_seed_ft_types(q, q)
			n = n2
		else:
			r = randlc_seed_ft_types(r, q)
			n = n - 1

	r = randlc_seed_ft_types(r, q)
	return r
#END ipow46_types()


@omp(compile=use_compiled())
def compute_initial_conditions(pointer_u0,
		d1,
		d2,
		d3):
	starts = numpy.empty(d3, numpy.float64)
	start = SEED

	an = ipow46(A, 0)
	start = randlc_seed_ft(start, an)
	an = ipow46(A, 2 * d1 * d2)

	starts[0] = start
	for kz in range(1, d3):
		start = randlc_seed_ft(start, an)
		starts[kz] = start

	with omp("parallel for"):
		for kp in range(d3):
			x0 = starts[kp]
			for j in range(d2):
				idx = (kp * d2 + j) * d1
				x0 = vranlc_complex_ft(2 * d1, x0, A, pointer_u0[idx:])
#END compute_initial_conditions()


@omp(compile=use_compiled_types())
def compute_initial_conditions_types(pointer_u0,
		d1: int,
		d2: int,
		d3: int):
	u0_view: cython.doublecomplex[:] = pointer_u0
	starts = numpy.empty(d3, numpy.float64)
	starts_view: cython.double[:] = starts
	start: float = SEED
	an: float = ipow46_types(A, 0)
	start = randlc_seed_ft_types(start, an)
	an = ipow46_types(A, 2 * d1 * d2)

	starts_view[0] = start
	for kz in range(1, d3):
		start = randlc_seed_ft_types(start, an)
		starts_view[kz] = start

	r23: float = pow(0.5, 23.0)
	r46: float = pow(0.5, 46.0)
	t23: float = pow(2.0, 23.0)
	t46: float = pow(2.0, 46.0)
	a: float = A
	kp: int
	j: int
	i: int
	row: int
	a1: int
	x1_seed: int
	t2_int: int
	t4_int: int
	t1: float
	t3: float
	a2: float
	x2_seed: float
	z: float
	x_seed: float
	real_value: float
	imag_value: float

	with omp("parallel for private(kp,j,i,row,a1,x1_seed,t2_int,t4_int,t1,t3,a2,x2_seed,z,x_seed,real_value,imag_value)"):
		for kp in range(d3):
			x_seed = starts_view[kp]
			for j in range(d2):
				row = (kp * d2 + j) * d1
				t1 = r23 * a
				a1 = int(t1)
				a2 = a - t23 * a1
				for i in range(d1):
					t1 = r23 * x_seed
					x1_seed = int(t1)
					x2_seed = x_seed - t23 * x1_seed
					t1 = a1 * x2_seed + a2 * x1_seed
					t2_int = int(r23 * t1)
					z = t1 - t23 * t2_int
					t3 = t23 * z + a2 * x2_seed
					t4_int = int(r46 * t3)
					x_seed = t3 - t46 * t4_int
					real_value = r46 * x_seed

					t1 = r23 * x_seed
					x1_seed = int(t1)
					x2_seed = x_seed - t23 * x1_seed
					t1 = a1 * x2_seed + a2 * x1_seed
					t2_int = int(r23 * t1)
					z = t1 - t23 * t2_int
					t3 = t23 * z + a2 * x2_seed
					t4_int = int(r46 * t3)
					x_seed = t3 - t46 * t4_int
					imag_value = r46 * x_seed

					u0_view[row + i] = complex(real_value, imag_value)
#END compute_initial_conditions_types()


@omp(compile=use_compiled())
def compute_indexmap(pointer_twiddle,
					d1,
					d2,
					d3):
	twiddle = numpy.reshape(pointer_twiddle, (d3, d2, d1))
	m_exp = math.exp
	ap = -4.0 * ALPHA * PI * PI

	with omp("parallel for"):
		for k in range(d3):
			kk = int(((k + d3 / 2) % d3) - d3 / 2)
			kk2 = kk * kk
			for j in range(d2):
				jj = int(((j + d2 / 2) % d2) - d2 / 2)
				kj2 = jj * jj + kk2
				for i in range(d1):
					ii = int(((i + d1 / 2) % d1) - d1 / 2)
					twiddle[k][j][i] = m_exp(ap * (ii * ii + kj2))
#END compute_indexmap()


@omp(compile=use_compiled_types())
def compute_indexmap_types(pointer_twiddle,
					d1: int,
					d2: int,
					d3: int):
	twiddle_view: cython.double[:] = pointer_twiddle
	ap: float = -4.0 * ALPHA * PI * PI
	k: int
	j: int
	i: int
	kk: int
	jj: int
	ii: int
	kk2: int
	kj2: int
	idx: int

	with omp("parallel for private(k,j,i,kk,jj,ii,kk2,kj2,idx)"):
		for k in range(d3):
			kk = int(((k + d3 / 2) % d3) - d3 / 2)
			kk2 = kk * kk
			for j in range(d2):
				jj = int(((j + d2 / 2) % d2) - d2 / 2)
				kj2 = jj * jj + kk2
				for i in range(d1):
					ii = int(((i + d1 / 2) % d1) - d1 / 2)
					idx = (k * d2 + j) * d1 + i
					twiddle_view[idx] = math.exp(ap * (ii * ii + kj2))
#END compute_indexmap_types()


def setup():
	global timers_enabled
	global niter
	global dims

	timers_enabled = os.path.isfile("timer.flag")
	niter = npbparams.NITER_DEFAULT

	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - FT Benchmark\n")
	print(" Size                : %4dx%4dx%4d" % (NX, NY, NZ))
	print(" Iterations                  :%7d" % (niter))
	print()

	dims[0] = NX
	dims[1] = NY
	dims[2] = NZ
#END setup()


def main():
	global sums, twiddle, u, u0, u1

	for i in range(T_MAX):
		c_timers.timer_clear(i)

	setup()
	if use_compiled_types():
		compute_indexmap_types(twiddle, dims[0], dims[1], dims[2])
	else:
		compute_indexmap(twiddle, dims[0], dims[1], dims[2])
	if use_compiled_types():
		compute_initial_conditions_types(u1, dims[0], dims[1], dims[2])
	else:
		compute_initial_conditions(u1, dims[0], dims[1], dims[2])
	fft_init(npbparams.MAXDIM, u)
	if use_compiled_types():
		fft_types(1, u1, u0, dims[0], dims[1], dims[2], u, FFTBLOCK, FFTBLOCKPAD, npbparams.MAXDIM)
	else:
		fft(1, u1, u0, dims[0], dims[1], dims[2], u, FFTBLOCK, FFTBLOCKPAD, npbparams.MAXDIM)

	for i in range(T_MAX):
		c_timers.timer_clear(i)

	c_timers.timer_start(T_TOTAL)
	if timers_enabled:
		c_timers.timer_start(T_SETUP)

	if use_compiled_types():
		compute_indexmap_types(twiddle, dims[0], dims[1], dims[2])
	else:
		compute_indexmap(twiddle, dims[0], dims[1], dims[2])
	if use_compiled_types():
		compute_initial_conditions_types(u1, dims[0], dims[1], dims[2])
	else:
		compute_initial_conditions(u1, dims[0], dims[1], dims[2])
	fft_init(npbparams.MAXDIM, u)

	if timers_enabled:
		c_timers.timer_stop(T_SETUP)
		c_timers.timer_start(T_FFT)

	if use_compiled_types():
		fft_types(1, u1, u0, dims[0], dims[1], dims[2], u, FFTBLOCK, FFTBLOCKPAD, npbparams.MAXDIM)
	else:
		fft(1, u1, u0, dims[0], dims[1], dims[2], u, FFTBLOCK, FFTBLOCKPAD, npbparams.MAXDIM)

	if timers_enabled:
		c_timers.timer_stop(T_FFT)

	for it in range(1, niter + 1):
		if timers_enabled:
			c_timers.timer_start(T_EVOLVE)

		if use_compiled_types():
			evolve_types(u0, u1, twiddle, dims[0], dims[1], dims[2])
		else:
			evolve(u0, u1, twiddle, dims[0], dims[1], dims[2])

		if timers_enabled:
			c_timers.timer_stop(T_EVOLVE)
			c_timers.timer_start(T_FFT)

		if use_compiled_types():
			fft_types(-1, u1, u1, dims[0], dims[1], dims[2], u, FFTBLOCK, FFTBLOCKPAD, npbparams.MAXDIM)
		else:
			fft(-1, u1, u1, dims[0], dims[1], dims[2], u, FFTBLOCK, FFTBLOCKPAD, npbparams.MAXDIM)

		if timers_enabled:
			c_timers.timer_stop(T_FFT)
			c_timers.timer_start(T_CHECKSUM)

		if use_compiled_types():
			checksum_types(it, u1, dims[0], dims[1], dims[2], sums)
		else:
			checksum(it, u1, dims[0], dims[1], dims[2], sums)

		if timers_enabled:
			c_timers.timer_stop(T_CHECKSUM)

	verified = verify(NX, NY, NZ, niter)

	c_timers.timer_stop(T_TOTAL)
	total_time = c_timers.timer_read(T_TOTAL)

	mflops = 0.0
	if total_time != 0.0:
		mflops = (1.0e-6 * npbparams.NTOTAL *
			(14.8157 + 7.19641 * math.log(npbparams.NTOTAL)
			 + (5.23518 + 7.21113 * math.log(npbparams.NTOTAL)) * niter)
			/ total_time)

	c_print_results.c_print_results("FT",
			npbparams.CLASS,
			npbparams.NX,
			npbparams.NY,
			npbparams.NZ,
			niter,
			total_time,
			mflops,
			"          floating point",
			verified)

	if timers_enabled:
		print_timers()
#END main()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-SER FT')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
	parser.add_argument("-m", "--mode", type=int, default=1, help="Mode: 0=pure, 1=hybrid, 2=compiled, 3=compiled with types")
	args = parser.parse_args()

	set_omp_mode(args.mode)
	set_omp_threads(args.threads)
	npbparams.set_ft_info(args.CLASS)
	set_global_variables()

	main()
