# ------------------------------------------------------------------------------
#
# The original NPB 3.4.1 version was written in Fortran and belongs to:
# 	http://www.nas.nasa.gov/Software/NPB/
#
# Authors of the Fortran code:
#	M. Yarrow
#	H. Jin
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

from omputils import (
	omp,
	omp_get_max_threads,
	omp_get_num_threads,
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
		long = []

# Local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import npbparams
import c_timers
import c_print_results


T_BENCHMARKING = 0
T_INITIALIZATION = 1
T_SORTING = 2
T_TOTAL_EXECUTION = 3

USE_BUCKETS = False

TOTAL_KEYS_LOG_2 = 16
MAX_KEY_LOG_2 = 11
NUM_BUCKETS_LOG_2 = 9

TOTAL_KEYS = 0
MAX_KEY = 0
NUM_BUCKETS = 0
NUM_KEYS = 0
SIZE_OF_BUFFERS = 0

MAX_ITERATIONS = 10
TEST_ARRAY_SIZE = 5

key_buff_ptr_global = None
passed_verification = 0

key_array = None
key_buff1 = None
key_buff2 = None
partial_verify_vals = None

test_rank_array = None
test_index_array = None

bucket_size = None
bucket_ptrs = None


def get_worker_slots():
	return max(1, get_omp_threads(), omp_get_max_threads(), os.cpu_count() or 1)
# END get_worker_slots()


def set_global_variables():
	global TOTAL_KEYS_LOG_2, MAX_KEY_LOG_2, NUM_BUCKETS_LOG_2
	global TOTAL_KEYS, MAX_KEY, NUM_BUCKETS, NUM_KEYS, SIZE_OF_BUFFERS
	global key_array, key_buff1, key_buff2, partial_verify_vals
	global bucket_ptrs, key_buff_ptr_global

	if npbparams.CLASS == "W":
		TOTAL_KEYS_LOG_2 = 20
		MAX_KEY_LOG_2 = 16
		NUM_BUCKETS_LOG_2 = 10
	elif npbparams.CLASS == "A":
		TOTAL_KEYS_LOG_2 = 23
		MAX_KEY_LOG_2 = 19
		NUM_BUCKETS_LOG_2 = 10
	elif npbparams.CLASS == "B":
		TOTAL_KEYS_LOG_2 = 25
		MAX_KEY_LOG_2 = 21
		NUM_BUCKETS_LOG_2 = 10
	elif npbparams.CLASS == "C":
		TOTAL_KEYS_LOG_2 = 27
		MAX_KEY_LOG_2 = 23
		NUM_BUCKETS_LOG_2 = 10
	elif npbparams.CLASS == "D":
		TOTAL_KEYS_LOG_2 = 31
		MAX_KEY_LOG_2 = 27
		NUM_BUCKETS_LOG_2 = 10

	TOTAL_KEYS = 1 << TOTAL_KEYS_LOG_2
	MAX_KEY = 1 << MAX_KEY_LOG_2
	NUM_BUCKETS = 1 << NUM_BUCKETS_LOG_2
	NUM_KEYS = TOTAL_KEYS
	SIZE_OF_BUFFERS = NUM_KEYS

	key_array = numpy.repeat(0, SIZE_OF_BUFFERS).astype(numpy.int64)
	key_buff1 = numpy.repeat(0, MAX_KEY).astype(numpy.int64)
	key_buff2 = numpy.repeat(0, SIZE_OF_BUFFERS).astype(numpy.int64)
	partial_verify_vals = numpy.repeat(0, TEST_ARRAY_SIZE).astype(numpy.int64)

	bucket_ptrs = numpy.zeros((get_worker_slots(), NUM_BUCKETS), dtype=numpy.int64)
	key_buff_ptr_global = numpy.repeat(0, MAX_KEY).astype(numpy.int64)
# END set_global_variables()


def create_verification_arrays():
	global test_index_array, test_rank_array

	if npbparams.CLASS == "S":
		test_index_array = numpy.array([48427, 17148, 23627, 62548, 4431], dtype=numpy.int64)
		test_rank_array = numpy.array([0, 18, 346, 64917, 65463], dtype=numpy.int64)
	elif npbparams.CLASS == "W":
		test_index_array = numpy.array([357773, 934767, 875723, 898999, 404505], dtype=numpy.int64)
		test_rank_array = numpy.array([1249, 11698, 1039987, 1043896, 1048018], dtype=numpy.int64)
	elif npbparams.CLASS == "A":
		test_index_array = numpy.array([2112377, 662041, 5336171, 3642833, 4250760], dtype=numpy.int64)
		test_rank_array = numpy.array([104, 17523, 123928, 8288932, 8388264], dtype=numpy.int64)
	elif npbparams.CLASS == "B":
		test_index_array = numpy.array([41869, 812306, 5102857, 18232239, 26860214], dtype=numpy.int64)
		test_rank_array = numpy.array([33422937, 10244, 59149, 33135281, 99], dtype=numpy.int64)
	elif npbparams.CLASS == "C":
		test_index_array = numpy.array([44172927, 72999161, 74326391, 129606274, 21736814], dtype=numpy.int64)
		test_rank_array = numpy.array([61147, 882988, 266290, 133997595, 133525895], dtype=numpy.int64)
	elif npbparams.CLASS == "D":
		test_index_array = numpy.array([1317351170, 995930646, 1157283250, 1503301535, 1453734525], dtype=numpy.int64)
		test_rank_array = numpy.array([1, 36538729, 1978098519, 2145192618, 2147425337], dtype=numpy.int64)
# END create_verification_arrays()


@omp(compile=use_compiled())
def randlc_is(x, a):
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
# END randlc_is()


@omp(compile=use_compiled())
def randlc_seed_is(x, a):
	_, x = randlc_is(x, a)
	return x
# END randlc_seed_is()


@omp(compile=use_compiled_types())
def randlc_seed_is_types(x: float, a: float):
	r23: float = pow(0.5, 23.0)
	r46: float = pow(0.5, 46.0)
	t23: float = pow(2.0, 23.0)
	t46: float = pow(2.0, 46.0)
	t1: float
	t2: cython.long
	t3: float
	t4: cython.long
	a1: cython.long
	a2: float
	x1: cython.long
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
# END randlc_seed_is_types()


@omp(compile=use_compiled())
def find_my_seed(kn, np_count, nn, s, a):
	if kn == 0:
		return s

	mq = int((nn / 4 + np_count - 1) / np_count)
	nq = int(mq * 4 * kn)

	t1 = s
	t2 = a
	kk = nq
	while kk > 1:
		ik = int(kk / 2)
		if (2 * ik) == kk:
			t2 = randlc_seed_is(t2, t2)
			kk = ik
		else:
			t1 = randlc_seed_is(t1, t2)
			kk = kk - 1

	t1 = randlc_seed_is(t1, t2)

	return t1
# END find_my_seed()


@omp(compile=use_compiled_types())
def find_my_seed_types(kn: int, np_count: int, nn: cython.long, s: float, a: float):
	if kn == 0:
		return s

	mq: cython.long = int((nn / 4 + np_count - 1) / np_count)
	nq: cython.long = int(mq * 4 * kn)
	t1: float = s
	t2: float = a
	kk: cython.long = nq
	ik: cython.long

	while kk > 1:
		ik = int(kk / 2)
		if (2 * ik) == kk:
			t2 = randlc_seed_is_types(t2, t2)
			kk = ik
		else:
			t1 = randlc_seed_is_types(t1, t2)
			kk = kk - 1

	t1 = randlc_seed_is_types(t1, t2)
	return t1
# END find_my_seed_types()


def create_seq(seed, a, p_key_array):
	an = a
	s = seed
	k = int(MAX_KEY / 4)

	for i in range(NUM_KEYS):
		x, s = randlc_is(s, an)
		x_aux, s = randlc_is(s, an)
		x = x + x_aux
		x_aux, s = randlc_is(s, an)
		x = x + x_aux
		x_aux, s = randlc_is(s, an)
		x = x + x_aux
		p_key_array[i] = int(k * x)
# END create_seq()



@omp(compile=use_compiled())
def create_seq_parallel(seed: float, a: float, p_key_array, num_keys, max_key):
	key_array_view = p_key_array
	r23 = pow(0.5, 23.0)
	r46 = pow(0.5, 46.0)
	t23 = pow(2.0, 23.0)
	t46 = pow(2.0, 46.0)
	key_scale = max_key / 4.0
	myid: int = 0
	num_procs: int = 1
	mq: int = 0
	start_key: int = 0
	end_key: int = 0
	s: float = 0.0
	a1: int = 0
	x1: int = 0
	t2: int = 0
	t4: int = 0
	a2: float = 0.0
	x2: float = 0.0
	z: float = 0.0
	x: float = 0.0
	t1: float = 0.0
	t3: float = 0.0
	i: int = 0
	draw: int = 0

	with omp("parallel private(myid,num_procs,mq,start_key,end_key,s,a1,a2,x,t1,t3,x1,x2,z,t2,t4,i,draw)"):
		myid = omp_get_thread_num()
		num_procs = omp_get_num_threads()

		mq = (num_keys + num_procs - 1) // num_procs
		start_key = myid * mq
		end_key = min(start_key + mq, num_keys)

		s = find_my_seed(myid, num_procs, 4 * num_keys, seed, a)
		t1 = r23 * a
		a1 = int(t1)
		a2 = a - t23 * a1

		for i in range(start_key, end_key):
			x = 0.0
			for draw in range(4):
				t1 = r23 * s
				x1 = int(t1)
				x2 = s - t23 * x1
				t1 = a1 * x2 + a2 * x1
				t2 = int(r23 * t1)
				z = t1 - t23 * t2
				t3 = t23 * z + a2 * x2
				t4 = int(r46 * t3)
				s = t3 - t46 * t4
				x = x + r46 * s
			key_array_view[i] = int(key_scale * x)
# END create_seq_parallel()


@omp(compile=use_compiled_types())
def create_seq_parallel_types(seed: float, a: float, p_key_array, num_keys: cython.long, max_key: cython.long):
	key_array_view: cython.long[:] = p_key_array
	r23: float = pow(0.5, 23.0)
	r46: float = pow(0.5, 46.0)
	t23: float = pow(2.0, 23.0)
	t46: float = pow(2.0, 46.0)
	key_scale: float = max_key / 4.0
	myid: int = 0
	num_procs: int = 1
	mq: cython.long = 0
	start_key: cython.long = 0
	end_key: cython.long = 0
	s: float = 0.0
	a1: cython.long = 0
	x1: cython.long = 0
	t2: cython.long = 0
	t4: cython.long = 0
	a2: float = 0.0
	x2: float = 0.0
	z: float = 0.0
	x: float = 0.0
	t1: float = 0.0
	t3: float = 0.0
	i: cython.long = 0
	draw: int = 0

	with omp("parallel private(myid,num_procs,mq,start_key,end_key,s,a1,a2,x,t1,t3,x1,x2,z,t2,t4,i,draw)"):
		myid = omp_get_thread_num()
		num_procs = omp_get_num_threads()

		mq = (num_keys + num_procs - 1) // num_procs
		start_key = myid * mq
		end_key = min(start_key + mq, num_keys)

		s = find_my_seed_types(myid, num_procs, 4 * num_keys, seed, a)
		t1 = r23 * a
		a1 = int(t1)
		a2 = a - t23 * a1

		for i in range(start_key, end_key):
			x = 0.0
			for draw in range(4):
				t1 = r23 * s
				x1 = int(t1)
				x2 = s - t23 * x1
				t1 = a1 * x2 + a2 * x1
				t2 = int(r23 * t1)
				z = t1 - t23 * t2
				t3 = t23 * z + a2 * x2
				t4 = int(r46 * t3)
				s = t3 - t46 * t4
				x = x + r46 * s
			key_array_view[i] = int(key_scale * x)
# END create_seq_parallel_types()


@omp(compile=False)
def create_seq_hybrid(seed, a, p_key_array):
	with omp("parallel"):
		myid = omp_get_thread_num()
		num_procs = omp_get_num_threads()
		k = int(MAX_KEY / 4)
		mq = (NUM_KEYS + num_procs - 1) // num_procs
		start_key = myid * mq
		end_key = min(start_key + mq, NUM_KEYS)
		s = find_my_seed(myid, num_procs, 4 * NUM_KEYS, seed, a)
		an = a
		for i in range(start_key, end_key):
			x, s = randlc_is(s, an)
			x_aux, s = randlc_is(s, an)
			x = x + x_aux
			x_aux, s = randlc_is(s, an)
			x = x + x_aux
			x_aux, s = randlc_is(s, an)
			x = x + x_aux
			p_key_array[i] = int(k * x)
# END create_seq_hybrid()


def alloc_key_buff():
	global bucket_size

	if USE_BUCKETS:
		bucket_size = numpy.zeros((get_worker_slots(), NUM_BUCKETS), dtype=numpy.int64)
		if use_compiled_types():
			clear_key_buff2_types(key_buff2, NUM_KEYS)
		else:
			clear_key_buff2(key_buff2, NUM_KEYS)
	return bucket_size
# END alloc_key_buff()


@omp(compile=use_compiled())
def clear_key_buff2(p_key_buff2, num_keys):
	key_buff2_view = p_key_buff2

	with omp("parallel for"):
		for i in range(num_keys):
			key_buff2_view[i] = 0
# END clear_key_buff2()


@omp(compile=use_compiled_types())
def clear_key_buff2_types(p_key_buff2, num_keys: cython.long):
	key_buff2_view: cython.long[:] = p_key_buff2
	i: cython.long

	with omp("parallel for"):
		for i in range(num_keys):
			key_buff2_view[i] = 0
# END clear_key_buff2_types()


@omp(compile=use_compiled())
def rank(iteration,
		p_key_array,
		p_key_buff1,
		p_key_buff2,
		p_partial_verify_vals,
		p_key_buff_ptr_global,
		p_bucket_size,
			p_bucket_ptrs,
			p_test_index_array,
			p_test_rank_array,
			class_npb: str,
			total_keys,
			max_key,
			num_buckets,
			num_keys,
			max_key_log_2,
			num_buckets_log_2):
	key_array_view = p_key_array
	key_buff1_view = p_key_buff1
	key_buff2_view = p_key_buff2
	partial_verify_view = p_partial_verify_vals
	key_buff_ptr_global_view = p_key_buff_ptr_global
	test_index_view = p_test_index_array
	test_rank_view = p_test_rank_array
	local_verification = 0
	shift = 0
	num_bucket_keys = 0
	if USE_BUCKETS:
		bucket_size_view = p_bucket_size
		bucket_ptrs_view = p_bucket_ptrs
		shift = max_key_log_2 - num_buckets_log_2
		num_bucket_keys = 1 << shift

	key_array_view[iteration] = iteration
	key_array_view[iteration + MAX_ITERATIONS] = max_key - iteration

	for verify_pos in range(TEST_ARRAY_SIZE):
		partial_verify_view[verify_pos] = key_array_view[test_index_view[verify_pos]]

	if USE_BUCKETS:
		myid = 0
		num_procs = 1
		init_bucket = 0
		key_pos = 0
		key = 0
		bucket_idx = 0
		prior_thread = 0
		next_thread = 0
		k1 = 0
		k2 = 0
		key_value = 0
		m = 0
		bucket_pos = 0
		with omp("parallel private(myid,num_procs,init_bucket,key_pos,key,bucket_idx,prior_thread,next_thread,k1,k2,key_value,m,bucket_pos)"):
			myid = omp_get_thread_num()
			num_procs = omp_get_num_threads()

			for init_bucket in range(num_buckets):
				bucket_size_view[myid, init_bucket] = 0

			with omp("for schedule(static)"):
				for key_pos in range(num_keys):
					bucket_idx = key_array_view[key_pos] >> shift
					bucket_size_view[myid, bucket_idx] = bucket_size_view[myid, bucket_idx] + 1

			bucket_ptrs_view[myid, 0] = 0
			for prior_thread in range(myid):
				bucket_ptrs_view[myid, 0] = bucket_ptrs_view[myid, 0] + bucket_size_view[prior_thread, 0]

			for bucket_idx in range(1, num_buckets):
				bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx - 1]
				for prior_thread in range(myid):
					bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx] + bucket_size_view[prior_thread, bucket_idx]
				for next_thread in range(myid, num_procs):
					bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx] + bucket_size_view[next_thread, bucket_idx - 1]

			with omp("for schedule(static)"):
				for key_pos in range(num_keys):
					key = key_array_view[key_pos]
					bucket_idx = key >> shift
					key_buff2_view[bucket_ptrs_view[myid, bucket_idx]] = key
					bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx] + 1

			if myid < (num_procs - 1):
				for bucket_idx in range(num_buckets):
					for next_thread in range(myid + 1, num_procs):
						bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx] + bucket_size_view[next_thread, bucket_idx]

			with omp("barrier"):
				pass

			with omp("for"):
				for bucket_idx in range(num_buckets):
					k1 = bucket_idx * num_bucket_keys
					k2 = k1 + num_bucket_keys
					for key_value in range(k1, k2):
						key_buff1_view[key_value] = 0

					m = bucket_ptrs_view[0, bucket_idx - 1] if bucket_idx > 0 else 0
					for bucket_pos in range(m, bucket_ptrs_view[0, bucket_idx]):
						key_buff1_view[key_buff2_view[bucket_pos]] = key_buff1_view[key_buff2_view[bucket_pos]] + 1

					key_buff1_view[k1] = key_buff1_view[k1] + m
					for key_value in range(k1 + 1, k2):
						key_buff1_view[key_value] = key_buff1_view[key_value] + key_buff1_view[key_value - 1]
	else:
		thread_key_pos = 0
		thread_key = 0
		thread_scan_key = 0
		thread_copy_key = 0
		prv_buff1 = None

		with omp("parallel private(thread_key_pos,thread_key,thread_scan_key,thread_copy_key,prv_buff1)"):
			with omp("single"):
				for clear_key in range(max_key):
					key_buff1_view[clear_key] = 0

			prv_buff1 = numpy.zeros(max_key, dtype=numpy.int64)

			with omp("for nowait"):
				for thread_key_pos in range(num_keys):
					thread_key = key_array_view[thread_key_pos]
					key_buff2_view[thread_key_pos] = thread_key
					prv_buff1[thread_key] = prv_buff1[thread_key] + 1

			for thread_scan_key in range(max_key - 1):
				prv_buff1[thread_scan_key + 1] = prv_buff1[thread_scan_key + 1] + prv_buff1[thread_scan_key]

			with omp("critical"):
				for thread_copy_key in range(max_key):
					key_buff1_view[thread_copy_key] = key_buff1_view[thread_copy_key] + prv_buff1[thread_copy_key]

	for verify_pos in range(TEST_ARRAY_SIZE):
		k = partial_verify_view[verify_pos]
		if 0 < k and k <= num_keys - 1:
			key_rank = key_buff1_view[k - 1]
			failed = False

			if class_npb == "S":
				if verify_pos <= 2:
					if key_rank != (test_rank_view[verify_pos] + iteration):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "W":
				if verify_pos < 2:
					if key_rank != (test_rank_view[verify_pos] + (iteration - 2)):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "A":
				if verify_pos <= 2:
					if key_rank != (test_rank_view[verify_pos] + (iteration - 1)):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - (iteration - 1)):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "B":
				if verify_pos == 1 or verify_pos == 2 or verify_pos == 4:
					if key_rank != (test_rank_view[verify_pos] + iteration):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "C":
				if verify_pos <= 2:
					if key_rank != (test_rank_view[verify_pos] + iteration):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "D":
				if verify_pos < 2:
					if key_rank != (test_rank_view[verify_pos] + iteration):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1

			if failed:
				print("Failed partial verification: iteration, ", iteration, ", test key ", verify_pos)

	if iteration == MAX_ITERATIONS:
		for copy_key in range(max_key):
			key_buff_ptr_global_view[copy_key] = key_buff1_view[copy_key]

	return local_verification
# END rank()


@omp(compile=use_compiled_types())
def rank_types(iteration,
		p_key_array,
		p_key_buff1,
		p_key_buff2,
		p_partial_verify_vals,
		p_key_buff_ptr_global,
		p_bucket_size,
			p_bucket_ptrs,
			p_test_index_array,
			p_test_rank_array,
			class_npb: str,
			total_keys: cython.long,
			max_key: cython.long,
			num_buckets: cython.long,
			num_keys: cython.long,
			max_key_log_2: cython.long,
			num_buckets_log_2: cython.long):
	key_array_view: cython.long[:] = p_key_array
	key_buff1_view: cython.long[:] = p_key_buff1
	key_buff2_view: cython.long[:] = p_key_buff2
	partial_verify_view: cython.long[:] = p_partial_verify_vals
	key_buff_ptr_global_view: cython.long[:] = p_key_buff_ptr_global
	test_index_view: cython.long[:] = p_test_index_array
	test_rank_view: cython.long[:] = p_test_rank_array
	local_verification: int = 0
	shift: cython.long = 0
	num_bucket_keys: cython.long = 0
	verify_pos: int
	k: cython.long
	key_rank: cython.long
	failed: bool
	clear_key: cython.long
	serial_key_pos: cython.long
	scan_key: cython.long
	copy_key: cython.long
	if USE_BUCKETS:
		bucket_size_view: cython.long[:, :] = p_bucket_size
		bucket_ptrs_view: cython.long[:, :] = p_bucket_ptrs
		shift = max_key_log_2 - num_buckets_log_2
		num_bucket_keys = 1 << shift

	key_array_view[iteration] = iteration
	key_array_view[iteration + MAX_ITERATIONS] = max_key - iteration

	for verify_pos in range(TEST_ARRAY_SIZE):
		partial_verify_view[verify_pos] = key_array_view[test_index_view[verify_pos]]

	if USE_BUCKETS:
		myid: cython.long = 0
		num_procs: cython.long = 1
		init_bucket: cython.long = 0
		key_pos: cython.long = 0
		key: cython.long = 0
		bucket_idx: cython.long = 0
		prior_thread: cython.long = 0
		next_thread: cython.long = 0
		k1: cython.long = 0
		k2: cython.long = 0
		key_value: cython.long = 0
		m: cython.long = 0
		bucket_pos: cython.long = 0
		with omp("parallel private(myid,num_procs,init_bucket,key_pos,key,bucket_idx,prior_thread,next_thread,k1,k2,key_value,m,bucket_pos)"):
			myid = omp_get_thread_num()
			num_procs = omp_get_num_threads()

			for init_bucket in range(num_buckets):
				bucket_size_view[myid, init_bucket] = 0

			with omp("for schedule(static)"):
				for key_pos in range(num_keys):
					bucket_idx = key_array_view[key_pos] >> shift
					bucket_size_view[myid, bucket_idx] = bucket_size_view[myid, bucket_idx] + 1

			bucket_ptrs_view[myid, 0] = 0
			for prior_thread in range(myid):
				bucket_ptrs_view[myid, 0] = bucket_ptrs_view[myid, 0] + bucket_size_view[prior_thread, 0]

			for bucket_idx in range(1, num_buckets):
				bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx - 1]
				for prior_thread in range(myid):
					bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx] + bucket_size_view[prior_thread, bucket_idx]
				for next_thread in range(myid, num_procs):
					bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx] + bucket_size_view[next_thread, bucket_idx - 1]

			with omp("for schedule(static)"):
				for key_pos in range(num_keys):
					key = key_array_view[key_pos]
					bucket_idx = key >> shift
					key_buff2_view[bucket_ptrs_view[myid, bucket_idx]] = key
					bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx] + 1

			if myid < (num_procs - 1):
				for bucket_idx in range(num_buckets):
					for next_thread in range(myid + 1, num_procs):
						bucket_ptrs_view[myid, bucket_idx] = bucket_ptrs_view[myid, bucket_idx] + bucket_size_view[next_thread, bucket_idx]

			with omp("barrier"):
				pass

			with omp("for"):
				for bucket_idx in range(num_buckets):
					k1 = bucket_idx * num_bucket_keys
					k2 = k1 + num_bucket_keys
					for key_value in range(k1, k2):
						key_buff1_view[key_value] = 0

					m = bucket_ptrs_view[0, bucket_idx - 1] if bucket_idx > 0 else 0
					for bucket_pos in range(m, bucket_ptrs_view[0, bucket_idx]):
						key_buff1_view[key_buff2_view[bucket_pos]] = key_buff1_view[key_buff2_view[bucket_pos]] + 1

					key_buff1_view[k1] = key_buff1_view[k1] + m
					for key_value in range(k1 + 1, k2):
						key_buff1_view[key_value] = key_buff1_view[key_value] + key_buff1_view[key_value - 1]
	else:
		thread_key_pos: cython.long = 0
		thread_key: cython.long = 0
		thread_scan_key: cython.long = 0
		thread_copy_key: cython.long = 0
		prv_buff1_values = None
		prv_buff1: cython.long[:] = None

		with omp("parallel private(thread_key_pos,thread_key,thread_scan_key,thread_copy_key,prv_buff1_values,prv_buff1)"):
			with omp("single"):
				for clear_key in range(max_key):
					key_buff1_view[clear_key] = 0

			prv_buff1_values = numpy.zeros(max_key, dtype=numpy.int64)
			prv_buff1 = prv_buff1_values

			with omp("for nowait"):
				for thread_key_pos in range(num_keys):
					thread_key = key_array_view[thread_key_pos]
					key_buff2_view[thread_key_pos] = thread_key
					prv_buff1[thread_key] = prv_buff1[thread_key] + 1

			for thread_scan_key in range(max_key - 1):
				prv_buff1[thread_scan_key + 1] = prv_buff1[thread_scan_key + 1] + prv_buff1[thread_scan_key]

			with omp("critical"):
				for thread_copy_key in range(max_key):
					key_buff1_view[thread_copy_key] = key_buff1_view[thread_copy_key] + prv_buff1[thread_copy_key]

	for verify_pos in range(TEST_ARRAY_SIZE):
		k = partial_verify_view[verify_pos]
		if 0 < k and k <= num_keys - 1:
			key_rank = key_buff1_view[k - 1]
			failed = False

			if class_npb == "S":
				if verify_pos <= 2:
					if key_rank != (test_rank_view[verify_pos] + iteration):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "W":
				if verify_pos < 2:
					if key_rank != (test_rank_view[verify_pos] + (iteration - 2)):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "A":
				if verify_pos <= 2:
					if key_rank != (test_rank_view[verify_pos] + (iteration - 1)):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - (iteration - 1)):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "B":
				if verify_pos == 1 or verify_pos == 2 or verify_pos == 4:
					if key_rank != (test_rank_view[verify_pos] + iteration):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "C":
				if verify_pos <= 2:
					if key_rank != (test_rank_view[verify_pos] + iteration):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1
			elif class_npb == "D":
				if verify_pos < 2:
					if key_rank != (test_rank_view[verify_pos] + iteration):
						failed = True
					else:
						local_verification = local_verification + 1
				else:
					if key_rank != (test_rank_view[verify_pos] - iteration):
						failed = True
					else:
						local_verification = local_verification + 1

			if failed:
				print("Failed partial verification: iteration, ", iteration, ", test key ", verify_pos)

	if iteration == MAX_ITERATIONS:
		for copy_key in range(max_key):
			key_buff_ptr_global_view[copy_key] = key_buff1_view[copy_key]

	return local_verification
# END rank_types()


@omp(compile=use_compiled())
def full_verify(p_key_buff_ptr_global,
		p_key_array,
		p_key_buff2,
		p_bucket_ptrs,
		num_buckets,
		num_keys):
	key_buff_ptr_global_view = p_key_buff_ptr_global
	key_array_view = p_key_array
	key_buff2_view = p_key_buff2
	j = 0
	

	for i in range(num_keys):
		key_buff2_view[i] = key_array_view[i]

	for i in range(num_keys):
		key_buff_ptr_global_view[key_buff2_view[i]] = key_buff_ptr_global_view[key_buff2_view[i]] - 1
		k = key_buff_ptr_global_view[key_buff2_view[i]]
		key_array_view[k] = key_buff2_view[i]

	for verify_pos in range(1, num_keys):
		if key_array_view[verify_pos - 1] > key_array_view[verify_pos]:
			j = j + 1

	if j != 0:
		print("Full_verify: number of keys out of sort: ", j)
		return 0

	return 1
# END full_verify()


@omp(compile=use_compiled_types())
def full_verify_types(p_key_buff_ptr_global,
		p_key_array,
		p_key_buff2,
		p_bucket_ptrs,
		num_buckets: cython.long,
		num_keys: cython.long):
	key_buff_ptr_global_view: cython.long[:] = p_key_buff_ptr_global
	key_array_view: cython.long[:] = p_key_array
	key_buff2_view: cython.long[:] = p_key_buff2
	j: cython.long = 0
	i: cython.long
	k: cython.long
	key_value: cython.long
	verify_pos: cython.long

	for i in range(num_keys):
		key_buff2_view[i] = key_array_view[i]

	for i in range(num_keys):
		key_value = key_buff2_view[i]
		key_buff_ptr_global_view[key_value] = key_buff_ptr_global_view[key_value] - 1
		k = key_buff_ptr_global_view[key_value]
		key_array_view[k] = key_value

	for verify_pos in range(1, num_keys):
		if key_array_view[verify_pos - 1] > key_array_view[verify_pos]:
			j = j + 1

	if j != 0:
		print("Full_verify: number of keys out of sort: ", j)
		return 0

	return 1
# END full_verify_types()


def main():
	global passed_verification
	global USE_BUCKETS

	USE_BUCKETS = False

	timer_on = os.path.isfile("timer.flag")
	c_timers.timer_clear(T_BENCHMARKING)
	if timer_on:
		c_timers.timer_clear(T_INITIALIZATION)
		c_timers.timer_clear(T_SORTING)
		c_timers.timer_clear(T_TOTAL_EXECUTION)

	if timer_on:
		c_timers.timer_start(T_TOTAL_EXECUTION)

	create_verification_arrays()

	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - IS Benchmark\n")
	print(" Size:  %ld  (class %s)  (%s)" % (TOTAL_KEYS, npbparams.CLASS, ("Using buckets" if USE_BUCKETS else "Not using buckets")))
	print(" Iterations:   %d\n" % (MAX_ITERATIONS))

	if timer_on:
		c_timers.timer_start(T_INITIALIZATION)

	if use_compiled_types():
		create_seq_parallel_types(314159265.00,
				1220703125.00,
				key_array,
				NUM_KEYS,
				MAX_KEY)
	elif use_compiled():
		create_seq_parallel(314159265.00,
				1220703125.00,
				key_array,
				NUM_KEYS,
				MAX_KEY)
	elif not use_pure():
		create_seq_hybrid(314159265.00,
				1220703125.00,
				key_array)
	else:
		create_seq(314159265.00,
				1220703125.00,
				key_array)

	local_bucket_size = alloc_key_buff()
	if timer_on:
		c_timers.timer_stop(T_INITIALIZATION)

	_rank_fn = rank_types if use_compiled_types() else rank
	_rank_fn(1,
		key_array, key_buff1, key_buff2,
		partial_verify_vals,
		key_buff_ptr_global,
		local_bucket_size, bucket_ptrs,
		test_index_array, test_rank_array,
		npbparams.CLASS,
		TOTAL_KEYS, MAX_KEY, NUM_BUCKETS, NUM_KEYS,
		MAX_KEY_LOG_2, NUM_BUCKETS_LOG_2)

	passed_verification = 0

	if npbparams.CLASS != "S":
		print("\n   iteration")

	c_timers.timer_start(T_BENCHMARKING)

	for iteration in range(1, MAX_ITERATIONS + 1):
		if npbparams.CLASS != "S":
			print("        %d" % (iteration))
		passed_verification = passed_verification + _rank_fn(iteration,
								key_array, key_buff1, key_buff2,
								partial_verify_vals,
								key_buff_ptr_global,
								local_bucket_size, bucket_ptrs,
								test_index_array, test_rank_array,
								npbparams.CLASS,
								TOTAL_KEYS, MAX_KEY, NUM_BUCKETS, NUM_KEYS,
								MAX_KEY_LOG_2, NUM_BUCKETS_LOG_2)

	c_timers.timer_stop(T_BENCHMARKING)
	timecounter = c_timers.timer_read(T_BENCHMARKING)

	if timer_on:
		c_timers.timer_start(T_SORTING)
	_full_verify_fn = full_verify_types if use_compiled_types() else full_verify
	passed_verification = passed_verification + _full_verify_fn(key_buff_ptr_global,
									key_array, key_buff2, bucket_ptrs,
									NUM_BUCKETS, NUM_KEYS)
	if timer_on:
		c_timers.timer_stop(T_SORTING)

	if timer_on:
		c_timers.timer_stop(T_TOTAL_EXECUTION)

	if passed_verification != (5 * MAX_ITERATIONS + 1):
		passed_verification = 0

	c_print_results.c_print_results("IS",
			npbparams.CLASS,
			int(TOTAL_KEYS / 64),
			64,
			0,
			MAX_ITERATIONS,
			timecounter,
			(MAX_ITERATIONS * TOTAL_KEYS) / timecounter / 1000000.0,
			"keys ranked",
			passed_verification > 0)

	if timer_on:
		t_total = c_timers.timer_read(T_TOTAL_EXECUTION)
		print("\nAdditional timers -")
		print(" Total execution: %8.3f" % (t_total))
		if t_total == 0.0:
			t_total = 1.0
		timecounter = c_timers.timer_read(T_INITIALIZATION)
		t_percent = timecounter / t_total * 100.0
		print(" Initialization : %8.3f (%5.2f%%)" % (timecounter, t_percent))
		timecounter = c_timers.timer_read(T_BENCHMARKING)
		t_percent = timecounter / t_total * 100.0
		print(" Benchmarking   : %8.3f (%5.2f%%)" % (timecounter, t_percent))
		timecounter = c_timers.timer_read(T_SORTING)
		t_percent = timecounter / t_total * 100.0
		print(" Sorting        : %8.3f (%5.2f%%)" % (timecounter, t_percent))
# END main()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="NPB-PYTHON-SER IS")
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
	parser.add_argument("-m", "--mode", type=int, default=1, help="Mode: 0=pure, 1=hybrid, 2=compiled, 3=compiled with types")
	args = parser.parse_args()

	set_omp_mode(args.mode)
	set_omp_threads(args.threads)
	npbparams.set_is_info(args.CLASS)
	set_global_variables()

	main()
