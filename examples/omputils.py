import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MODE_NAMES = {
    -1: "pyomp",
    0: "pure",
    1: "hybrid",
    2: "compiled",
    3: "compiled with types",
}


def _valid_mode(value):
    return value if value in _MODE_NAMES else None


def _try_mode(value):
    try:
        return _valid_mode(int(value))
    except (TypeError, ValueError):
        return None


def _initial_mode():
    args = sys.argv[1:]

    if Path(sys.argv[0]).name == "main.py" and args:
        mode = _try_mode(args[0])
        if mode is not None:
            return mode

    for i, arg in enumerate(args):
        if arg in ("-m", "--mode") and i + 1 < len(args):
            mode = _try_mode(args[i + 1])
            if mode is not None:
                return mode
        elif arg.startswith("--mode="):
            mode = _try_mode(arg.split("=", 1)[1])
            if mode is not None:
                return mode

    return 1


def _mode_display(mode):
    return _MODE_NAMES[mode]


_mode = _initial_mode()
_threads = 1
use_pure = lambda: _mode == 0
use_compiled = lambda: _mode == 2 or _mode == 3
use_compiled_types = lambda: _mode == 3
use_pyomp = lambda: _mode == -1

try:
    from numba.openmp import njit
    from numba.openmp import omp_set_num_threads as pyomp_set_num_threads, openmp_context as pyomp

    has_pyomp = True
    print("pyomp found", file=sys.stderr)
except:
    has_pyomp = False
    pyomp = lambda *a, **k: None
    njit = lambda x: x

try:
    import omp4py as _omp4py
    import omp4py.cruntime as _omp4py_cruntime
    from omp4py import (
        omp,
        omp_get_max_threads as omp4py_runtime_get_max_threads,
        omp_get_num_threads as omp4py_runtime_get_num_threads,
        omp_get_thread_num as omp4py_runtime_get_thread_num,
        omp_set_num_threads as omp4py_runtime_set_num_threads,
    )
    from omp4py.pure import omp as omp_pure, omp_set_num_threads as omp4py_pure_set_num_threads

    _CRUNTIME_BUILD = PROJECT_ROOT / "build" / "libs" / "omp4py" / "cruntime"
    _compiled_api = None
    omp4py_compiled_set_num_threads = None
    omp4py_compiled_get_max_threads = None
    omp4py_compiled_get_num_threads = None
    omp4py_compiled_get_thread_num = None
    if _CRUNTIME_BUILD.is_dir():
        _cruntime_path = str(_CRUNTIME_BUILD)
        if _cruntime_path not in _omp4py_cruntime.__path__:
            _omp4py_cruntime.__path__.insert(0, _cruntime_path)
        try:
            from omp4py.cruntime import api as _compiled_api
        except ImportError:
            _compiled_api = None

    if _compiled_api is not None:
        omp4py_compiled_set_num_threads = _compiled_api.omp_set_num_threads
        omp4py_compiled_get_max_threads = _compiled_api.omp_get_max_threads
        omp4py_compiled_get_num_threads = _compiled_api.omp_get_num_threads
        omp4py_compiled_get_thread_num = _compiled_api.omp_get_thread_num

    has_omp4py = True
    print("omp4py found", file=sys.stderr)
    try:
        import pythran

        print("pythran found", file=sys.stderr)
    except ImportError:
        pass
except:
    has_omp4py = False
    omp = lambda *a, **k: (lambda *a, **k: None)
    omp_pure = lambda *a, **k: (lambda *a, **k: None)
    omp4py_runtime_set_num_threads = lambda n: None
    omp4py_runtime_get_max_threads = lambda: _threads
    omp4py_runtime_get_num_threads = lambda: _threads
    omp4py_runtime_get_thread_num = lambda: 0
    omp4py_compiled_set_num_threads = None
    omp4py_compiled_get_max_threads = None
    omp4py_compiled_get_num_threads = None
    omp4py_compiled_get_thread_num = None


def set_omp_threads(n):
    global _threads
    _threads = n
    if has_pyomp:
        pyomp_set_num_threads(n)

    if has_omp4py:
        omp4py_runtime_set_num_threads(n)
        if omp4py_compiled_set_num_threads is not None:
            omp4py_compiled_set_num_threads(n)
        omp4py_pure_set_num_threads(n)

    print("threads : " + str(n))


def get_omp_threads():
    return _threads

def omp_get_max_threads():
    if use_compiled() and omp4py_compiled_get_max_threads is not None:
        return omp4py_compiled_get_max_threads()
    return omp4py_runtime_get_max_threads()

def omp_get_num_threads():
    if use_compiled() and omp4py_compiled_get_num_threads is not None:
        return omp4py_compiled_get_num_threads()
    return omp4py_runtime_get_num_threads()

def omp_get_thread_num():
    if use_compiled() and omp4py_compiled_get_thread_num is not None:
        return omp4py_compiled_get_thread_num()
    return omp4py_runtime_get_thread_num()

def set_omp_mode(mode):
    global _mode
    mode = _valid_mode(mode)
    if mode is None:
        raise ValueError("invalid OpenMP mode")
    _mode = mode
    print("mode : " + _mode_display(mode))
