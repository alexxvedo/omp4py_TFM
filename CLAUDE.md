# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OMP4Py is a native Python implementation of the OpenMP programming model for multithreading parallelism. It provides a dual-runtime architecture: a pure Python runtime (`omp4py/runtime/`) and a native C-based runtime compiled via Cython (`omp4py/cruntime/`). The library is API-compliant with OpenMP 3.0 and requires Python >= 3.12 (free-threading via Python 3.13+ needed for actual parallel scaling).

## Build & Development

Package management uses **Poetry** (>= 2.0). Build system requires `poetry-core`, `setuptools`, and `Cython >= 3.1.0`.

```bash
poetry install                # install dependencies including test deps
pip install -e .              # editable install (pure mode only, no Cython compile)
```

The Cython native runtime is built by `scripts/compile.py`, invoked automatically by Poetry's build hook. On Python < 3.13 or when `OMP4PY_NO_COMPILE` is set, the native runtime is stubbed out (`omp4py_compiled = False`).

## Testing

Tests run in **separate subprocesses** via `test/utils/proctest.py` to isolate thread/GIL state. The `conftest.py` wraps each test function with `proctest(timeout=3)`.

```bash
poetry run pytest                        # run all tests with coverage
poetry run pytest test/basic/test_for.py  # run a single test file
poetry run pytest -k test_for_parallel   # run a single test by name
poetry run coverage html                 # generate coverage report
```

Test timeout is 3 seconds (configured in `pyproject.toml` under `[tool.pytest.ini_options]`).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OMP4PY_DUMP` | `false` | Save generated code to files for inspection |
| `OMP4PY_CACHE` | `false` | Cache compiled results across runs |
| `OMP4PY_FORCE` | `false` | Force recompilation even if cached |
| `OMP4PY_DEBUG` | `false` | Enable debug mode (AST dumps, Cython annotations) |
| `OMP4PY_COMPILE` | `false` | Compile decorated functions to native code via Cython |
| `OMP4PY_PURE` | `false` | Use pure Python runtime |
| `OMP4PY_FORCE_PURE` | `false` | Force pure runtime, skip cruntime import entirely |
| `OMP4PY_OPTIONS` | `{}` | JSON dict of compiler/parser options |
| `OMP4PY_DOPTIONS` | `{}` | JSON dict of default compiler/parser options |
| `OMP4PY_CACHE_DIR` | platform-dependent | Override cache directory |
| `OMP4PY_NO_COMPILE` | unset | Skip Cython compilation during build |

## Architecture

### How `@omp` works

The `omp` function serves as both a **decorator** and a **directive marker**:

1. **As decorator** (`@omp` or `@omp(compile=True)`): Triggers AST transformation of the decorated function/class. The source is re-parsed, OpenMP directives are extracted and processed, and new code is generated.
2. **As directive** (`with omp("parallel for reduction(+:x)")`): At parse time, these calls are recognized by the AST transformer and replaced with runtime calls. At runtime without the decorator, `omp(string)` returns a no-op context manager.

### Processing Pipeline

`omp4py/core/parser.py` — Entry point. `OmpTransformer` (AST NodeTransformer) walks the function's AST, finds `omp(...)` calls, and dispatches to processors.

`omp4py/core/directive/` — Tokenizer, parser, and schema for OpenMP directive strings. Converts `"parallel for reduction(+:x)"` into structured `OmpDirective`/`OmpClause` objects.

`omp4py/core/processor/` — Each OpenMP construct has a processor registered via `@omp_processor(name)` into `OMP_PROCESSOR` dict. Processors transform AST nodes, injecting runtime calls. Key modules: `parallelism.py`, `workdistribution.py`, `synchronization.py`, `tasking.py`, `reduction.py`, `combined.py`.

`omp4py/core/processor/builder.py` — Compiles the transformed AST back to executable code. Handles caching and Cython compilation (`compile=True` mode).

### Dual Runtime

`omp4py/runtime/` — Pure Python runtime. Thread management, work distribution, synchronization primitives, all implemented in Python using `threading`.

`omp4py/cruntime/` — Cython `.pxd` declaration files that mirror the runtime API. The actual `.py` source files are copied from `runtime/` and cythonized during build (see `scripts/compile.py`). The `__omp` module alias points to the active runtime; `__ompp` always points to the pure runtime.

### Four Execution Modes

- **Pure**: `from omp4py.pure import *` — forces pure Python runtime
- **Hybrid** (default): `from omp4py import *` — uses cruntime if compiled, falls back to pure
- **Compiled**: `@omp(compile=True)` — user function is also Cython-compiled
- **Compiled-with-types**: `@omp(compile=True)` with type annotations — enables full Cython type optimization

### Test Architecture

Tests define bare functions with `omp` directives, then call `omp(func)` explicitly to trigger transformation. Each test runs in a spawned subprocess (via `multiprocessing.get_context("spawn")`) to avoid thread state leakage. `tblib` is used to serialize tracebacks across process boundaries.
