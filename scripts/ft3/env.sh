#!/usr/bin/env bash
set -euo pipefail

export FT3_ROOT="${FT3_ROOT:-$HOME/omp4py_TFM}"

export PYTHON_313T_PREFIX="${PYTHON_313T_PREFIX:-$HOME/opt/python/3.13.13t}"
export PYTHON_314T_PREFIX="${PYTHON_314T_PREFIX:-$HOME/opt/python/3.14.4t}"
export PYTHON_315T_PREFIX="${PYTHON_315T_PREFIX:-$HOME/opt/python/3.15.0b1t}"

ft3_python_prefix() {
    case "$1" in
        3.13t) printf '%s\n' "$PYTHON_313T_PREFIX" ;;
        3.14t) printf '%s\n' "$PYTHON_314T_PREFIX" ;;
        3.15t) printf '%s\n' "$PYTHON_315T_PREFIX" ;;
        *) printf 'unknown Python tag: %s\n' "$1" >&2; return 2 ;;
    esac
}

ft3_python_bin_name() {
    case "$1" in
        3.13t) printf '%s\n' "python3.13t" ;;
        3.14t) printf '%s\n' "python3.14t" ;;
        3.15t) printf '%s\n' "python3.15t" ;;
        *) printf 'unknown Python tag: %s\n' "$1" >&2; return 2 ;;
    esac
}

ft3_venv_python() {
    printf '%s/.venv-%s/bin/python\n' "$FT3_ROOT" "$1"
}

ft3_export_python_runtime() {
    local tag="$1"
    local prefix
    prefix="$(ft3_python_prefix "$tag")"
    export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHON_GIL=0
}
