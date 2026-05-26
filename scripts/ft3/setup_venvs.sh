#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

mkdir -p "$FT3_ROOT"
cd "$FT3_ROOT"

if command -v module >/dev/null 2>&1; then
    module load gcc/12.3.0 >/dev/null 2>&1 || true
fi

tags=("$@")
if [ "${#tags[@]}" -eq 0 ]; then
    tags=(3.13t 3.14t 3.15t)
fi

for tag in "${tags[@]}"; do
    prefix="$(ft3_python_prefix "$tag")"
    bin_name="$(ft3_python_bin_name "$tag")"
    base_python="$prefix/bin/$bin_name"
    venv_dir="$FT3_ROOT/.venv-$tag"

    if [ ! -x "$base_python" ]; then
        printf 'missing Python executable: %s\n' "$base_python" >&2
        exit 1
    fi

    ft3_export_python_runtime "$tag"
    "$base_python" -m venv "$venv_dir"
    "$venv_dir/bin/python" -m pip install -U pip setuptools wheel
    "$venv_dir/bin/python" -m pip install -U "Cython>=3.1.0" numpy
    "$venv_dir/bin/python" -m pip install -e "$FT3_ROOT"
    "$venv_dir/bin/python" -c 'import sys; print(sys.version); print("gil_enabled =", sys._is_gil_enabled())'
done
