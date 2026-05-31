#!/usr/bin/env bash
# Compile hash-locked requirements files for the heavy ML stack.
# Run via: make compile-heavy
# Outputs: requirements-heavy-cpu.txt  requirements-heavy-cuda.txt
#
# Requires: uv (astral.sh/uv). Both files are committed; Dockerfiles install
# from them with --require-hashes so every wheel byte is verified.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPU_IN="$REPO_ROOT/scripts/heavy-deps-cpu.in"
CUDA_IN="$REPO_ROOT/scripts/heavy-deps-cuda.in"
CPU_OUT="$REPO_ROOT/requirements-heavy-cpu.txt"
CUDA_OUT="$REPO_ROOT/requirements-heavy-cuda.txt"

for f in "$CPU_IN" "$CUDA_IN"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

echo "[compile-heavy] CPU variant → requirements-heavy-cpu.txt"
uv pip compile \
    --python-version 3.12 \
    --generate-hashes \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --no-header \
    --no-config \
    -o "$CPU_OUT" \
    "$CPU_IN"

echo "[compile-heavy] CUDA variant → requirements-heavy-cuda.txt"
uv pip compile \
    --python-version 3.12 \
    --generate-hashes \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    --no-header \
    --no-config \
    -o "$CUDA_OUT" \
    "$CUDA_IN"

echo "[compile-heavy] done"
