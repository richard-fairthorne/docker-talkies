#!/bin/bash
# Talkies integration test dispatcher.
#
# Each test_*.sh / e2e_*.sh in this directory is fully self-contained: it
# spawns its own --rm --gpus all CUDA container via harness.sh, runs its
# checks, tears the container down on exit. This script just runs all of
# them as subprocesses and summarises pass/fail.
#
# Per-file invocation works too — no shared orchestration required:
#
#     bash tests/integration/test_endpoints.sh
#     bash tests/integration/test_speech.sh
#     bash tests/integration/e2e_kokoro_nvidia.sh
#
# Env knobs:
#   TALKIES_SKIP_BUILD=1  skip `make build-cuda` — use whatever's tagged
#   HARNESS_IMAGE         override the docker image tag the tests use
#   HARNESS_CACHE_DIR     override the on-host /data cache dir
#   TALKIES_TEST_FILTER   only run files matching this glob (default *.sh)
#
# CLI args (optional): list of test file basenames to run. Empty = all.
#
#     bash tests/integration/run.sh test_speech.sh e2e_kokoro_nvidia.sh

set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# ── pre-flight ───────────────────────────────────────────────────────────────

command -v docker >/dev/null 2>&1 || { echo "FATAL: docker not on PATH" >&2; exit 2; }
command -v curl   >/dev/null 2>&1 || { echo "FATAL: curl not on PATH"   >&2; exit 2; }
command -v jq     >/dev/null 2>&1 || { echo "FATAL: jq not on PATH"     >&2; exit 2; }

if ! docker info 2>/dev/null | grep -qiE "nvidia|cdi:"; then
    echo "FATAL: docker daemon has no NVIDIA runtime — needs --gpus all" >&2
    exit 2
fi

# ── build (unless skipped) ───────────────────────────────────────────────────

if [ "${TALKIES_SKIP_BUILD:-0}" != "1" ]; then
    echo "[run] building CUDA image..."
    make build-cuda >/dev/null
fi

# ── collect test files ───────────────────────────────────────────────────────

_DIR="$(dirname "${BASH_SOURCE[0]}")"

shopt -s nullglob
declare -a TEST_FILES
for f in "$_DIR"/test_*.sh "$_DIR"/e2e_*.sh; do
    TEST_FILES+=("$(basename "$f")")
done
shopt -u nullglob

# Allow CLI selection by basename: `run.sh test_speech.sh e2e_kokoro_nvidia.sh`
if [ "$#" -gt 0 ]; then
    SELECTED=("$@")
else
    SELECTED=("${TEST_FILES[@]}")
fi

if [ "${#SELECTED[@]}" -eq 0 ]; then
    echo "[run] no test files found in $_DIR" >&2
    exit 1
fi

# ── run each as its own subprocess (own container, own port, --rm) ───────────

PASS=0
FAIL=0
FAILED=()

for tf in "${SELECTED[@]}"; do
    full="${_DIR}/${tf}"
    if [ ! -f "$full" ]; then
        echo ""
        echo "[run] SKIP: $tf — file not found"
        continue
    fi
    echo ""
    echo "============================================================="
    echo "  RUN  $tf"
    echo "============================================================="
    if bash "$full"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED+=("$tf")
    fi
done

# ── summary ──────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  integration suite: pass=$PASS fail=$FAIL total=$((PASS + FAIL))"
if [ "$FAIL" -ne 0 ]; then
    echo "  failed files:"
    for tf in "${FAILED[@]}"; do
        echo "    - $tf"
    done
fi
echo "═══════════════════════════════════════════════════════════"

[ "$FAIL" -eq 0 ]
