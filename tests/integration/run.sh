#!/bin/bash
# Talkies integration test driver — CUDA only.
#
# Builds the CUDA image, starts a fresh container with --gpus all, waits for
# /healthz, runs every test_*.sh file in this directory, prints a summary,
# and tears the container down.
#
# Why CUDA-only: the alternative (CPU image + faster-whisper-large on a
# desktop CPU) takes >30 min per transcription, which is useless for a
# regression suite. Whisper-turbo on a GPU is sub-second.
#
# Env knobs:
#   TALKIES_TEST_PORT       host port to publish (default 18000)
#   TALKIES_TEST_CACHE      host dir for model cache (default ~/.talkies-models)
#   TALKIES_TEST_IMAGE      image to use (default psyb0t/talkies:local-cuda)
#   TALKIES_TEST_KEEP=1     don't `docker rm` the container at exit (debug)
#   TALKIES_SKIP_BUILD=1    skip `make build-cuda` — use whatever's tagged
#   TALKIES_ENABLED_MODELS  comma slugs to download; empty = all 7 (default)

set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TALKIES_TEST_PORT="${TALKIES_TEST_PORT:-18000}"
TALKIES_TEST_CACHE="${TALKIES_TEST_CACHE:-$HOME/.talkies-models}"
TALKIES_TEST_IMAGE="${TALKIES_TEST_IMAGE:-psyb0t/talkies:local-cuda}"
CONTAINER_NAME="talkies-integration-test-$$"

export TALKIES_TEST_PORT
export TALKIES_BASE_URL="http://127.0.0.1:${TALKIES_TEST_PORT}"

# Caller may set TALKIES_ENABLED_MODELS to scope what gets downloaded.
# Empty means "every model in models.json" — fine when the cache is warm.
TALKIES_ENABLED_MODELS="${TALKIES_ENABLED_MODELS:-}"
export TALKIES_ENABLED_MODELS

# ── pre-flight ───────────────────────────────────────────────────────────────

command -v docker >/dev/null 2>&1 || { echo "FATAL: docker not on PATH" >&2; exit 2; }
command -v curl   >/dev/null 2>&1 || { echo "FATAL: curl not on PATH"   >&2; exit 2; }
command -v jq     >/dev/null 2>&1 || { echo "FATAL: jq not on PATH"     >&2; exit 2; }

if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "FATAL: docker daemon doesn't report an NVIDIA runtime — this suite needs --gpus all." >&2
    echo "       Install nvidia-container-toolkit and restart dockerd." >&2
    exit 2
fi

mkdir -p "$TALKIES_TEST_CACHE"

# ── build (unless skipped) ───────────────────────────────────────────────────

if [ "${TALKIES_SKIP_BUILD:-0}" != "1" ]; then
    echo "[run] building CUDA image ($TALKIES_TEST_IMAGE)..."
    make build-cuda >/dev/null
fi

# ── start container ──────────────────────────────────────────────────────────

cleanup() {
    if [ "${TALKIES_TEST_KEEP:-0}" = "1" ]; then
        echo "[run] TALKIES_TEST_KEEP=1 — leaving container $CONTAINER_NAME running"
        echo "      tail logs: docker logs -f $CONTAINER_NAME"
        echo "      remove:    docker rm -f $CONTAINER_NAME"
        return
    fi
    echo "[run] stopping $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[run] launching $CONTAINER_NAME (port=$TALKIES_TEST_PORT cache=$TALKIES_TEST_CACHE)"
docker run -d --rm --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$TALKIES_TEST_CACHE:/data" \
    -e TALKIES_DEVICE=cuda \
    -e TALKIES_ENABLED_MODELS="$TALKIES_ENABLED_MODELS" \
    -p "${TALKIES_TEST_PORT}:8000" \
    "$TALKIES_TEST_IMAGE" >/dev/null

# ── wait for ready ───────────────────────────────────────────────────────────

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "[run] waiting for talkies /healthz (first boot may download all weights)..."
if ! talkies_wait_ready "${TALKIES_READY_TIMEOUT:-1800}"; then
    echo "[run] talkies never came up — last 80 log lines:"
    docker logs --tail 80 "$CONTAINER_NAME" || true
    exit 1
fi
echo "[run] talkies is ready"

# ── load test files ──────────────────────────────────────────────────────────

shopt -s nullglob
TEST_FILES=("$(dirname "${BASH_SOURCE[0]}")"/test_*.sh)
shopt -u nullglob

for f in "${TEST_FILES[@]}"; do
    # shellcheck disable=SC1090
    source "$f"
done

# ── run + summarize ──────────────────────────────────────────────────────────

if [ "${#ALL_TESTS[@]}" -eq 0 ]; then
    echo "[run] no tests registered — nothing to do" >&2
    exit 1
fi

# Allow CLI selection: `run.sh test_talkies_healthz test_talkies_models_list`
if [ "$#" -gt 0 ]; then
    SELECTED=("$@")
else
    SELECTED=("${ALL_TESTS[@]}")
fi

PASS=0
FAIL=0
FAILED_TESTS=()
for t in "${SELECTED[@]}"; do
    echo ""
    echo "──[ $t ]──"
    if "$t"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$t")
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  pass=$PASS fail=$FAIL total=$((PASS + FAIL))"
if [ "$FAIL" -ne 0 ]; then
    echo "  failed:"
    for t in "${FAILED_TESTS[@]}"; do
        echo "    - $t"
    done
fi
echo "═══════════════════════════════════════════════════════════"

[ "$FAIL" -eq 0 ]
