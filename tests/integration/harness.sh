#!/bin/bash
# shellcheck shell=bash
# Container lifecycle harness for talkies integration tests.
#
# Each test_*.sh / e2e_*.sh file sources this, declares the model slugs it
# needs, calls harness_start to spawn its own --rm --gpus all container on
# an ephemeral port, runs its checks via harness_run_tests, and the EXIT
# trap tears the container down. No shared state between files, no global
# orchestrator required — invoke any test file directly:
#
#     bash tests/integration/test_endpoints.sh
#     bash tests/integration/test_speech.sh
#     bash tests/integration/e2e_kokoro_nvidia.sh
#
# Env knobs (sane defaults; override only when needed):
#   HARNESS_IMAGE          docker image (default psyb0t/talkies:local-cuda)
#   HARNESS_CACHE_DIR      host dir for /data mount (default $REPO_ROOT/.e2e-cache).
#                          Reused across runs to skip the ~1 GB prefetch.
#   HARNESS_READY_TIMEOUT  seconds to wait for /healthz (default 900)
#   HARNESS_KEEP=1         leave container running on exit (debug)
#
# Sets for callers (read-only contract):
#   HARNESS_PORT           ephemeral host port the container is mapped to
#   TALKIES_BASE_URL       http://127.0.0.1:$HARNESS_PORT — consume from
#                          curl-based helpers in common.sh
#   HARNESS_ENABLED_MODELS comma-separated slugs the container is serving
#   HARNESS_CONTAINER      docker container name (for debugging)

HARNESS_IMAGE="${HARNESS_IMAGE:-psyb0t/talkies:local-cuda}"
HARNESS_READY_TIMEOUT="${HARNESS_READY_TIMEOUT:-900}"

# Default cache dir — resolved relative to the repo root, NOT pwd of the
# caller, so the same dir is reused whether you `cd` into tests/ first or
# not. We resolve once at source time.
_HARNESS_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS_CACHE_DIR="${HARNESS_CACHE_DIR:-${_HARNESS_REPO_ROOT}/.e2e-cache}"

# Populated by harness_start.
HARNESS_PORT=""
TALKIES_BASE_URL=""
HARNESS_ENABLED_MODELS=""
HARNESS_CONTAINER=""

# ── pre-flight ───────────────────────────────────────────────────────────────

harness_preflight() {
    local bin
    for bin in docker curl jq python3; do
        command -v "$bin" >/dev/null 2>&1 || {
            echo "FATAL: $bin not on PATH" >&2
            return 2
        }
    done
    if ! docker info 2>/dev/null | grep -qiE "nvidia|cdi:"; then
        echo "FATAL: docker daemon has no NVIDIA runtime — needs --gpus all" >&2
        return 2
    fi
    if ! docker image inspect "$HARNESS_IMAGE" >/dev/null 2>&1; then
        echo "FATAL: image $HARNESS_IMAGE not on host — build it first (make build-cuda)" >&2
        return 2
    fi
    mkdir -p "$HARNESS_CACHE_DIR"
    return 0
}

# ── container lifecycle ──────────────────────────────────────────────────────

# Pick a free ephemeral port via the kernel. Tiny race window between
# bind+close and docker -p — if it does collide, docker run fails fast and
# the caller bails.
_harness_pick_port() {
    python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

_harness_cleanup() {
    local rc=$?
    if [ "${HARNESS_KEEP:-0}" = "1" ] && [ -n "$HARNESS_CONTAINER" ]; then
        echo ""
        echo "[harness] HARNESS_KEEP=1 — leaving ${HARNESS_CONTAINER} on port ${HARNESS_PORT}"
        echo "          logs: docker logs -f ${HARNESS_CONTAINER}"
        echo "          rm:   docker rm -f ${HARNESS_CONTAINER}"
        return $rc
    fi
    if [ -n "$HARNESS_CONTAINER" ]; then
        echo ""
        echo "[harness] tearing down ${HARNESS_CONTAINER}"
        docker rm -f "$HARNESS_CONTAINER" >/dev/null 2>&1 || true
    fi
    return $rc
}

# harness_start <models_csv>
# Spawns the container, waits /healthz, sets the read-only contract vars.
# Exits the calling script on any setup failure (set -eo pipefail propagates).
harness_start() {
    local models="$1"
    if [ -z "$models" ]; then
        echo "FATAL: harness_start needs a comma-separated model list" >&2
        return 2
    fi

    harness_preflight || return $?

    HARNESS_PORT="$(_harness_pick_port)"
    TALKIES_BASE_URL="http://127.0.0.1:${HARNESS_PORT}"
    HARNESS_ENABLED_MODELS="$models"
    HARNESS_CONTAINER="talkies-e2e-$$-${RANDOM}"
    export TALKIES_BASE_URL HARNESS_PORT HARNESS_ENABLED_MODELS HARNESS_CONTAINER

    trap _harness_cleanup EXIT

    echo "[harness] starting ${HARNESS_CONTAINER}"
    echo "          image:  ${HARNESS_IMAGE}"
    echo "          port:   ${HARNESS_PORT}"
    echo "          cache:  ${HARNESS_CACHE_DIR}"
    echo "          models: ${HARNESS_ENABLED_MODELS}"

    docker run -d --rm --gpus all \
        --name "$HARNESS_CONTAINER" \
        -v "${HARNESS_CACHE_DIR}:/data" \
        -e TALKIES_DEVICE=cuda \
        -e TALKIES_ENABLED_MODELS="${HARNESS_ENABLED_MODELS}" \
        -p "${HARNESS_PORT}:8000" \
        "$HARNESS_IMAGE" >/dev/null

    echo "[harness] waiting for /healthz (timeout ${HARNESS_READY_TIMEOUT}s)..."
    local i
    for ((i = 0; i < HARNESS_READY_TIMEOUT; i += 2)); do
        if curl -sf --max-time 5 "${TALKIES_BASE_URL}/healthz" >/dev/null 2>&1; then
            echo "[harness] /healthz ok (after ${i}s)"
            return 0
        fi
        if ! docker inspect -f '{{.State.Running}}' "$HARNESS_CONTAINER" 2>/dev/null \
            | grep -q true; then
            echo "[harness] container exited during boot — last 80 lines:" >&2
            docker logs --tail 80 "$HARNESS_CONTAINER" >&2 2>&1 || true
            return 1
        fi
        sleep 2
    done
    echo "[harness] /healthz never came up in ${HARNESS_READY_TIMEOUT}s. Last logs:" >&2
    docker logs --tail 80 "$HARNESS_CONTAINER" >&2 2>&1 || true
    return 1
}

# ── test runner ──────────────────────────────────────────────────────────────

# harness_run_tests <test_func> [<test_func> ...]
# Invokes each named bash function, counts pass/fail, prints a summary, and
# returns 0 only if every test passed.
harness_run_tests() {
    local pass=0 fail=0
    local failed=()
    local t
    for t in "$@"; do
        echo ""
        echo "──[ $t ]──"
        if "$t"; then
            pass=$((pass + 1))
        else
            fail=$((fail + 1))
            failed+=("$t")
        fi
    done
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  $(basename "${BASH_SOURCE[1]:-suite}"): pass=$pass fail=$fail total=$((pass + fail))"
    if [ "$fail" -ne 0 ]; then
        echo "  failed:"
        for t in "${failed[@]}"; do
            echo "    - $t"
        done
    fi
    echo "═══════════════════════════════════════════════════════════"
    [ "$fail" -eq 0 ]
}
