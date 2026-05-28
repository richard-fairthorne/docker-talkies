#!/bin/bash
# shellcheck shell=bash
# Shared helpers for talkies integration tests.
#
# Unlike the aigate harness this doesn't `docker compose exec` into a sidecar
# — the test runner starts the talkies container with a published port and
# we hit http://127.0.0.1:$TALKIES_TEST_PORT directly.

TALKIES_TEST_PORT="${TALKIES_TEST_PORT:-18000}"
TALKIES_BASE_URL="http://127.0.0.1:${TALKIES_TEST_PORT}"

# shellcheck disable=SC2034  # populated by test_*.sh files, consumed by run.sh
ALL_TESTS=()

# ── assertions ───────────────────────────────────────────────────────────────

assert_eq() {
    local actual="$1" expected="$2" name="$3"
    if [ "$actual" = "$expected" ]; then
        echo "  OK: $name"
        return 0
    fi
    echo "  FAIL: $name: expected '$expected', got '$actual'"
    return 1
}

assert_contains() {
    local actual="$1" expected="$2" name="$3"
    if [[ "$actual" == *"$expected"* ]]; then
        echo "  OK: $name"
        return 0
    fi
    echo "  FAIL: $name: expected to contain '$expected'"
    echo "  actual: ${actual:0:500}"
    return 1
}

assert_not_empty() {
    local actual="$1" name="$2"
    if [ -n "$actual" ]; then
        echo "  OK: $name"
        return 0
    fi
    echo "  FAIL: $name: expected non-empty output"
    return 1
}

# ── HTTP helpers ─────────────────────────────────────────────────────────────

talkies_get() {
    curl -sf --max-time 30 "${TALKIES_BASE_URL}$1"
}

talkies_method() {
    local method="$1" path="$2"
    curl -sf --max-time 30 -X "$method" "${TALKIES_BASE_URL}${path}"
}

talkies_method_status() {
    local method="$1" path="$2"
    curl -s -o /dev/null -w "%{http_code}" --max-time 30 -X "$method" "${TALKIES_BASE_URL}${path}"
}

# Multipart upload to /v1/audio/transcriptions. talkies cap on upload size is
# 100 MB by default — fixtures must be smaller.
#
# args:
#   $1 = model slug
#   $2 = path to local audio fixture
#   $3 = response_format (defaults to json)
#   $4..$N = extra "key=value" form fields (e.g. "timestamp_granularities[]=word")
#
# Successful HTTP 2xx → body on stdout, exit 0.
# Anything else      → stderr explains, exit 1.
talkies_transcribe() {
    local model="$1" fixture="$2" response_format="${3:-json}"
    shift 3

    local extras=()
    local kv
    for kv in "$@"; do
        extras+=(-F "$kv")
    done

    local tmp
    tmp=$(mktemp -t talkies_resp.XXXXXX) || return 2
    local code
    code=$(curl -s -o "$tmp" -w "%{http_code}" --max-time 900 \
        -F "model=$model" \
        -F "response_format=$response_format" \
        "${extras[@]}" \
        -F "file=@${fixture}" \
        "${TALKIES_BASE_URL}/v1/audio/transcriptions" 2>/dev/null) || {
        rm -f "$tmp"
        return 2
    }
    if [ "$code" -lt 200 ] || [ "$code" -ge 300 ]; then
        echo "  HTTP $code: $(head -c 500 "$tmp")" >&2
        rm -f "$tmp"
        return 1
    fi
    cat "$tmp"
    rm -f "$tmp"
}

# Find an audio fixture under tests/integration/.fixtures (any common ext).
# Returns the path on stdout, empty string if none.
talkies_find_fixture() {
    local dir="${BASH_SOURCE%/*}/.fixtures"
    local ext fixture=""
    for ext in wav mp3 m4a flac ogg; do
        if [ -f "${dir}/audio.${ext}" ]; then
            fixture="${dir}/audio.${ext}"
            break
        fi
    done
    echo "$fixture"
}

# Wait until the talkies /healthz endpoint comes back ok.
# Long timeout because the first boot has to download N models.
talkies_wait_ready() {
    local max="${1:-1800}"  # 30 min cap on first boot
    local i=0
    while [ "$i" -lt "$max" ]; do
        if curl -sf --max-time 5 "${TALKIES_BASE_URL}/healthz" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        i=$((i + 2))
    done
    echo "  TIMEOUT: ${TALKIES_BASE_URL}/healthz never became ready in ${max}s" >&2
    return 1
}

# Models we expect /v1/models to expose, derived from TALKIES_ENABLED_MODELS
# (or all CUDA models when that's empty — but the runner pins it).
talkies_expected_models() {
    if [ -n "${TALKIES_ENABLED_MODELS:-}" ]; then
        echo "$TALKIES_ENABLED_MODELS" | tr ',' ' '
        return
    fi
    # Default CUDA full set, matches models.json.
    echo "whisper-large-v3 whisper-large-v3-turbo distil-whisper-large-v3 parakeet-tdt-0.6b-v3 canary-180m-flash canary-1b-flash canary-qwen-2.5b"
}
