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

# POST /v1/audio/speech with a JSON body. Writes the raw audio bytes to the
# given output file. Useful for both "did we get non-zero bytes" smoke and the
# cross-modality test that round-trips Kokoro output through an ASR model.
#
# args:
#   $1 = model slug (TTS)
#   $2 = voice (pass "" to fall back to the model's default_voice)
#   $3 = text to synthesize
#   $4 = response_format (mp3 / opus / aac / flac / wav / pcm; default mp3)
#   $5 = output file path
#
# Returns 0 on HTTP 2xx, 1 on non-2xx (body dumped to stderr), 2 on transport.
talkies_speech() {
    local model="$1" voice="$2" text="$3" fmt="${4:-mp3}" outfile="$5"
    local body
    if [ -n "$voice" ]; then
        body=$(jq -n --arg m "$model" --arg v "$voice" --arg t "$text" --arg f "$fmt" \
            '{model:$m, voice:$v, input:$t, response_format:$f}') || return 2
    else
        # Omit voice → server uses model.default_voice.
        body=$(jq -n --arg m "$model" --arg t "$text" --arg f "$fmt" \
            '{model:$m, input:$t, response_format:$f}') || return 2
    fi
    local code
    code=$(curl -s -o "$outfile" -w "%{http_code}" --max-time 300 \
        -H "Content-Type: application/json" \
        -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech" 2>/dev/null) || return 2
    if [ "$code" -lt 200 ] || [ "$code" -ge 300 ]; then
        echo "  HTTP $code: $(head -c 500 "$outfile")" >&2
        return 1
    fi
    return 0
}

# Return the first slug from the preference list that's actually configured on
# the running server (read from /v1/models). Used by the cross-modality test
# to pick a fast ASR backend for round-tripping Kokoro output regardless of
# what TALKIES_ENABLED_MODELS happens to be.
talkies_pick_fast_asr_model() {
    local models_json prefer slug
    models_json=$(talkies_get "/v1/models") || return 1
    for prefer in parakeet-tdt-0.6b-v3 whisper-large-v3-turbo distil-whisper-large-v3 canary-180m-flash whisper-large-v3; do
        if echo "$models_json" | jq -e --arg p "$prefer" '.data[] | select(.id==$p)' >/dev/null 2>&1; then
            echo "$prefer"
            return 0
        fi
    done
    return 1
}

# Normalize transcribed text for fuzzy comparison: lowercase, strip
# punctuation, collapse whitespace. ASR output has leading spaces, varying
# capitalisation, sometimes inserted commas/periods — none of which we care
# about for word-presence assertions.
talkies_normalize_text() {
    tr '[:upper:]' '[:lower:]' \
        | tr -d '.,!?;:"()[]{}' \
        | tr -s '[:space:]' ' ' \
        | sed -e 's/^ //' -e 's/ $//'
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
