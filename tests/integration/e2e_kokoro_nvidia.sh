#!/bin/bash
# Focused end-to-end coverage for the kokoro-82m-nvidia (ONNX/ORT) slug —
# tighter battery than the full test_speech.sh, useful for iterating on the
# new backend in isolation. Self-contained: spawns its own --rm --gpus all
# container via the harness, tears it down on exit. Invoke directly:
#
#     bash tests/integration/e2e_kokoro_nvidia.sh

set -eo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=harness.sh
source "${_DIR}/harness.sh"
# shellcheck source=common.sh
source "${_DIR}/common.sh"

KN_MODELS="whisper-large-v3-turbo,kokoro-82m,kokoro-82m-nvidia"
harness_start "$KN_MODELS"

TTS_SLUG="kokoro-82m-nvidia"
TTS_SLUG_PYTORCH="kokoro-82m"
ASR_SLUG="whisper-large-v3-turbo"

TEST_PHRASE="The quick brown fox jumps over the lazy dog."
EXPECTED_WORDS=(quick brown fox jumps lazy dog)

# Shared scratch dir, removed on exit via a trap that also chains harness
# cleanup so the container still tears down even if the work-dir rm fails.
KN_WORK="$(mktemp -d -t kn_e2e.XXXXXX)"
trap 'rm -rf "$KN_WORK"; _harness_cleanup' EXIT

# ── 1. /v1/models lists the slug ─────────────────────────────────────────────

test_models_lists_slug() {
    local models_json
    models_json=$(talkies_get "/v1/models") || { echo "  FAIL: /v1/models unreachable"; return 1; }
    echo "$models_json" | jq -e --arg m "$TTS_SLUG" '.data[] | select(.id==$m)' >/dev/null \
        || { echo "  FAIL: /v1/models missing $TTS_SLUG"; return 1; }
    echo "  ok: $TTS_SLUG present in /v1/models"
    echo "OK: $FUNCNAME"
}

# ── 2. /v1/audio/voices: non-empty catalog + at least one default ────────────

test_voices_catalog() {
    local voices_json voice_count default_count
    voices_json=$(talkies_get "/v1/audio/voices") || { echo "  FAIL: /v1/audio/voices unreachable"; return 1; }
    voice_count=$(echo "$voices_json" \
        | jq --arg m "$TTS_SLUG" '[.voices[] | select(.model==$m)] | length')
    [ "$voice_count" -ge 1 ] || { echo "  FAIL: $TTS_SLUG reports 0 voices"; return 1; }
    default_count=$(echo "$voices_json" \
        | jq --arg m "$TTS_SLUG" '[.voices[] | select(.model==$m and .default==true)] | length')
    [ "$default_count" -ge 1 ] || { echo "  FAIL: $TTS_SLUG has no default voice"; return 1; }
    echo "  ok: $TTS_SLUG voices=$voice_count default=$default_count"
    echo "OK: $FUNCNAME"
}

# ── 3. Default-voice synth → valid RIFF WAV ──────────────────────────────────

test_default_voice_wav() {
    local wav="$KN_WORK/default.wav" code head4 size
    code=$(curl -s -o "$wav" -w "%{http_code}" --max-time 300 \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg m "$TTS_SLUG" '{model:$m, input:"Default voice test.", response_format:"wav"}')" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    [ "$code" = "200" ] || { echo "  FAIL: HTTP $code (body: $(head -c 200 "$wav"))"; return 1; }
    head4=$(head -c 4 "$wav" | od -An -c | tr -d ' \n')
    [ "$head4" = "RIFF" ] || { echo "  FAIL: missing RIFF (got '$head4')"; return 1; }
    size=$(stat -c %s "$wav" 2>/dev/null || stat -f %z "$wav")
    [ "$size" -ge 4096 ] || { echo "  FAIL: wav too small ($size bytes)"; return 1; }
    echo "  ok: $size bytes, RIFF"
    echo "OK: $FUNCNAME"
}

# ── 4. Synth → ASR round-trip: every expected word in transcript ─────────────

test_round_trip_through_asr() {
    local rt_wav="$KN_WORK/roundtrip.wav" rt_size asr_resp="$KN_WORK/asr.json"
    curl -sf --max-time 30 -X POST "${TALKIES_BASE_URL}/unload" >/dev/null || true
    if ! talkies_speech "$TTS_SLUG" "af_heart" "$TEST_PHRASE" "wav" "$rt_wav"; then
        echo "  FAIL: synth"; return 1
    fi
    rt_size=$(stat -c %s "$rt_wav" 2>/dev/null || stat -f %z "$rt_wav")
    [ "$rt_size" -ge 4096 ] || { echo "  FAIL: wav too small ($rt_size)"; return 1; }

    local asr_code
    asr_code=$(curl -s -o "$asr_resp" -w "%{http_code}" --max-time 300 \
        -F "model=${ASR_SLUG}" \
        -F "response_format=json" \
        -F "file=@${rt_wav}" \
        "${TALKIES_BASE_URL}/v1/audio/transcriptions")
    [ "$asr_code" = "200" ] || { echo "  FAIL: ASR HTTP $asr_code (body: $(head -c 200 "$asr_resp"))"; return 1; }

    local transcript normalized
    transcript=$(jq -r '.text' "$asr_resp")
    [ -n "$transcript" ] && [ "$transcript" != "null" ] \
        || { echo "  FAIL: empty transcript"; return 1; }
    normalized=$(echo "$transcript" | talkies_normalize_text)
    echo "  transcript: \"$normalized\""

    local missing=() w
    for w in "${EXPECTED_WORDS[@]}"; do
        [[ " $normalized " == *" $w "* ]] || missing+=("$w")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "  spoken: \"$TEST_PHRASE\""
        echo "  raw:    \"$transcript\""
        echo "  FAIL: transcript missing: ${missing[*]}"
        return 1
    fi
    echo "  ok: all expected words present"
    echo "OK: $FUNCNAME"
}

# ── 5. Sibling eviction — kokoro-82m → kokoro-82m-nvidia back-to-back ────────

test_sibling_eviction() {
    local a="$KN_WORK/pytorch.wav" b="$KN_WORK/onnx.wav" a_size b_size
    curl -sf --max-time 30 -X POST "${TALKIES_BASE_URL}/unload" >/dev/null || true
    talkies_speech "$TTS_SLUG_PYTORCH" "af_heart" "Hello from pytorch." "wav" "$a" \
        || { echo "  FAIL: $TTS_SLUG_PYTORCH synth"; return 1; }
    talkies_speech "$TTS_SLUG"         "af_heart" "Hello from onnx."    "wav" "$b" \
        || { echo "  FAIL: $TTS_SLUG synth after eviction"; return 1; }
    a_size=$(stat -c %s "$a" 2>/dev/null || stat -f %z "$a")
    b_size=$(stat -c %s "$b" 2>/dev/null || stat -f %z "$b")
    [ "$a_size" -ge 4096 ] && [ "$b_size" -ge 4096 ] \
        || { echo "  FAIL: back-to-back wavs too small (a=$a_size b=$b_size)"; return 1; }
    echo "  ok: pytorch→onnx round-trip survived (a=${a_size}B b=${b_size}B)"
    echo "OK: $FUNCNAME"
}

# ── 6. Unknown voice → 400 ───────────────────────────────────────────────────

test_unknown_voice_400() {
    local body code
    body=$(jq -n --arg m "$TTS_SLUG" \
        '{model:$m, voice:"this_voice_does_not_exist", input:"hi", response_format:"wav"}')
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    assert_eq "$code" "400" "unknown voice → 400" || return 1
    echo "OK: $FUNCNAME"
}

# ── 7. Empty input → 400 ─────────────────────────────────────────────────────

test_empty_input_400() {
    local body code
    body=$(jq -n --arg m "$TTS_SLUG" '{model:$m, input:"   ", response_format:"wav"}')
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    assert_eq "$code" "400" "empty input → 400" || return 1
    echo "OK: $FUNCNAME"
}

harness_run_tests \
    test_models_lists_slug \
    test_voices_catalog \
    test_default_voice_wav \
    test_round_trip_through_asr \
    test_sibling_eviction \
    test_unknown_voice_400 \
    test_empty_input_400
