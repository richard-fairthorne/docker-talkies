#!/bin/bash
# Qwen3-TTS integration tests — voice cloning via /v1/audio/speech, CUDA-only.
# Self-contained: spawns its own --rm --gpus all container via the harness,
# tears it down on exit. Invoke directly: bash tests/integration/test_qwen3.sh
#
# Exercises:
#   - Builtin voices show up in /v1/audio/voices with origin=builtin.
#   - Custom voices dropped into the data volume's custom-voices/ surface as
#     nested-path voice names with origin=custom and shadow builtins on
#     name collision.
#   - Synthesis with a baked-in voice produces well-formed PCM/WAV.
#   - Cross-modality round-trip Qwen3 → ASR → expected words.
#   - 400 on unknown voice and on empty input.

set -eo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=harness.sh
source "${_DIR}/harness.sh"
# shellcheck source=common.sh
source "${_DIR}/common.sh"

QWEN3_MODELS="whisper-large-v3-turbo,kokoro-82m,qwen3-tts-0.6b"
harness_start "$QWEN3_MODELS"

QWEN3_MODEL="qwen3-tts-0.6b"
QWEN3_TEST_PHRASE="The quick brown fox jumps over the lazy dog."
QWEN3_EXPECTED_WORDS=(quick brown fox jumps lazy dog)

# ── Builtin voices show up tagged origin=builtin ─────────────────────────────

test_qwen3_voices_builtin_listed() {
    local out builtin_count
    out=$(talkies_get "/v1/audio/voices") || { echo "  FAIL: /v1/audio/voices unreachable"; return 1; }
    builtin_count=$(echo "$out" | jq --arg m "$QWEN3_MODEL" \
        '[.voices[] | select(.model==$m) | select(.origin=="builtin")] | length' 2>/dev/null || echo 0)
    if [ "$builtin_count" -lt 1 ]; then
        echo "  FAIL: expected at least 1 builtin voice for $QWEN3_MODEL, got $builtin_count"
        echo "  raw: $(echo "$out" | jq -c --arg m "$QWEN3_MODEL" '[.voices[] | select(.model==$m)]')"
        return 1
    fi
    echo "  ok: $builtin_count builtin voice(s)"
    echo "OK: $FUNCNAME"
}

# ── Custom voices dropped at runtime show up as origin=custom with nested name

test_qwen3_voices_custom_discovery() {
    local src_wav="${_HARNESS_REPO_ROOT}/voices/qwen3/alloy.wav"
    if [ ! -f "$src_wav" ]; then
        echo "  SKIP: $src_wav missing — repo not laid out as expected"
        return 0
    fi
    local custom_dir="$HARNESS_CACHE_DIR/custom-voices/foo/bar"
    mkdir -p "$custom_dir"
    cp "$src_wav" "$custom_dir/test_clone.wav"
    # shellcheck disable=SC2064
    trap "rm -f '$custom_dir/test_clone.wav'" RETURN

    local out custom_entry
    out=$(talkies_get "/v1/audio/voices") || { echo "  FAIL: /v1/audio/voices unreachable"; return 1; }
    custom_entry=$(echo "$out" | jq -c --arg m "$QWEN3_MODEL" \
        '.voices[] | select(.model==$m) | select(.voice=="foo/bar/test_clone")' 2>/dev/null)
    if [ -z "$custom_entry" ]; then
        echo "  FAIL: nested custom voice foo/bar/test_clone not in /v1/audio/voices"
        echo "  qwen3 voices: $(echo "$out" | jq -c --arg m "$QWEN3_MODEL" '[.voices[] | select(.model==$m) | .voice]')"
        return 1
    fi
    local origin
    origin=$(echo "$custom_entry" | jq -r '.origin')
    if [ "$origin" != "custom" ]; then
        echo "  FAIL: expected origin=custom for foo/bar/test_clone, got '$origin'"
        return 1
    fi
    echo "  ok: foo/bar/test_clone origin=custom"
    echo "OK: $FUNCNAME"
}

# ── Custom voice with builtin name shadows the builtin ──────────────────────

test_qwen3_voices_custom_shadows_builtin() {
    local src_wav="${_HARNESS_REPO_ROOT}/voices/qwen3/alloy.wav"
    if [ ! -f "$src_wav" ]; then
        echo "  SKIP: $src_wav missing"
        return 0
    fi
    local custom_dir="$HARNESS_CACHE_DIR/custom-voices"
    mkdir -p "$custom_dir"
    cp "$src_wav" "$custom_dir/alloy.wav"
    # shellcheck disable=SC2064
    trap "rm -f '$custom_dir/alloy.wav'" RETURN

    local out alloy_origin
    out=$(talkies_get "/v1/audio/voices") || { echo "  FAIL: /v1/audio/voices unreachable"; return 1; }
    alloy_origin=$(echo "$out" | jq -r --arg m "$QWEN3_MODEL" \
        '.voices[] | select(.model==$m) | select(.voice=="alloy") | .origin' 2>/dev/null)
    if [ "$alloy_origin" != "custom" ]; then
        echo "  FAIL: expected alloy.origin=custom after shadowing, got '$alloy_origin'"
        return 1
    fi
    echo "  ok: builtin alloy shadowed by custom override"
    echo "OK: $FUNCNAME"
}

# ── Synthesize with builtin voice → well-formed wav ─────────────────────────

test_qwen3_speech_builtin_voice() {
    local outfile size head4
    outfile=$(mktemp -t qwen3_speech.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -f '$outfile'" RETURN
    if ! talkies_speech "$QWEN3_MODEL" "alloy" "Hello world." "wav" "$outfile"; then
        echo "  FAIL: qwen3 alloy synthesis"
        return 1
    fi
    size=$(stat -c %s "$outfile" 2>/dev/null || stat -f %z "$outfile" 2>/dev/null || echo 0)
    if [ "$size" -lt 4096 ]; then
        echo "  FAIL: wav suspiciously small ($size bytes)"
        return 1
    fi
    head4=$(head -c 4 "$outfile" | od -An -c | tr -d ' \n')
    if [ "$head4" != "RIFF" ]; then
        echo "  FAIL: wav missing RIFF header (got '$head4')"
        return 1
    fi
    echo "  ok: alloy wav size=${size}B"
    echo "OK: $FUNCNAME"
}

# ── Cross-modality round-trip: Qwen3 → ASR → expected words present ────────

test_qwen3_speech_round_trip_through_asr() {
    local asr_model
    asr_model=$(talkies_pick_fast_asr_model) || {
        echo "  SKIP: no fast ASR model available on server"
        return 0
    }
    echo "  using asr_model=$asr_model"

    local tmp wavfile
    tmp=$(mktemp -d -t qwen3_roundtrip.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN
    wavfile="${tmp}/spoken.wav"

    # Fresh slate so the test exercises cold-load + sibling eviction (qwen3
    # captures CUDA graphs on first call — the slow path).
    talkies_method POST "/unload" >/dev/null 2>&1 || true

    if ! talkies_speech "$QWEN3_MODEL" "alloy" "$QWEN3_TEST_PHRASE" "wav" "$wavfile"; then
        echo "  FAIL: qwen3 synthesis"
        return 1
    fi
    local size
    size=$(stat -c %s "$wavfile" 2>/dev/null || stat -f %z "$wavfile" 2>/dev/null || echo 0)
    if [ "$size" -lt 4096 ]; then
        echo "  FAIL: synthesized wav too small ($size bytes)"
        return 1
    fi
    echo "  qwen3 produced wav ($size bytes)"

    local out text normalized
    out=$(talkies_transcribe "$asr_model" "$wavfile" "json") || {
        echo "  FAIL: ASR round-trip via $asr_model"
        return 1
    }
    text=$(echo "$out" | jq -r '.text' 2>/dev/null || echo "")
    if [ -z "$text" ] || [ "$text" = "null" ]; then
        echo "  FAIL: ASR returned empty text"
        return 1
    fi
    normalized=$(echo "$text" | talkies_normalize_text)
    echo "  transcribed: \"$normalized\""

    local missing=() word
    for word in "${QWEN3_EXPECTED_WORDS[@]}"; do
        if [[ " $normalized " != *" $word "* ]]; then
            missing+=("$word")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "  FAIL: round-trip transcript missing words: ${missing[*]}"
        echo "  spoken phrase: \"$QWEN3_TEST_PHRASE\""
        echo "  raw asr text:  \"$text\""
        return 1
    fi
    echo "  ok: all expected words present (${QWEN3_EXPECTED_WORDS[*]})"
    echo "OK: $FUNCNAME"
}

# ── Error path: unknown voice → 400 ──────────────────────────────────────────

test_qwen3_speech_unknown_voice_400() {
    local body code
    body=$(jq -n --arg m "$QWEN3_MODEL" \
        '{model:$m, voice:"does/not/exist", input:"hi", response_format:"wav"}')
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    assert_eq "$code" "400" "qwen3 unknown voice → 400" || return 1
    echo "OK: $FUNCNAME"
}

# ── Error path: empty input → 400 ────────────────────────────────────────────

test_qwen3_speech_empty_input_400() {
    local body code
    body=$(jq -n --arg m "$QWEN3_MODEL" \
        '{model:$m, voice:"alloy", input:"   ", response_format:"wav"}')
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    assert_eq "$code" "400" "qwen3 empty input → 400" || return 1
    echo "OK: $FUNCNAME"
}

# ── `instructions` field is threaded through to faster-qwen3-tts ─────────────
# PR #1 (martincohen) wired the OpenAI `instructions` field as the `instruct`
# parameter on `generate_voice_clone`. We can't assert the audio actually
# changed in any specific way without subjective listening, but we can
# verify (a) the request returns 200 (plumbing works end-to-end, no
# TypeError from a missing param) and (b) the audio is well-formed.
# Pairing with a no-instructions call also catches regressions where the
# param gets swallowed by Pydantic but never reaches the backend.

test_qwen3_speech_instructions_field_accepted() {
    local outfile size head4 body code
    outfile=$(mktemp -t qwen3_instr.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -f '$outfile'" RETURN

    body=$(jq -n --arg m "$QWEN3_MODEL" \
        '{model:$m, voice:"alloy", input:"Hello with instructions.", instructions:"Speak in a calm, soft tone.", response_format:"wav"}')
    code=$(curl -s -o "$outfile" -w "%{http_code}" --max-time 300 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    if [ "$code" != "200" ]; then
        echo "  FAIL: qwen3 with instructions HTTP $code (body head: $(head -c 200 "$outfile"))"
        return 1
    fi
    size=$(stat -c %s "$outfile" 2>/dev/null || stat -f %z "$outfile" 2>/dev/null || echo 0)
    if [ "$size" -lt 4096 ]; then
        echo "  FAIL: wav too small ($size bytes — instructions may have crashed the backend)"
        return 1
    fi
    head4=$(head -c 4 "$outfile" | od -An -c | tr -d ' \n')
    if [ "$head4" != "RIFF" ]; then
        echo "  FAIL: wav missing RIFF (got '$head4')"
        return 1
    fi
    echo "  ok: instructions threaded, wav size=${size}B"
    echo "OK: $FUNCNAME"
}

# ── x-vector fallback when voice has no sibling .txt ─────────────────────────
# PR #1 also fixed the case where a custom voice's `.wav` had no sibling
# `.txt` transcript — faster-qwen3-tts raises 400 in ICL mode without
# ref_text. The backend now auto-detects and flips x_vector_only_mode=True
# with a warning log. Without this fallback the request would 400 like
# pre-PR; this test guards the new code path.

test_qwen3_speech_no_ref_txt_falls_back_to_xvector() {
    local src_wav="${_HARNESS_REPO_ROOT}/voices/qwen3/alloy.wav"
    if [ ! -f "$src_wav" ]; then
        echo "  SKIP: $src_wav missing — repo not laid out as expected"
        return 0
    fi
    local custom_dir="$HARNESS_CACHE_DIR/custom-voices"
    local voice_name="no_txt_clone_$$"
    mkdir -p "$custom_dir"
    cp "$src_wav" "$custom_dir/${voice_name}.wav"
    # Important: do NOT create $voice_name.txt — that's the path under test.
    # shellcheck disable=SC2064
    trap "rm -f '$custom_dir/${voice_name}.wav'" RETURN

    # Custom voices are scanned live (no restart needed). Probe /v1/audio/voices
    # to confirm the server sees it before issuing the synth call.
    local voices_out
    voices_out=$(talkies_get "/v1/audio/voices") || { echo "  FAIL: /v1/audio/voices unreachable"; return 1; }
    if ! echo "$voices_out" | jq -e --arg m "$QWEN3_MODEL" --arg v "$voice_name" \
        '.voices[] | select(.model==$m and .voice==$v)' >/dev/null 2>&1; then
        echo "  FAIL: $voice_name not picked up by live voice scan"
        return 1
    fi

    local outfile size head4 body code
    outfile=$(mktemp -t qwen3_xvec.XXXXXX) || return 2
    body=$(jq -n --arg m "$QWEN3_MODEL" --arg v "$voice_name" \
        '{model:$m, voice:$v, input:"Hello via x-vector.", response_format:"wav"}')
    code=$(curl -s -o "$outfile" -w "%{http_code}" --max-time 300 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    if [ "$code" != "200" ]; then
        # Pre-PR behaviour was 400 here — this is the regression we're guarding.
        echo "  FAIL: no-ref-txt synth HTTP $code (expected 200 via x-vector fallback)"
        echo "  body head: $(head -c 300 "$outfile")"
        rm -f "$outfile"
        return 1
    fi
    size=$(stat -c %s "$outfile" 2>/dev/null || stat -f %z "$outfile" 2>/dev/null || echo 0)
    head4=$(head -c 4 "$outfile" | od -An -c | tr -d ' \n')
    rm -f "$outfile"
    if [ "$size" -lt 4096 ] || [ "$head4" != "RIFF" ]; then
        echo "  FAIL: x-vector synth wav malformed (size=$size head='$head4')"
        return 1
    fi
    echo "  ok: x-vector fallback synth size=${size}B"
    echo "OK: $FUNCNAME"
}

# ── Kokoro accepts (and ignores) `instructions` for OpenAI parity ────────────
# Sanity check that the PR #1 protocol change to TTSBackend.synthesize didn't
# break Kokoro: passing `instructions` against a kokoro slug must still
# return 200 — the backend has no instruction-prompt input, so the field is
# silently dropped, but the param signature must accept it.

test_kokoro_speech_instructions_field_accepted() {
    if ! echo "$HARNESS_ENABLED_MODELS" | grep -q "kokoro-82m"; then
        echo "  SKIP: no kokoro slug enabled in this suite"
        return 0
    fi
    local outfile size head4 body code
    outfile=$(mktemp -t kokoro_instr.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -f '$outfile'" RETURN

    body=$(jq -n \
        '{model:"kokoro-82m", voice:"af_heart", input:"Hello kokoro.", instructions:"Speak softly.", response_format:"wav"}')
    code=$(curl -s -o "$outfile" -w "%{http_code}" --max-time 300 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    if [ "$code" != "200" ]; then
        echo "  FAIL: kokoro with instructions HTTP $code (body: $(head -c 200 "$outfile"))"
        return 1
    fi
    size=$(stat -c %s "$outfile" 2>/dev/null || stat -f %z "$outfile" 2>/dev/null || echo 0)
    head4=$(head -c 4 "$outfile" | od -An -c | tr -d ' \n')
    if [ "$size" -lt 1024 ] || [ "$head4" != "RIFF" ]; then
        echo "  FAIL: kokoro wav malformed (size=$size head='$head4')"
        return 1
    fi
    echo "  ok: kokoro accepted instructions silently, wav size=${size}B"
    echo "OK: $FUNCNAME"
}

harness_run_tests \
    test_qwen3_voices_builtin_listed \
    test_qwen3_voices_custom_discovery \
    test_qwen3_voices_custom_shadows_builtin \
    test_qwen3_speech_builtin_voice \
    test_qwen3_speech_round_trip_through_asr \
    test_qwen3_speech_unknown_voice_400 \
    test_qwen3_speech_empty_input_400 \
    test_qwen3_speech_instructions_field_accepted \
    test_qwen3_speech_no_ref_txt_falls_back_to_xvector \
    test_kokoro_speech_instructions_field_accepted
