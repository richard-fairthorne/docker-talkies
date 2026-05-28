#!/bin/bash
# shellcheck shell=bash disable=SC2154  # ALL_TESTS comes from common.sh

# TTS integration tests — Kokoro via /v1/audio/speech. CUDA-only suite, but
# Kokoro itself ships in both images; we run here because the harness is
# already CUDA and the cross-modality round-trip wants a fast ASR backend.
#
# Skips cleanly if kokoro-82m is filtered out by TALKIES_ENABLED_MODELS so the
# rest of the suite still runs.

TTS_MODEL="kokoro-82m"

# Phrase chosen for the round-trip test: short, common English words, no
# rare proper nouns. Stays well within Kokoro's typical synthesis quality
# envelope so the ASR-side assertions aren't flaky.
TTS_TEST_PHRASE="The quick brown fox jumps over the lazy dog."

# Words we require to be present in the round-tripped transcript (lowercased,
# punctuation-stripped). Picked to be unambiguous tokens unlikely to be
# misheard ("the" / "over" excluded — too generic, common ASR filler).
TTS_EXPECTED_WORDS=(quick brown fox jumps lazy dog)

# Pre-flight: is the TTS model actually loadable on this server? If the
# operator restricted TALKIES_ENABLED_MODELS to ASR-only slugs, skip cleanly.
_tts_model_available() {
    local models_json
    models_json=$(talkies_get "/v1/models") || return 1
    echo "$models_json" | jq -e --arg m "$TTS_MODEL" '.data[] | select(.id==$m)' >/dev/null 2>&1
}

# ── GET /v1/audio/voices returns a non-empty list ────────────────────────────

test_talkies_voices_list() {
    if ! _tts_model_available; then
        echo "  SKIP: $TTS_MODEL not in /v1/models (TALKIES_ENABLED_MODELS excludes it)"
        return 0
    fi
    local out count default_count
    out=$(talkies_get "/v1/audio/voices") || { echo "  FAIL: /v1/audio/voices unreachable"; return 1; }
    count=$(echo "$out" | jq '.voices | length' 2>/dev/null || echo 0)
    if [ "$count" -lt 1 ]; then
        echo "  FAIL: /v1/audio/voices empty (count=$count)"
        return 1
    fi
    default_count=$(echo "$out" | jq '[.voices[] | select(.default==true)] | length' 2>/dev/null || echo 0)
    if [ "$default_count" -lt 1 ]; then
        echo "  FAIL: no voice marked default=true"
        return 1
    fi
    echo "  ok: voices=$count default=$default_count"
    echo "OK: talkies_voices_list"
}

# ── POST /v1/audio/speech with each format → non-zero bytes + correct magic ──

test_talkies_speech_all_formats() {
    if ! _tts_model_available; then
        echo "  SKIP: $TTS_MODEL not in /v1/models"
        return 0
    fi
    local rc=0 fmt outfile size head4 head2
    local tmp
    tmp=$(mktemp -d -t talkies_speech.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN

    for fmt in mp3 opus aac flac wav pcm; do
        outfile="${tmp}/out.${fmt}"
        if ! talkies_speech "$TTS_MODEL" "" "Hello world." "$fmt" "$outfile"; then
            echo "  FAIL: $fmt synthesis"
            rc=1
            continue
        fi
        size=$(stat -c %s "$outfile" 2>/dev/null || stat -f %z "$outfile" 2>/dev/null || echo 0)
        if [ "$size" -lt 1024 ]; then
            echo "  FAIL: $fmt suspiciously small ($size bytes — likely an error page or empty body)"
            rc=1
            continue
        fi
        head4=$(head -c 4 "$outfile" | od -An -c | tr -d ' \n')
        head2=$(head -c 2 "$outfile" | od -An -tx1 | tr -d ' \n')
        case "$fmt" in
            wav)
                if [ "$head4" != "RIFF" ]; then
                    echo "  FAIL: wav missing RIFF header (got '$head4')"; rc=1; continue
                fi
                ;;
            flac)
                if [ "$head4" != "fLaC" ]; then
                    echo "  FAIL: flac missing fLaC magic (got '$head4')"; rc=1; continue
                fi
                ;;
            opus)
                if [ "$head4" != "OggS" ]; then
                    echo "  FAIL: opus missing OggS container magic (got '$head4')"; rc=1; continue
                fi
                ;;
            aac)
                # ADTS sync word: first 12 bits = 0xFFF. Byte 0 = 0xFF, byte 1
                # high nibble = 0xF.
                if [ "${head2:0:2}" != "ff" ] || [ "${head2:2:1}" != "f" ]; then
                    echo "  FAIL: aac missing ADTS sync word (got '$head2')"; rc=1; continue
                fi
                ;;
            mp3)
                # ID3v2 prefix ("ID3") OR MPEG frame sync (0xFFFB / 0xFFFA /
                # 0xFFF3 / 0xFFF2 — first 11 bits set, layer III).
                if [ "${head4:0:3}" != "ID3" ] && [ "${head2:0:2}" != "ff" ]; then
                    echo "  FAIL: mp3 missing ID3 / frame sync (got '$head4'/'$head2')"; rc=1; continue
                fi
                ;;
            pcm)
                # Raw s16le, no container. Size check above is the only signal.
                :
                ;;
        esac
        echo "  ok: $fmt size=${size}B head4='$head4'"
    done

    if [ "$rc" -eq 0 ]; then
        echo "OK: talkies_speech_all_formats"
    fi
    return $rc
}

# ── Default voice fallback: voice omitted → server uses model default ───────

test_talkies_speech_default_voice() {
    if ! _tts_model_available; then
        echo "  SKIP: $TTS_MODEL not in /v1/models"
        return 0
    fi
    local outfile size
    outfile=$(mktemp -t talkies_default.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -f '$outfile'" RETURN

    if ! talkies_speech "$TTS_MODEL" "" "Default voice test." "wav" "$outfile"; then
        echo "  FAIL: synthesis with omitted voice"
        return 1
    fi
    size=$(stat -c %s "$outfile" 2>/dev/null || stat -f %z "$outfile" 2>/dev/null || echo 0)
    if [ "$size" -lt 1024 ]; then
        echo "  FAIL: default-voice output suspiciously small ($size bytes)"
        return 1
    fi
    echo "  ok: default-voice wav size=${size}B"
    echo "OK: talkies_speech_default_voice"
}

# ── Error path: unknown voice → 400 ──────────────────────────────────────────

test_talkies_speech_unknown_voice_400() {
    if ! _tts_model_available; then
        echo "  SKIP: $TTS_MODEL not in /v1/models"
        return 0
    fi
    local body code
    body=$(jq -n --arg m "$TTS_MODEL" \
        '{model:$m, voice:"this_voice_does_not_exist", input:"hi", response_format:"wav"}')
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    assert_eq "$code" "400" "unknown voice → 400" || return 1
    echo "OK: talkies_speech_unknown_voice_400"
}

# ── Error path: empty input → 400 ────────────────────────────────────────────

test_talkies_speech_empty_input_400() {
    if ! _tts_model_available; then
        echo "  SKIP: $TTS_MODEL not in /v1/models"
        return 0
    fi
    local body code
    body=$(jq -n --arg m "$TTS_MODEL" \
        '{model:$m, input:"   ", response_format:"wav"}')
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    assert_eq "$code" "400" "empty input → 400" || return 1
    echo "OK: talkies_speech_empty_input_400"
}

# ── Error path: ASR model posted to /v1/audio/speech → 400 ───────────────────

test_talkies_speech_rejects_asr_model() {
    if ! _tts_model_available; then
        echo "  SKIP: $TTS_MODEL not in /v1/models"
        return 0
    fi
    local asr_model body code
    asr_model=$(talkies_pick_fast_asr_model) || {
        echo "  SKIP: no ASR model configured on server"
        return 0
    }
    body=$(jq -n --arg m "$asr_model" \
        '{model:$m, input:"hi", response_format:"wav"}')
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    assert_eq "$code" "400" "ASR slug on /v1/audio/speech → 400" || return 1
    echo "OK: talkies_speech_rejects_asr_model"
}

# ── Error path: TTS model posted to /v1/audio/transcriptions → 400 ──────────

test_talkies_transcribe_rejects_tts_model() {
    if ! _tts_model_available; then
        echo "  SKIP: $TTS_MODEL not in /v1/models"
        return 0
    fi
    local fixture
    fixture=$(talkies_find_fixture)
    if [ -z "$fixture" ]; then
        echo "  SKIP: tests/integration/.fixtures/audio.* missing"
        return 0
    fi
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -F "model=$TTS_MODEL" \
        -F "file=@${fixture}" \
        "${TALKIES_BASE_URL}/v1/audio/transcriptions")
    assert_eq "$code" "400" "TTS slug on /v1/audio/transcriptions → 400" || return 1
    echo "OK: talkies_transcribe_rejects_tts_model"
}

# ── Cross-modality round-trip: Kokoro → ASR → expected words present ────────
#
# Generates a WAV with a known phrase, feeds it back into a fast ASR backend,
# checks the transcript contains the high-signal words. This catches:
#   - Kokoro returning correct sample rate / channel layout (otherwise ASR
#     output is gibberish even if the bytes are well-formed).
#   - The ffmpeg encode path not corrupting the audio.
#   - Sibling-eviction working both directions (Kokoro loaded → unloaded →
#     ASR loaded — without OOM or stuck-resident state).

test_talkies_speech_round_trip_through_asr() {
    if ! _tts_model_available; then
        echo "  SKIP: $TTS_MODEL not in /v1/models"
        return 0
    fi
    local asr_model
    asr_model=$(talkies_pick_fast_asr_model) || {
        echo "  SKIP: no fast ASR model available on server for round-trip"
        return 0
    }
    echo "  using asr_model=$asr_model"

    local tmp wavfile
    tmp=$(mktemp -d -t talkies_roundtrip.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN
    wavfile="${tmp}/spoken.wav"

    # Fresh slate: unload anything currently resident so the test exercises
    # the cold-load → evict → cold-load path.
    talkies_method POST "/unload" >/dev/null 2>&1 || true

    if ! talkies_speech "$TTS_MODEL" "af_heart" "$TTS_TEST_PHRASE" "wav" "$wavfile"; then
        echo "  FAIL: kokoro synthesis"
        return 1
    fi
    local size
    size=$(stat -c %s "$wavfile" 2>/dev/null || stat -f %z "$wavfile" 2>/dev/null || echo 0)
    if [ "$size" -lt 4096 ]; then
        echo "  FAIL: synthesized wav too small ($size bytes)"
        return 1
    fi
    echo "  kokoro produced wav ($size bytes)"

    # Round-trip back through ASR. Plain json, no diarization, no language hint
    # (Whisper auto-detects; Parakeet/Canary default to English).
    local out text normalized
    out=$(talkies_transcribe "$asr_model" "$wavfile" "json") || {
        echo "  FAIL: ASR round-trip transcribe via $asr_model"
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
    for word in "${TTS_EXPECTED_WORDS[@]}"; do
        if [[ " $normalized " != *" $word "* ]]; then
            missing+=("$word")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "  FAIL: round-trip transcript missing words: ${missing[*]}"
        echo "  spoken phrase: \"$TTS_TEST_PHRASE\""
        echo "  raw asr text:  \"$text\""
        return 1
    fi
    echo "  ok: all expected words present (${TTS_EXPECTED_WORDS[*]})"
    echo "OK: talkies_speech_round_trip_through_asr"
}

ALL_TESTS+=(
    test_talkies_voices_list
    test_talkies_speech_all_formats
    test_talkies_speech_default_voice
    test_talkies_speech_unknown_voice_400
    test_talkies_speech_empty_input_400
    test_talkies_speech_rejects_asr_model
    test_talkies_transcribe_rejects_tts_model
    test_talkies_speech_round_trip_through_asr
)
