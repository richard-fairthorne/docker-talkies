#!/bin/bash
# Qwen3-TTS end-to-end across all three modes (base / custom_voice /
# voice_design). Each model variant synthesises the same English phrase
# using whatever knobs that mode supports, then the WAV gets transcribed
# by a fast ASR backend and we assert all expected words round-trip.
#
# Mode coverage:
#   base                qwen3-tts-0.6b          voice=alloy, instructions threaded
#   custom_voice 0.6b   qwen3-tts-0.6b-custom   voice=Ryan, instructions IGNORED
#                                               by upstream (silently dropped)
#   custom_voice 1.7b   qwen3-tts-1.7b-custom   voice=Ryan + instructions="clearly"
#   voice_design 1.7b   qwen3-tts-1.7b-design   instructions=NL voice description
#
# This is the heaviest test in the suite — needs ~20 GB of snapshot
# downloads on first run (5 enabled slugs + ASR). HARNESS_READY_TIMEOUT
# is bumped accordingly. Cache reuse across runs is automatic via
# HARNESS_CACHE_DIR.
#
# Invoke directly:  bash tests/integration/e2e_qwen3_modes.sh

set -eo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=harness.sh
source "${_DIR}/harness.sh"
# shellcheck source=common.sh
source "${_DIR}/common.sh"

# Snapshot downloads can be slow on first run — give the entrypoint
# plenty of headroom before the harness gives up waiting for /healthz.
export HARNESS_READY_TIMEOUT="${HARNESS_READY_TIMEOUT:-3600}"

QWEN3_MODES_MODELS="whisper-large-v3-turbo,qwen3-tts-0.6b,qwen3-tts-1.7b,qwen3-tts-0.6b-custom,qwen3-tts-1.7b-custom,qwen3-tts-1.7b-design"
harness_start "$QWEN3_MODES_MODELS"

PHRASE="The quick brown fox jumps over the lazy dog."
EXPECTED_WORDS=(quick brown fox jumps lazy dog)
ASR_MODEL=""

# Voice-cloning reference WAV — converted once at suite startup from the
# shared audio.mp3 fixture. Sibling .txt with the spoken transcript is
# committed to fixtures so we don't depend on a live ASR for the canonical
# text. Conversion uses ffmpeg inside the talkies image (already on PATH).
FIXTURE_MP3="${_HARNESS_REPO_ROOT}/tests/integration/.fixtures/audio.mp3"
FIXTURE_TXT="${_HARNESS_REPO_ROOT}/tests/integration/.fixtures/audio.mp3.txt"
REF_WAV_HOST=""
REF_TEXT=""

prepare_reference_clone_wav() {
    [ -n "$REF_WAV_HOST" ] && return 0
    if [ ! -f "$FIXTURE_MP3" ] || [ ! -f "$FIXTURE_TXT" ]; then
        echo "  SKIP: missing fixture(s) $FIXTURE_MP3 / $FIXTURE_TXT"
        return 1
    fi
    REF_TEXT=$(tr -d '\r\n' < "$FIXTURE_TXT")
    [ -n "$REF_TEXT" ] || { echo "  SKIP: audio.mp3.txt empty"; return 1; }
    local stage="$HARNESS_CACHE_DIR/.test-fixtures"
    mkdir -p "$stage"
    REF_WAV_HOST="$stage/audio.wav"
    if [ ! -f "$REF_WAV_HOST" ]; then
        echo "  preparing reference WAV via ffmpeg (one-shot)…"
        docker run --rm \
            -v "$FIXTURE_MP3:/in.mp3:ro" \
            -v "$stage:/out" \
            --entrypoint ffmpeg \
            "$HARNESS_IMAGE" \
            -hide_banner -loglevel error -y \
            -i /in.mp3 -ac 1 -ar 24000 -c:a pcm_s16le /out/audio.wav \
            >/dev/null 2>&1 || {
                echo "  FAIL: ffmpeg conversion of audio.mp3 → audio.wav"
                return 1
            }
    fi
    echo "  ref wav ready: $REF_WAV_HOST  (transcript: \"$REF_TEXT\")"
    return 0
}

resolve_asr() {
    [ -n "$ASR_MODEL" ] && return 0
    ASR_MODEL=$(talkies_pick_fast_asr_model) || {
        echo "  FAIL: no ASR backend available on server"
        return 1
    }
    echo "  using asr_model=$ASR_MODEL"
}

# ── per-mode synth helper ────────────────────────────────────────────────────
# Args:
#   $1 = model slug
#   $2 = voice (empty string for default)
#   $3 = instructions (empty string to omit)
#   $4 = language (empty string for default)
#   $5 = output wav path
# Returns 0 on HTTP 200 + RIFF + size>=4 KB, 1 otherwise.
synth_with_params() {
    local model="$1" voice="$2" instructions="$3" language="$4" outfile="$5"
    local body filters=()
    filters=(--arg m "$model" --arg t "$PHRASE")
    local jq_obj='{model:$m, input:$t, response_format:"wav"}'
    if [ -n "$voice" ]; then
        filters+=(--arg v "$voice")
        jq_obj="${jq_obj} + {voice:\$v}"
    fi
    if [ -n "$instructions" ]; then
        filters+=(--arg i "$instructions")
        jq_obj="${jq_obj} + {instructions:\$i}"
    fi
    if [ -n "$language" ]; then
        filters+=(--arg l "$language")
        jq_obj="${jq_obj} + {language:\$l}"
    fi
    body=$(jq -n "${filters[@]}" "$jq_obj") || return 1

    local code
    code=$(curl -s -o "$outfile" -w "%{http_code}" --max-time 600 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech") || return 1
    if [ "$code" != "200" ]; then
        echo "  HTTP $code body head: $(head -c 400 "$outfile")"
        return 1
    fi
    local size head4
    size=$(stat -c %s "$outfile" 2>/dev/null || stat -f %z "$outfile" 2>/dev/null || echo 0)
    head4=$(head -c 4 "$outfile" | od -An -c | tr -d ' \n')
    if [ "$size" -lt 4096 ] || [ "$head4" != "RIFF" ]; then
        echo "  malformed wav (size=$size head='$head4')"
        return 1
    fi
    echo "  wav size=${size}B"
    return 0
}

# Round-trip a synthesized wav through ASR and assert every expected word.
assert_round_trip() {
    local wavfile="$1"
    local out text normalized missing=() word
    out=$(talkies_transcribe "$ASR_MODEL" "$wavfile" "json") || {
        echo "  FAIL: ASR via $ASR_MODEL"
        return 1
    }
    text=$(echo "$out" | jq -r '.text' 2>/dev/null || echo "")
    if [ -z "$text" ] || [ "$text" = "null" ]; then
        echo "  FAIL: ASR returned empty text"
        return 1
    fi
    normalized=$(echo "$text" | talkies_normalize_text)
    echo "  transcribed: \"$normalized\""
    for word in "${EXPECTED_WORDS[@]}"; do
        if [[ " $normalized " != *" $word "* ]]; then
            missing+=("$word")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "  FAIL: missing words: ${missing[*]}"
        return 1
    fi
    return 0
}

# Free GPU memory before loading the next variant. The sibling-eviction
# logic in /v1/audio/speech does this automatically when a TTS request
# hits a different model, but we also POST /unload between cases to
# stress that path explicitly + keep VRAM clean if a model leaks.
gpu_reset() {
    talkies_method POST "/unload" >/dev/null 2>&1 || true
}

# ── base-mode voice-cloning helpers (use audio.mp3 fixture as ref WAV) ───────
# Two modes per model:
#   * xvec  — wav only, no sibling .txt → backend logs warning + uses
#             x-vector-only mode (lower fidelity, still intelligible).
#   * icl   — wav + sibling .txt containing the ref transcript → backend
#             runs full ICL voice cloning (higher fidelity).
# After synth we ASR-round-trip the result to verify intelligibility.
# Parametrised by model slug so the same logic exercises both 0.6b and 1.7b
# base checkpoints.

run_clone_round_trip() {
    local model="$1" mode="$2"  # mode: "xvec" or "icl"
    [ "$mode" = "xvec" ] || [ "$mode" = "icl" ] || {
        echo "  FAIL: bad mode $mode"; return 2
    }
    resolve_asr || return 1
    prepare_reference_clone_wav || return 1
    gpu_reset

    local stamp="$$_$(date +%s%N)"
    local custom_dir="$HARNESS_CACHE_DIR/custom-voices/e2e_modes"
    local voice_basename="clone_${mode}_${stamp}"
    local voice_name="e2e_modes/${voice_basename}"
    mkdir -p "$custom_dir"
    cp "$REF_WAV_HOST" "$custom_dir/${voice_basename}.wav"
    if [ "$mode" = "icl" ]; then
        # ICL: sibling .txt = transcript of ref WAV.
        printf '%s\n' "$REF_TEXT" > "$custom_dir/${voice_basename}.txt"
        printf 'English\n' > "$custom_dir/${voice_basename}.lang"
    else
        # xvec: deliberately NO .txt sibling — drives the fallback path.
        printf 'English\n' > "$custom_dir/${voice_basename}.lang"
    fi
    local cleanup="rm -f \
        '$custom_dir/${voice_basename}.wav' \
        '$custom_dir/${voice_basename}.txt' \
        '$custom_dir/${voice_basename}.lang'"
    # shellcheck disable=SC2064
    trap "$cleanup" RETURN

    # Verify live discovery + correct origin
    local voices_out origin
    voices_out=$(talkies_get "/v1/audio/voices") || {
        echo "  FAIL: /v1/audio/voices unreachable"; return 1
    }
    origin=$(echo "$voices_out" | jq -r --arg m "$model" --arg v "$voice_name" \
        '.voices[] | select(.model==$m and .voice==$v) | .origin')
    if [ "$origin" != "custom" ]; then
        echo "  FAIL: $voice_name not picked up (origin='$origin')"
        return 1
    fi
    echo "  voice=$voice_name origin=custom mode=$mode"

    local tmp wavfile
    tmp=$(mktemp -d -t qwen3_clone.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'; $cleanup" RETURN
    wavfile="${tmp}/clone.wav"
    if ! synth_with_params "$model" "$voice_name" "" "" "$wavfile"; then
        echo "  FAIL: $model $mode synth"
        return 1
    fi
    if ! assert_round_trip "$wavfile"; then
        return 1
    fi
    return 0
}

test_qwen3_clone_xvec_0_6b()  { run_clone_round_trip qwen3-tts-0.6b xvec  && echo "OK: $FUNCNAME"; }
test_qwen3_clone_icl_0_6b()   { run_clone_round_trip qwen3-tts-0.6b icl   && echo "OK: $FUNCNAME"; }
test_qwen3_clone_xvec_1_7b()  { run_clone_round_trip qwen3-tts-1.7b xvec  && echo "OK: $FUNCNAME"; }
test_qwen3_clone_icl_1_7b()   { run_clone_round_trip qwen3-tts-1.7b icl   && echo "OK: $FUNCNAME"; }

# ── base mode: voice cloning from baked-in alloy.wav ─────────────────────────

test_qwen3_base_round_trip() {
    resolve_asr || return 1
    gpu_reset
    local tmp wavfile
    tmp=$(mktemp -d -t qwen3_base.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN
    wavfile="${tmp}/base.wav"
    echo "  [base] voice=alloy instructions='Speak clearly'"
    if ! synth_with_params "qwen3-tts-0.6b" "alloy" "Speak clearly." "" "$wavfile"; then
        echo "  FAIL: base mode synth"
        return 1
    fi
    if ! assert_round_trip "$wavfile"; then
        return 1
    fi
    echo "OK: $FUNCNAME"
}

# ── custom_voice 0.6b: preset speaker, NO instructions support ───────────────

test_qwen3_custom_voice_0_6b_round_trip() {
    resolve_asr || return 1
    gpu_reset
    local tmp wavfile
    tmp=$(mktemp -d -t qwen3_cv06.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN
    wavfile="${tmp}/custom06.wav"
    echo "  [custom_voice 0.6b] voice=Ryan"
    if ! synth_with_params "qwen3-tts-0.6b-custom" "Ryan" "" "English" "$wavfile"; then
        echo "  FAIL: custom_voice 0.6b synth"
        return 1
    fi
    if ! assert_round_trip "$wavfile"; then
        return 1
    fi
    echo "OK: $FUNCNAME"
}

# ── custom_voice 1.7b: preset speaker + instructions (emotion) ───────────────

test_qwen3_custom_voice_1_7b_with_instructions_round_trip() {
    resolve_asr || return 1
    gpu_reset
    local tmp wavfile
    tmp=$(mktemp -d -t qwen3_cv17.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN
    wavfile="${tmp}/custom17.wav"
    echo "  [custom_voice 1.7b] voice=Ryan instructions='Speak calmly and clearly'"
    if ! synth_with_params "qwen3-tts-1.7b-custom" "Ryan" \
        "Speak calmly and clearly." "English" "$wavfile"; then
        echo "  FAIL: custom_voice 1.7b synth"
        return 1
    fi
    if ! assert_round_trip "$wavfile"; then
        return 1
    fi
    echo "OK: $FUNCNAME"
}

# ── voice_design 1.7b: NL voice description in `instructions` ────────────────

test_qwen3_voice_design_round_trip() {
    resolve_asr || return 1
    gpu_reset
    local tmp wavfile
    tmp=$(mktemp -d -t qwen3_design.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN
    wavfile="${tmp}/design.wav"
    echo "  [voice_design] instructions='A clear American male voice, neutral tone'"
    if ! synth_with_params "qwen3-tts-1.7b-design" "" \
        "A clear American male voice with a neutral, friendly tone." \
        "English" "$wavfile"; then
        echo "  FAIL: voice_design synth"
        return 1
    fi
    if ! assert_round_trip "$wavfile"; then
        return 1
    fi
    echo "OK: $FUNCNAME"
}

# ── sampling extra params honored (greedy + tight top-k → still intelligible) ─

test_qwen3_sampling_extra_params_round_trip() {
    resolve_asr || return 1
    gpu_reset
    local outfile body code
    outfile=$(mktemp -t qwen3_sampling.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -f '$outfile'" RETURN
    # Pin every sampling knob the API supports; if any one is rejected the
    # request 422s and this test catches the regression. do_sample=false is
    # the strongest signal — switches the model into greedy decode.
    body=$(jq -n \
        --arg m "qwen3-tts-1.7b-custom" \
        --arg v "Ryan" \
        --arg t "$PHRASE" \
        '{
            model:$m, voice:$v, input:$t, response_format:"wav",
            temperature: 0.7,
            top_k: 30,
            top_p: 0.95,
            repetition_penalty: 1.1,
            max_new_tokens: 1024,
            do_sample: false,
            language: "English"
         }')
    code=$(curl -s -o "$outfile" -w "%{http_code}" --max-time 600 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    if [ "$code" != "200" ]; then
        echo "  FAIL: sampling-params request HTTP $code body=$(head -c 400 "$outfile")"
        return 1
    fi
    local size head4
    size=$(stat -c %s "$outfile" 2>/dev/null || stat -f %z "$outfile" 2>/dev/null || echo 0)
    head4=$(head -c 4 "$outfile" | od -An -c | tr -d ' \n')
    if [ "$size" -lt 4096 ] || [ "$head4" != "RIFF" ]; then
        echo "  FAIL: sampling wav malformed (size=$size head='$head4')"
        return 1
    fi
    if ! assert_round_trip "$outfile"; then
        return 1
    fi
    echo "  ok: all 6 sampling knobs accepted, wav size=${size}B"
    echo "OK: $FUNCNAME"
}

# ── sampling out-of-range → 422 ───────────────────────────────────────────────

test_qwen3_sampling_out_of_range_422() {
    local outfile body code
    outfile=$(mktemp -t qwen3_sampling_oor.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -f '$outfile'" RETURN
    # temperature > 2.0 must fail Pydantic validation before reaching the backend.
    body=$(jq -n --arg m "qwen3-tts-1.7b-custom" --arg t "$PHRASE" \
        '{model:$m, voice:"Ryan", input:$t, response_format:"wav", temperature: 5.0}')
    code=$(curl -s -o "$outfile" -w "%{http_code}" --max-time 30 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    if [ "$code" != "422" ]; then
        echo "  FAIL: temperature=5.0 expected 422 got $code"
        return 1
    fi
    echo "  ok: temperature out-of-range → 422"
    echo "OK: $FUNCNAME"
}

# ── voice_design 1.7b: missing instructions → 400 ────────────────────────────

test_qwen3_voice_design_missing_instructions_400() {
    local outfile body code
    outfile=$(mktemp -t qwen3_design_err.XXXXXX) || return 2
    # shellcheck disable=SC2064
    trap "rm -f '$outfile'" RETURN
    body=$(jq -n --arg m "qwen3-tts-1.7b-design" --arg t "$PHRASE" \
        '{model:$m, input:$t, response_format:"wav"}')
    code=$(curl -s -o "$outfile" -w "%{http_code}" --max-time 60 \
        -H "Content-Type: application/json" -d "$body" \
        "${TALKIES_BASE_URL}/v1/audio/speech")
    if [ "$code" != "400" ]; then
        echo "  FAIL: voice_design without instructions HTTP $code (expected 400)"
        echo "  body: $(head -c 300 "$outfile")"
        return 1
    fi
    echo "  ok: empty instructions → 400"
    echo "OK: $FUNCNAME"
}

# ── /v1/audio/voices reports right catalog per mode ──────────────────────────

test_qwen3_modes_voices_catalog() {
    local out
    out=$(talkies_get "/v1/audio/voices") || { echo "  FAIL: voices unreachable"; return 1; }

    # custom_voice models should expose the 9 preset speakers
    local cv_voices
    cv_voices=$(echo "$out" | jq -r --arg m "qwen3-tts-1.7b-custom" \
        '[.voices[] | select(.model==$m) | .voice] | sort | join(",")')
    if [ "$cv_voices" != "Aiden,Dylan,Eric,Ono_Anna,Ryan,Serena,Sohee,Uncle_Fu,Vivian" ]; then
        echo "  FAIL: custom 1.7b voices unexpected: '$cv_voices'"
        return 1
    fi
    echo "  ok: custom 1.7b lists 9 preset speakers"

    # voice_design model should expose the sentinel only
    local vd_voices
    vd_voices=$(echo "$out" | jq -r --arg m "qwen3-tts-1.7b-design" \
        '[.voices[] | select(.model==$m) | .voice] | sort | join(",")')
    if [ "$vd_voices" != "design" ]; then
        echo "  FAIL: design voices unexpected: '$vd_voices' (expected 'design')"
        return 1
    fi
    echo "  ok: design model lists ['design'] sentinel"

    # base model still scans WAV catalog (must have at least 1 builtin)
    local base_builtin_count
    base_builtin_count=$(echo "$out" | jq --arg m "qwen3-tts-0.6b" \
        '[.voices[] | select(.model==$m) | select(.origin=="builtin")] | length')
    if [ "$base_builtin_count" -lt 1 ]; then
        echo "  FAIL: base model has no builtin voices ($base_builtin_count)"
        return 1
    fi
    echo "  ok: base model has $base_builtin_count builtin voice(s)"
    echo "OK: $FUNCNAME"
}

harness_run_tests \
    test_qwen3_modes_voices_catalog \
    test_qwen3_sampling_out_of_range_422 \
    test_qwen3_voice_design_missing_instructions_400 \
    test_qwen3_base_round_trip \
    test_qwen3_clone_xvec_0_6b \
    test_qwen3_clone_icl_0_6b \
    test_qwen3_clone_xvec_1_7b \
    test_qwen3_clone_icl_1_7b \
    test_qwen3_custom_voice_0_6b_round_trip \
    test_qwen3_custom_voice_1_7b_with_instructions_round_trip \
    test_qwen3_sampling_extra_params_round_trip \
    test_qwen3_voice_design_round_trip
