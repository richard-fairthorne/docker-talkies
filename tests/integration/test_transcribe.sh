#!/bin/bash
# Per-model transcription tests. Requires a fixture at
# tests/integration/.fixtures/audio.<wav|mp3|m4a|flac|ogg> — anything else
# skips with a clear message.
#
# Self-contained: spawns its own --rm --gpus all container via the harness,
# tears it down on exit. Invoke directly: bash tests/integration/test_transcribe.sh
#
# CUDA-only by design: on a CPU host even one whisper-large-v3 inference
# can take minutes. Don't run against the CPU image.

set -eo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=harness.sh
source "${_DIR}/harness.sh"
# shellcheck source=common.sh
source "${_DIR}/common.sh"

# All ASR slugs in the registry — the suite exercises every one. Override via
# TRANSCRIBE_MODELS if you want to scope to fewer (e.g. when iterating).
TRANSCRIBE_MODELS="${TRANSCRIBE_MODELS:-whisper-large-v3,whisper-large-v3-turbo,parakeet-tdt-0.6b-v3,canary-180m-flash,canary-1b-flash,canary-qwen-2.5b}"
harness_start "$TRANSCRIBE_MODELS"

# ── plain json: every model returns non-empty text ───────────────────────────

test_talkies_transcribe_each_model_json() {
    local fixture
    fixture=$(talkies_find_fixture)
    if [ -z "$fixture" ]; then
        echo "  SKIP: tests/integration/.fixtures/audio.* missing"
        return 0
    fi

    local mid out text rc=0
    for mid in $(talkies_expected_asr_models); do
        out=$(talkies_transcribe "$mid" "$fixture" "json") || {
            echo "  FAIL: $mid json transcribe"
            rc=1
            continue
        }
        text=$(echo "$out" | jq -r '.text' 2>/dev/null || echo "")
        if [ -z "$text" ] || [ "$text" = "null" ]; then
            echo "  FAIL: $mid empty text in json response"
            rc=1
            continue
        fi
        echo "  ok: $mid text=\"$(echo "$text" | head -c 80)\""
    done
    if [ "$rc" -eq 0 ]; then
        echo "OK: $FUNCNAME"
    fi
    return $rc
}

# ── verbose_json: full Whisper-shape envelope, segments/words where supported

test_talkies_transcribe_each_model_verbose_json() {
    local fixture
    fixture=$(talkies_find_fixture)
    if [ -z "$fixture" ]; then
        echo "  SKIP: tests/integration/.fixtures/audio.* missing"
        return 0
    fi

    local mid out rc=0 segs words
    for mid in $(talkies_expected_asr_models); do
        out=$(talkies_transcribe "$mid" "$fixture" "verbose_json" \
            "timestamp_granularities[]=segment" \
            "timestamp_granularities[]=word") || {
            echo "  FAIL: $mid verbose_json transcribe"
            rc=1
            continue
        }
        assert_contains "$out" "\"task\":"     "$mid verbose_json task"     || { rc=1; continue; }
        assert_contains "$out" "\"language\":" "$mid verbose_json language" || { rc=1; continue; }
        assert_contains "$out" "\"duration\":" "$mid verbose_json duration" || { rc=1; continue; }
        assert_contains "$out" "\"segments\":" "$mid verbose_json segments" || { rc=1; continue; }
        assert_contains "$out" "\"words\":"    "$mid verbose_json words"    || { rc=1; continue; }
        segs=$(echo "$out"  | jq '.segments | length' 2>/dev/null || echo 0)
        words=$(echo "$out" | jq '.words    | length' 2>/dev/null || echo 0)
        # canary-qwen-2.5b (SALM) has no timestamp head — empty arrays are
        # the correct response, schema must still validate.
        if [ "$mid" = "canary-qwen-2.5b" ]; then
            echo "  ok: $mid (SALM, segments=$segs words=$words)"
            continue
        fi
        if [ "$segs" -lt 1 ]; then
            echo "  FAIL: $mid expected >=1 segment, got $segs"
            rc=1
            continue
        fi
        echo "  ok: $mid segments=$segs words=$words"
    done
    if [ "$rc" -eq 0 ]; then
        echo "OK: $FUNCNAME"
    fi
    return $rc
}

# ── srt: every backend returns subtitle blocks with timestamp arrows ─────────

test_talkies_transcribe_each_model_srt() {
    local fixture
    fixture=$(talkies_find_fixture)
    if [ -z "$fixture" ]; then
        echo "  SKIP: tests/integration/.fixtures/audio.* missing"
        return 0
    fi

    local mid out rc=0
    for mid in $(talkies_expected_asr_models); do
        out=$(talkies_transcribe "$mid" "$fixture" "srt") || {
            echo "  FAIL: $mid srt transcribe"
            rc=1
            continue
        }
        if ! echo "$out" | grep -q -- "-->"; then
            echo "  FAIL: $mid srt missing timestamp arrows"
            rc=1
            continue
        fi
        echo "  ok: $mid srt"
    done
    if [ "$rc" -eq 0 ]; then
        echo "OK: $FUNCNAME"
    fi
    return $rc
}

# ── vtt: WEBVTT header + at least one timestamp arrow ────────────────────────

test_talkies_transcribe_each_model_vtt() {
    local fixture
    fixture=$(talkies_find_fixture)
    if [ -z "$fixture" ]; then
        echo "  SKIP: tests/integration/.fixtures/audio.* missing"
        return 0
    fi

    local mid out rc=0
    for mid in $(talkies_expected_asr_models); do
        out=$(talkies_transcribe "$mid" "$fixture" "vtt") || {
            echo "  FAIL: $mid vtt transcribe"
            rc=1
            continue
        }
        assert_contains "$out" "WEBVTT" "$mid vtt has WEBVTT header" || { rc=1; continue; }
        if ! echo "$out" | grep -q -- "-->"; then
            echo "  FAIL: $mid vtt missing timestamp arrows"
            rc=1
            continue
        fi
        echo "  ok: $mid vtt"
    done
    if [ "$rc" -eq 0 ]; then
        echo "OK: $FUNCNAME"
    fi
    return $rc
}

# ── /api/ps reflects a loaded model after a real transcription ───────────────

test_talkies_api_ps_reflects_loaded_model() {
    local fixture mid="whisper-large-v3-turbo"
    fixture=$(talkies_find_fixture)
    if [ -z "$fixture" ]; then
        echo "  SKIP: tests/integration/.fixtures/audio.* missing"
        return 0
    fi
    talkies_method POST "/unload" >/dev/null 2>&1 || true

    talkies_transcribe "$mid" "$fixture" "json" >/dev/null || {
        echo "  FAIL: warm-up transcription failed"
        return 1
    }

    local ps
    ps=$(talkies_get "/api/ps") || { echo "  FAIL: /api/ps after load"; return 1; }
    assert_contains "$ps" "$mid" "/api/ps lists loaded $mid" || return 1
    echo "OK: $FUNCNAME"
}

# ── DELETE /api/ps/{slug} unloads a previously-loaded model ──────────────────

test_talkies_api_ps_delete_unloads() {
    local fixture mid="whisper-large-v3-turbo"
    fixture=$(talkies_find_fixture)
    if [ -z "$fixture" ]; then
        echo "  SKIP: tests/integration/.fixtures/audio.* missing"
        return 0
    fi
    talkies_transcribe "$mid" "$fixture" "json" >/dev/null || {
        echo "  FAIL: warm-up transcription failed"
        return 1
    }

    local code
    code=$(talkies_method_status DELETE "/api/ps/$mid")
    assert_eq "$code" "200" "DELETE /api/ps/$mid → 200" || return 1

    local ps
    ps=$(talkies_get "/api/ps") || { echo "  FAIL: /api/ps after unload"; return 1; }
    if echo "$ps" | jq -e --arg mid "$mid" '.models[] | select(.name==$mid)' >/dev/null 2>&1; then
        echo "  FAIL: $mid still listed in /api/ps after unload"
        return 1
    fi
    echo "OK: $FUNCNAME"
}

harness_run_tests \
    test_talkies_transcribe_each_model_json \
    test_talkies_transcribe_each_model_verbose_json \
    test_talkies_transcribe_each_model_srt \
    test_talkies_transcribe_each_model_vtt \
    test_talkies_api_ps_reflects_loaded_model \
    test_talkies_api_ps_delete_unloads
