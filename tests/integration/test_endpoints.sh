#!/bin/bash
# shellcheck shell=bash disable=SC2154  # ALL_TESTS comes from common.sh

# Fast endpoint smoke tests — no transcription, no model loading. Verifies
# the HTTP surface is wired up and the registry is sane.

# ── /healthz reachable, returns the configured model_ids ─────────────────────

test_talkies_healthz() {
    local out mid
    out=$(talkies_get "/healthz") || { echo "  FAIL: /healthz unreachable"; return 1; }
    assert_contains "$out" "\"ok\":true" "/healthz ok=true" || return 1
    for mid in $(talkies_expected_models); do
        assert_contains "$out" "$mid" "/healthz lists $mid" || return 1
    done
    echo "OK: talkies_healthz"
}

# ── /v1/models exposes the OpenAI list shape with every configured slug ──────

test_talkies_models_list() {
    local out mid
    out=$(talkies_get "/v1/models") || { echo "  FAIL: /v1/models unreachable"; return 1; }
    assert_contains "$out" "\"object\":\"list\"" "/v1/models openai shape" || return 1
    for mid in $(talkies_expected_models); do
        assert_contains "$out" "\"$mid\"" "/v1/models has $mid" || return 1
    done
    echo "OK: talkies_models_list"
}

# ── /api/ps responds with speaches-compat shape ──────────────────────────────

test_talkies_api_ps() {
    local out
    out=$(talkies_get "/api/ps") || { echo "  FAIL: /api/ps unreachable"; return 1; }
    assert_contains "$out" "models" "/api/ps has models field" || return 1
    echo "OK: talkies_api_ps"
}

# ── POST /unload always 200 ──────────────────────────────────────────────────

test_talkies_unload_all() {
    local code
    code=$(talkies_method_status POST "/unload")
    assert_eq "$code" "200" "/unload → 200" || return 1
    echo "OK: talkies_unload_all"
}

# ── DELETE /api/ps/{unknown} returns 404 ─────────────────────────────────────

test_talkies_delete_unknown_returns_404() {
    local code
    code=$(talkies_method_status DELETE "/api/ps/this-model-does-not-exist")
    assert_eq "$code" "404" "DELETE unknown model → 404" || return 1
    echo "OK: talkies_delete_unknown_returns_404"
}

# ── /v1/audio/transcriptions without a file → 422 (FastAPI multipart) ────────

test_talkies_transcribe_missing_file_returns_422() {
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -X POST -F "model=whisper-large-v3-turbo" \
        "${TALKIES_BASE_URL}/v1/audio/transcriptions")
    assert_eq "$code" "422" "missing file → 422" || return 1
    echo "OK: talkies_transcribe_missing_file_returns_422"
}

# ── /v1/audio/transcriptions with unknown model → 404 ────────────────────────

test_talkies_transcribe_unknown_model_returns_404() {
    local fixture
    fixture=$(talkies_find_fixture)
    if [ -z "$fixture" ]; then
        echo "  SKIP: tests/integration/.fixtures/audio.* missing"
        return 0
    fi
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 \
        -F "model=not-a-real-slug" \
        -F "file=@${fixture}" \
        "${TALKIES_BASE_URL}/v1/audio/transcriptions")
    assert_eq "$code" "404" "unknown model → 404" || return 1
    echo "OK: talkies_transcribe_unknown_model_returns_404"
}

ALL_TESTS+=(
    test_talkies_healthz
    test_talkies_models_list
    test_talkies_api_ps
    test_talkies_unload_all
    test_talkies_delete_unknown_returns_404
    test_talkies_transcribe_missing_file_returns_422
    test_talkies_transcribe_unknown_model_returns_404
)
