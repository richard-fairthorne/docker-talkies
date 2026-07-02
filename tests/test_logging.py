"""Unit tests for talkies.logging — level resolution + PII warning.

Pure-python; no ML deps. Mirrors test_config.py's reload-under-env pattern.
`_resolve_level` runs at container startup via configure(), so a bad
TALKIES_LOG_LEVEL now crashes the process on purpose (fail-fast, same
contract as config.py). These lock that behavior down.

Plan: .testing/2026-07-02/log-level-and-debug-bodies.md
"""

from __future__ import annotations

import importlib
import logging
import sys

import pytest


def _reload_logging(monkeypatch, **env: str):
    for var in ("TALKIES_LOG_LEVEL", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("talkies.logging", None)
    return importlib.import_module("talkies.logging")


# ── level resolution ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warn", logging.WARNING),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("fatal", logging.CRITICAL),
        ("critical", logging.CRITICAL),
        ("DEBUG", logging.DEBUG),
        ("Info", logging.INFO),
    ],
)
def test_resolve_level_accepts_all_names(monkeypatch, name, expected):
    mod = _reload_logging(monkeypatch, TALKIES_LOG_LEVEL=name)
    assert mod._resolve_level() == expected


def test_resolve_level_default_is_info(monkeypatch):
    mod = _reload_logging(monkeypatch)
    assert mod._resolve_level() == logging.INFO


def test_log_level_alias_takes_precedence_over_generic(monkeypatch):
    # TALKIES_LOG_LEVEL wins over LOG_LEVEL when both are set.
    mod = _reload_logging(monkeypatch, TALKIES_LOG_LEVEL="error", LOG_LEVEL="debug")
    assert mod._resolve_level() == logging.ERROR


def test_generic_log_level_used_when_prefixed_unset(monkeypatch):
    mod = _reload_logging(monkeypatch, LOG_LEVEL="warn")
    assert mod._resolve_level() == logging.WARNING


@pytest.mark.parametrize("bad", ["trace", "verbose", "notice", "10", "-"])
def test_resolve_level_bad_value_fails_loud(monkeypatch, bad):
    mod = _reload_logging(monkeypatch, TALKIES_LOG_LEVEL=bad)
    with pytest.raises(ValueError, match="not a valid level"):
        mod._resolve_level()


def test_empty_level_falls_back_to_info(monkeypatch):
    # Explicitly-empty env var = unset = default INFO, not an error.
    mod = _reload_logging(monkeypatch, TALKIES_LOG_LEVEL="")
    assert mod._resolve_level() == logging.INFO


# ── DEBUG emits the PII warning; higher levels don't ─────────────────────────
# configure() removes ALL root handlers (incl. pytest's caplog handler) and
# installs its own stdout JSON handler, so caplog can't see the record —
# assert on the JSON line captured from stdout via capsys instead.


def test_configure_debug_emits_pii_warning(monkeypatch, capsys):
    mod = _reload_logging(monkeypatch, TALKIES_LOG_LEVEL="debug")
    mod.configure()
    out = capsys.readouterr().out
    assert "PII" in out
    assert '"level": "WARNING"' in out


def test_configure_info_no_pii_warning(monkeypatch, capsys):
    mod = _reload_logging(monkeypatch, TALKIES_LOG_LEVEL="info")
    mod.configure()
    out = capsys.readouterr().out
    assert "PII" not in out
