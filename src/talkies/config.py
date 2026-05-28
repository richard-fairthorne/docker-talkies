"""Env-driven config — parsed at import time, fail-fast on bad input."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc


def _list_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [s.strip() for s in raw.split(",") if s.strip()]


_DURATION_RE = re.compile(
    r"^\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?\s*$",
    re.IGNORECASE,
)


def _duration_env(name: str, default: float) -> float:
    """Parse a duration env var.

    Accepts a bare number (seconds) or Go-style strings like "3h30m5s",
    "45m", "10s", "1h30m". Returns total seconds.
    """
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        pass
    match = _DURATION_RE.match(raw)
    if not match or not any(match.groups()):
        raise ValueError(
            f"{name}={raw!r} must be seconds (e.g. 600) or Go-style "
            "duration like '3h30m5s', '45m', '90s'"
        )
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


HOST: str = os.environ.get("TALKIES_HOST", "0.0.0.0")
PORT: int = _int_env("TALKIES_PORT", 8000)

DEVICE: str = os.environ.get("TALKIES_DEVICE", "auto").strip() or "auto"
if DEVICE not in ("auto", "cpu", "cuda") and not DEVICE.startswith("cuda:"):
    raise ValueError(
        f"TALKIES_DEVICE={DEVICE!r} must be 'auto', 'cpu', 'cuda', or 'cuda:N'"
    )

MODELS_FILE: Path = Path(
    os.environ.get("TALKIES_MODELS_FILE", "/app/models.json")
).resolve()

DATA_DIR: Path = Path(
    os.environ.get("TALKIES_DATA_DIR", "/data")
).resolve()

# Flat per-model snapshot directory: each enabled model gets
# DATA_DIR / models / <slug> / ... — populated by entrypoint.sh via
# snapshot_download(local_dir=...). Backends load directly from here;
# no HF cache, no models--org--repo/snapshots/<hash> indirection.
MODELS_DIR: Path = DATA_DIR / "models"

MODEL_IDLE_TIMEOUT_SECONDS: float = _duration_env("TALKIES_MODEL_TTL", 600.0)
SWEEPER_INTERVAL_SECONDS: float = _duration_env("TALKIES_SWEEPER_INTERVAL", 60.0)
LOAD_TIMEOUT_SECONDS: float = _duration_env("TALKIES_LOAD_TIMEOUT", 300.0)

MAX_UPLOAD_BYTES: int = _int_env("TALKIES_MAX_UPLOAD_BYTES", 100 * 1024 * 1024)

PRELOAD: list[str] = _list_env("TALKIES_PRELOAD")
ENABLED_MODELS: list[str] = _list_env("TALKIES_ENABLED_MODELS")

# VAD chunking — audio longer than this triggers VAD-based segmentation
# regardless of backend. Backends that don't support timeline assembly
# (canary SALM) ignore chunks and only see the first VAD region.
VAD_CHUNK_THRESHOLD_SECONDS: float = _float_env("TALKIES_VAD_CHUNK_THRESHOLD", 30.0)
VAD_MAX_SPEECH_SECONDS: float = _float_env("TALKIES_VAD_MAX_SPEECH", 28.0)
VAD_MIN_SILENCE_MS: int = _int_env("TALKIES_VAD_MIN_SILENCE_MS", 500)
VAD_SPEECH_PAD_MS: int = _int_env("TALKIES_VAD_SPEECH_PAD_MS", 200)
VAD_THRESHOLD: float = _float_env("TALKIES_VAD_THRESHOLD", 0.5)


def load_registry() -> dict[str, dict]:
    """Read models.json and return {model_id: {repo, executor, language?, ...}}."""
    if not MODELS_FILE.exists():
        raise FileNotFoundError(f"models.json not found at {MODELS_FILE}")
    with MODELS_FILE.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError(f"{MODELS_FILE}: expected top-level object with 'models' key")
    models = raw["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{MODELS_FILE}: 'models' must be a non-empty object")
    for model_id, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{MODELS_FILE}: model {model_id!r} entry must be an object")
        if "repo" not in entry:
            raise ValueError(f"{MODELS_FILE}: model {model_id!r} missing 'repo'")
        executor = entry.get("executor", "whisper")
        if executor not in ("whisper", "parakeet", "canary_multitask", "canary_salm"):
            raise ValueError(
                f"{MODELS_FILE}: model {model_id!r} executor={executor!r} must be one of "
                "'whisper', 'parakeet', 'canary_multitask', 'canary_salm'"
            )
    if ENABLED_MODELS:
        missing = [s for s in ENABLED_MODELS if s not in models]
        if missing:
            raise ValueError(
                f"TALKIES_ENABLED_MODELS references unknown slug(s) {missing}; "
                f"available in {MODELS_FILE}: {sorted(models)}"
            )
        models = {s: models[s] for s in ENABLED_MODELS}
    return models
