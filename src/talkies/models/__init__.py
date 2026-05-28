"""Backend factory — build backends keyed by model_id from the registry."""

from __future__ import annotations

from typing import Any

from .. import config
from .kokoro import KokoroBackend
from .multitask import MultitaskBackend
from .parakeet import ParakeetBackend
from .salm import SalmBackend
from .whisper import WhisperBackend


def build_backends(registry: dict[str, dict], device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_id, entry in registry.items():
        executor = entry.get("executor", "whisper")
        repo = entry["repo"]
        model_path = config.MODELS_DIR / model_id
        if executor == "whisper":
            out[model_id] = WhisperBackend(
                model_id=model_id,
                repo=repo,
                model_path=model_path,
                device=device,
            )
            continue
        if executor == "parakeet":
            out[model_id] = ParakeetBackend(
                model_id=model_id,
                repo=repo,
                model_path=model_path,
                device=device,
            )
            continue
        if executor == "canary_salm":
            out[model_id] = SalmBackend(
                model_id=model_id,
                repo=repo,
                model_path=model_path,
                device=device,
            )
            continue
        if executor == "kokoro":
            out[model_id] = KokoroBackend(
                model_id=model_id,
                repo=repo,
                model_path=model_path,
                device=device,
            )
            continue
        out[model_id] = MultitaskBackend(
            model_id=model_id,
            repo=repo,
            model_path=model_path,
            device=device,
        )
    return out


def is_tts_backend(backend: Any) -> bool:
    """Backends are duck-typed on the route layer — TTS backends have
    ``synthesize`` and ``voices``, ASR backends have ``transcribe``.
    """
    return hasattr(backend, "synthesize") and hasattr(backend, "voices")


def is_asr_backend(backend: Any) -> bool:
    return hasattr(backend, "transcribe")
