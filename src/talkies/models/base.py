"""Backend protocols — uniform lifecycle for ASR and TTS executors.

All backends share lazy load + idle-unload semantics (``loaded`` /
``unload`` / ``get_model`` / ``last_used_secs_ago``) so the idle sweeper
and sibling-eviction logic in ``server.py`` work uniformly across
modalities — one model resident at a time across the shared VRAM/RAM
pool.

ASR backends add ``transcribe()``; TTS backends add ``synthesize()`` +
voice introspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TranscribeResult:
    """ASR backend output. `segments` / `words` populated only when the backend
    supports timestamps and the caller requested them.

    Segments / words follow the OpenAI verbose_json shape:
        segment: {"id": int, "start": float, "end": float, "text": str}
        word:    {"word": str, "start": float, "end": float}

    Whisper-only fields (avg_logprob, no_speech_prob, compression_ratio, tokens)
    are filled with null/empty values by the server when emitting verbose_json
    so OpenAI clients reading them don't crash.
    """

    text: str
    segments: list[dict] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    supports_timestamps: bool = False


@dataclass
class SynthesisResult:
    """TTS backend output — raw mono int16 PCM at ``sample_rate`` Hz.

    The router (``tts.encode_audio``) converts these bytes into the
    requested ``response_format`` (mp3/opus/aac/flac/wav/pcm).
    """

    pcm_int16: bytes
    sample_rate: int


class BackendBase(Protocol):
    """Lifecycle surface shared by all backends.

    Backends are instantiated lazily on first request — ``get_model()`` populates
    the underlying model; later calls return the cached instance until
    ``unload()`` is called (manually or by the idle sweeper).
    """

    model_id: str
    repo: str

    async def get_model(self) -> object: ...

    async def unload(self) -> None: ...

    def loaded(self) -> bool: ...

    def last_used_secs_ago(self) -> float | None: ...


class ASRBackend(BackendBase, Protocol):
    """Per-ASR-model handle."""

    async def transcribe(
        self,
        audio_path: str,
        *,
        source_lang: str | None,
        target_lang: str | None,
        task: str,
        with_timestamps: bool = False,
    ) -> TranscribeResult: ...


class TTSBackend(BackendBase, Protocol):
    """Per-TTS-model handle.

    Each TTS backend declares its own voice catalog — the route validates
    the caller-supplied ``voice`` against ``voices()`` for the requested
    model. No cross-model voice aliasing.
    """

    def voices(self) -> list[str]: ...

    def default_voice(self) -> str: ...

    async def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        instructions: str | None = None,
    ) -> SynthesisResult: ...


Backend = ASRBackend
"""Backwards-compatibility alias — older code imported ``Backend`` for ASR."""
