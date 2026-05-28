"""Backend protocol — uniform load/transcribe/unload surface per executor type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TranscribeResult:
    """Backend output. `segments` / `words` populated only when the backend
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


class Backend(Protocol):
    """Per-model handle.

    Backends are instantiated lazily on first request — `get_model()` populates
    the underlying NeMo object; later calls return the cached instance until
    `unload()` is called (manually or by the idle sweeper).
    """

    model_id: str
    repo: str

    async def get_model(self) -> object: ...

    async def transcribe(
        self,
        audio_path: str,
        *,
        source_lang: str | None,
        target_lang: str | None,
        task: str,
        with_timestamps: bool = False,
    ) -> TranscribeResult: ...

    async def unload(self) -> None: ...

    def loaded(self) -> bool: ...

    def last_used_secs_ago(self) -> float | None: ...
