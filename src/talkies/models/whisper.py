"""Whisper backend (faster-whisper).

Native VAD-chunked transcription: silero-vad detects speech regions, the
wrapper merges them into ~28s chunks, and faster-whisper's `clip_timestamps`
parameter transcribes only those windows while still returning timestamps
on the absolute audio timeline (so timeline assembly is free).

For audio short enough to skip VAD, we transcribe the whole file in one
pass with `vad_filter=False` (no internal silero pass — we control it).
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from pathlib import Path
from typing import Any

from .. import config
from .. import vad as vad_mod
from .base import TranscribeResult


def _ct2_compute_type(device: str) -> str:
    """Pick a ctranslate2 compute_type that works on the target device."""
    if device.startswith("cuda"):
        return "float16"
    return "int8"


class WhisperBackend:
    def __init__(
        self, model_id: str, repo: str, model_path: Path, device: str
    ) -> None:
        self.model_id = model_id
        self.repo = repo
        self.model_path = model_path
        self._device = device
        self._lock = asyncio.Lock()
        self._model: Any = None
        self._last_used: float | None = None
        self._log = logging.getLogger(f"talkies.whisper.{model_id}")

    def loaded(self) -> bool:
        return self._model is not None

    def last_used_secs_ago(self) -> float | None:
        if self._last_used is None:
            return None
        return time.monotonic() - self._last_used

    async def get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            self._log.info("loading %s onto %s", self.repo, self._device)
            self._model = await asyncio.to_thread(self._load_sync)
            self._log.info("loaded %s", self.repo)
            return self._model

    def _load_sync(self) -> Any:
        from faster_whisper import WhisperModel

        device = "cuda" if self._device.startswith("cuda") else "cpu"
        compute_type = _ct2_compute_type(self._device)
        return WhisperModel(
            str(self.model_path), device=device, compute_type=compute_type
        )

    async def transcribe(
        self,
        audio_path: str,
        *,
        source_lang: str | None,
        target_lang: str | None,
        task: str,
        with_timestamps: bool = False,
    ) -> TranscribeResult:
        del target_lang  # whisper transcribes in source language; AST not supported here
        model = await self.get_model()
        async with self._lock:
            result = await asyncio.to_thread(
                self._transcribe_sync,
                model,
                audio_path,
                source_lang,
                task,
                with_timestamps,
            )
            self._last_used = time.monotonic()
            return result

    def _transcribe_sync(
        self,
        model: Any,
        audio_path: str,
        source_lang: str | None,
        task: str,
        with_timestamps: bool,
    ) -> TranscribeResult:
        audio = vad_mod.load_wav_16k_mono(audio_path)
        duration = audio.shape[0] / vad_mod.SAMPLE_RATE

        clip_timestamps: list[float] | None = None
        if duration > config.VAD_CHUNK_THRESHOLD_SECONDS:
            regions = vad_mod.detect_speech_regions(
                audio,
                threshold=config.VAD_THRESHOLD,
                min_silence_ms=config.VAD_MIN_SILENCE_MS,
                speech_pad_ms=config.VAD_SPEECH_PAD_MS,
            )
            chunks = vad_mod.merge_speech_regions(
                regions, max_speech_seconds=config.VAD_MAX_SPEECH_SECONDS
            )
            if chunks:
                clip_timestamps = []
                for c in chunks:
                    clip_timestamps.append(c.start_seconds)
                    clip_timestamps.append(c.end_seconds)
                self._log.info(
                    "vad chunked %.1fs into %d region(s)", duration, len(chunks)
                )

        kwargs: dict[str, Any] = {
            "task": "translate" if task == "translate" else "transcribe",
            "language": source_lang,
            "word_timestamps": with_timestamps,
            "vad_filter": False,
        }
        if clip_timestamps is not None:
            kwargs["clip_timestamps"] = clip_timestamps

        segments_iter, info = model.transcribe(audio_path, **kwargs)
        segments_list = list(segments_iter)

        segments_out: list[dict] = []
        words_out: list[dict] = []
        for idx, seg in enumerate(segments_list):
            segments_out.append(
                {
                    "id": idx,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text,
                    "tokens": list(seg.tokens) if getattr(seg, "tokens", None) else [],
                    "avg_logprob": _f_or_none(getattr(seg, "avg_logprob", None)),
                    "compression_ratio": _f_or_none(getattr(seg, "compression_ratio", None)),
                    "no_speech_prob": _f_or_none(getattr(seg, "no_speech_prob", None)),
                    "temperature": _f_or_none(getattr(seg, "temperature", None)) or 0.0,
                }
            )
            if with_timestamps and getattr(seg, "words", None):
                for w in seg.words:
                    words_out.append(
                        {
                            "word": w.word,
                            "start": float(w.start),
                            "end": float(w.end),
                        }
                    )

        text = "".join(s["text"] for s in segments_out).strip()
        return TranscribeResult(
            text=text,
            segments=segments_out,
            words=words_out,
            language=getattr(info, "language", None) or source_lang,
            duration=getattr(info, "duration", None) or duration,
            supports_timestamps=True,
        )

    async def unload(self) -> None:
        async with self._lock:
            if self._model is None:
                return
            self._log.info("unloading %s", self.repo)
            model = self._model
            self._model = None
            self._last_used = None
        # faster-whisper wraps ctranslate2 — drop the inner CT2 model
        # explicitly so its GPU buffers are released before gc.
        try:
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "unload_model"):
                inner.unload_model(to_cpu=False)
        except Exception:
            self._log.exception("ct2 unload_model failed for %s", self.repo)
        del model
        gc.collect()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        self._log.info("unloaded %s", self.repo)


def _f_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
