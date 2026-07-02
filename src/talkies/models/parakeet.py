"""Parakeet backend (NeMo RNNT — parakeet-tdt-* family).

Unlike speaches, we plumb `with_timestamps=True` through to NeMo's
`transcribe()` and map the resulting Hypothesis.timestamp dict to the
OpenAI Whisper-shape segments + words.

For audio longer than VAD_CHUNK_THRESHOLD_SECONDS we VAD-chunk, transcribe
each chunk, and offset its segments/words by the chunk start.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .. import config
from .. import vad as vad_mod
from ._nemo_paths import find_nemo_file
from .base import TranscribeResult


class ParakeetBackend:
    def __init__(self, model_id: str, repo: str, model_path: Path, device: str) -> None:
        self.model_id = model_id
        self.repo = repo
        self.model_path = model_path
        self._device = device
        self._lock = asyncio.Lock()
        self._model: Any = None
        self._last_used: float | None = None
        self._log = logging.getLogger(f"talkies.parakeet.{model_id}")

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
        from nemo.collections.asr.models import ASRModel

        nemo_path = find_nemo_file(self.model_path)
        self._log.info("loading %s from %s", self.model_id, nemo_path)
        model = ASRModel.restore_from(nemo_path, map_location="cpu")
        model = model.to(self._device).eval()
        return model

    async def transcribe(
        self,
        audio_path: str,
        *,
        source_lang: str | None,
        target_lang: str | None,
        task: str,
        with_timestamps: bool = False,
    ) -> TranscribeResult:
        del target_lang, task  # parakeet is monolingual ASR-only
        model = await self.get_model()
        async with self._lock:
            result = await asyncio.to_thread(
                self._transcribe_sync, model, audio_path, source_lang, with_timestamps
            )
            self._last_used = time.monotonic()
            return result

    def _transcribe_sync(
        self,
        model: Any,
        audio_path: str,
        source_lang: str | None,
        with_timestamps: bool,
    ) -> TranscribeResult:
        audio = vad_mod.load_wav_16k_mono(audio_path)
        duration = audio.shape[0] / vad_mod.SAMPLE_RATE

        if duration <= config.VAD_CHUNK_THRESHOLD_SECONDS:
            res = self._transcribe_path(model, audio_path, with_timestamps)
            return TranscribeResult(
                text=res.text,
                segments=res.segments,
                words=res.words,
                language=source_lang or "en",
                duration=duration,
                supports_timestamps=True,
            )

        regions = vad_mod.detect_speech_regions(
            audio,
            threshold=config.VAD_THRESHOLD,
            min_silence_ms=config.VAD_MIN_SILENCE_MS,
            speech_pad_ms=config.VAD_SPEECH_PAD_MS,
        )
        chunks = vad_mod.merge_speech_regions(
            regions, max_speech_seconds=config.VAD_MAX_SPEECH_SECONDS
        )
        if not chunks:
            return TranscribeResult(
                text="",
                language=source_lang or "en",
                duration=duration,
                supports_timestamps=True,
            )

        self._log.info(
            "vad chunked %.1fs into %d region(s) for %s",
            duration,
            len(chunks),
            self.model_id,
        )

        tmpdir = tempfile.mkdtemp(prefix="talkies-parakeet-")
        parts: list[tuple[float, TranscribeResult]] = []
        try:
            for i, region in enumerate(chunks):
                chunk_path = os.path.join(tmpdir, f"chunk-{i:04d}.wav")
                vad_mod.write_chunk_wav(audio, region, chunk_path)
                res = self._transcribe_path(model, chunk_path, with_timestamps)
                parts.append((region.start_seconds, res))
        finally:
            for f in os.listdir(tmpdir):
                try:
                    os.unlink(os.path.join(tmpdir, f))
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

        text, segments, words = vad_mod.stitch_results(parts)
        return TranscribeResult(
            text=text,
            segments=segments,
            words=words,
            language=source_lang or "en",
            duration=duration,
            supports_timestamps=True,
        )

    def _transcribe_path(
        self, model: Any, audio_path: str, with_timestamps: bool
    ) -> TranscribeResult:
        kwargs: dict[str, Any] = {"audio": [audio_path], "batch_size": 1}
        if with_timestamps:
            kwargs["timestamps"] = True

        results = model.transcribe(**kwargs)
        if not results:
            return TranscribeResult(text="", supports_timestamps=True)
        first = results[0]
        if isinstance(first, str):
            return TranscribeResult(text=first, supports_timestamps=True)

        text_attr = getattr(first, "text", None)
        text = text_attr if isinstance(text_attr, str) else str(first)

        segments: list[dict] = []
        words: list[dict] = []
        if with_timestamps:
            ts = getattr(first, "timestamp", None)
            if isinstance(ts, dict):
                segments = _segments_from_nemo(ts.get("segment", []))
                words = _words_from_nemo(ts.get("word", []))

        return TranscribeResult(
            text=text,
            segments=segments,
            words=words,
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
        try:
            model.cpu()
        except Exception:
            self._log.exception("model.cpu() failed for %s", self.repo)
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


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _segments_from_nemo(raw: Any) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = item.get("segment") or item.get("text") or ""
        start = _coerce_float(item.get("start"))
        end = _coerce_float(item.get("end"))
        if start is None or end is None:
            continue
        out.append({"id": idx, "start": start, "end": end, "text": str(text).strip()})
    return out


def _words_from_nemo(raw: Any) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        word = item.get("word") or item.get("text") or ""
        start = _coerce_float(item.get("start"))
        end = _coerce_float(item.get("end"))
        if start is None or end is None:
            continue
        out.append({"word": str(word).strip(), "start": start, "end": end})
    return out
