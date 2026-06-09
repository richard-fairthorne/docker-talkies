"""parakeet.cpp ASR backend — ctypes wrapper around libparakeet.so.

Drives NVIDIA's Parakeet family + Nemotron-3.5-ASR via mudler/parakeet.cpp
(MIT). Runs on CPU by default with optional GPU offload through ggml's
backends (built-in to the .so when compiled with -DPARAKEET_GGML_*). The
shared library is loaded once per backend instance and held for the
backend's lifetime; the C-API itself is thread-safe per context, but we
serialize calls behind an asyncio.Lock for consistency with the other
backends.

Model format: GGUF, single file per model. Talkies' standard "flat per-slug
directory under /data/models/<slug>/" still applies — we look for the first
``*.gguf`` file in the snapshot dir (HF repos like ``mudler/parakeet-cpp-gguf``
ship multiple quant variants in one repo, but we configure the desired one
via the registry entry's ``gguf_file`` field).

Output: ``TranscribeResult`` with verbose-json-shaped ``words`` (start/end
seconds + confidence) plus the flat ``text``. ``segments`` is left empty —
parakeet.cpp doesn't expose them natively; the server's serialization
layer fills in null/empty values for the Whisper-only fields.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from .base import TranscribeResult


DEFAULT_LIB_PATH = "/opt/parakeet/lib/libparakeet.so"

# Nemotron-3.5-ASR is prompt-conditioned and echoes a language tag at the end
# of its transcript (e.g. " <en-us>", " <de-de>"). Strip it before surfacing
# to the caller — it isn't speech content and would break ASR-round-trip
# assertions in client tests. Anchored to end-of-string so legitimate angle-
# bracket content earlier in the transcript is preserved.
_LANG_TAG_RE = re.compile(r"\s*<[a-z]{2,3}(?:-[a-z]{2,4})?>\s*$", re.IGNORECASE)


def _strip_lang_tag(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()

# Decoder selector (matches parakeet_capi.h):
#   0 = default (transducer for TDT/RNNT/hybrid, CTC for standalone CTC)
#   1 = ctc
#   2 = tdt/rnnt
_DECODER_DEFAULT = 0

# Silence-gap threshold (seconds) for grouping consecutive words into a
# segment. The C-API gives us per-word start/end + frame_sec but no segment
# boundaries — Whisper's verbose_json shape requires a segments array, so
# we synthesize them by cutting whenever the gap between two consecutive
# words exceeds this threshold. 0.5 s lines up with the "natural pause"
# range NeMo's offline grouper uses (segment_gap_threshold = 6 * frame_sec
# ≈ 0.48 s at the default 0.08 s/frame stride).
_SEGMENT_GAP_THRESHOLD_S = 0.5


class _CAPI:
    """Lazy singleton: bind libparakeet.so once per process."""

    _instance: _CAPI | None = None

    def __init__(self, lib_path: str) -> None:
        self.lib = ctypes.CDLL(lib_path)
        c = self.lib

        c.parakeet_capi_abi_version.restype = ctypes.c_int
        c.parakeet_capi_abi_version.argtypes = []

        c.parakeet_capi_load.restype = ctypes.c_void_p
        c.parakeet_capi_load.argtypes = [ctypes.c_char_p]

        c.parakeet_capi_free.restype = None
        c.parakeet_capi_free.argtypes = [ctypes.c_void_p]

        c.parakeet_capi_transcribe_path_json.restype = ctypes.c_void_p
        c.parakeet_capi_transcribe_path_json.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ]

        # parakeet.cpp C-API v3 added the lang-aware non-JSON variant for
        # multilingual prompt-conditioned models (nemotron). v4 adds streaming
        # JSON entry points but no path_json_lang yet — so we use the JSON
        # path for auto-language requests (gets timestamps + confidence) and
        # the non-JSON lang path when the caller asks for an explicit locale
        # (trades timestamps for language selection on nemotron).
        c.parakeet_capi_transcribe_path_lang.restype = ctypes.c_void_p
        c.parakeet_capi_transcribe_path_lang.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ]

        c.parakeet_capi_free_string.restype = None
        c.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]

        c.parakeet_capi_last_error.restype = ctypes.c_char_p
        c.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]

        self.abi_version = int(c.parakeet_capi_abi_version())

    @classmethod
    def get(cls, lib_path: str | None = None) -> _CAPI:
        if cls._instance is None:
            path = lib_path or os.environ.get(
                "TALKIES_PARAKEET_CPP_LIB", DEFAULT_LIB_PATH
            )
            cls._instance = cls(path)
        return cls._instance


class ParakeetCppBackend:
    """Per-model handle backed by a parakeet_ctx C handle."""

    def __init__(
        self,
        model_id: str,
        repo: str,
        model_path: Path,
        device: str,
        gguf_file: str | None = None,
        default_lang: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.repo = repo
        self.model_path = model_path
        self._device = device  # parakeet.cpp auto-selects, not honored explicitly
        self._gguf_file = gguf_file
        self._default_lang = default_lang or "auto"
        self._lock = asyncio.Lock()
        self._ctx: int | None = None  # raw void* as Python int
        self._last_used: float | None = None
        self._log = logging.getLogger(f"talkies.parakeet_cpp.{model_id}")

    def loaded(self) -> bool:
        return self._ctx is not None

    def last_used_secs_ago(self) -> float | None:
        if self._last_used is None:
            return None
        return time.monotonic() - self._last_used

    # ── lifecycle ────────────────────────────────────────────────────────

    def _resolve_gguf(self) -> Path:
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"parakeet.cpp snapshot dir missing at {self.model_path}"
            )
        if self._gguf_file:
            candidate = self.model_path / self._gguf_file
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"parakeet.cpp gguf file missing at {candidate} "
                    f"(model {self.model_id!r} requires gguf_file={self._gguf_file!r})"
                )
            return candidate
        # Fallback: first *.gguf in dir, alphabetical.
        ggufs = sorted(self.model_path.rglob("*.gguf"))
        if not ggufs:
            raise FileNotFoundError(
                f"no *.gguf files under {self.model_path} for model {self.model_id!r}"
            )
        return ggufs[0]

    async def get_model(self) -> Any:
        if self._ctx is not None:
            return self._ctx
        async with self._lock:
            if self._ctx is not None:
                return self._ctx
            self._log.info("loading %s (parakeet.cpp)", self.repo)
            self._ctx = await asyncio.to_thread(self._load_sync)
            self._log.info("loaded %s", self.repo)
            return self._ctx

    def _load_sync(self) -> int:
        capi = _CAPI.get()
        path = self._resolve_gguf()
        self._log.info("opening gguf %s (libparakeet ABI v%d)", path, capi.abi_version)
        ctx_ptr = capi.lib.parakeet_capi_load(str(path).encode("utf-8"))
        if not ctx_ptr:
            raise RuntimeError(
                f"parakeet_capi_load failed for {path} — "
                "check the file is a valid GGUF and the architecture is supported"
            )
        return int(ctx_ptr)

    async def unload(self) -> None:
        async with self._lock:
            if self._ctx is None:
                return
            self._log.info("unloading %s", self.repo)
            ctx = self._ctx
            self._ctx = None
            self._last_used = None
        await asyncio.to_thread(_CAPI.get().lib.parakeet_capi_free, ctx)
        gc.collect()
        self._log.info("unloaded %s", self.repo)

    # ── inference ────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_path: str,
        *,
        source_lang: str | None,
        target_lang: str | None,  # noqa: ARG002 — parakeet.cpp doesn't translate
        task: str,                # noqa: ARG002 — only "asr" is meaningful here
        with_timestamps: bool = False,  # noqa: ARG002 — JSON path always returns them
    ) -> TranscribeResult:
        if task and task != "asr":
            raise ValueError(
                f"parakeet.cpp supports task='asr' only; got task={task!r}"
            )
        lang = (source_lang or self._default_lang or "auto").strip() or "auto"
        ctx = await self.get_model()
        async with self._lock:
            result = await asyncio.to_thread(
                self._transcribe_sync, ctx, audio_path, lang
            )
            self._last_used = time.monotonic()
            return result

    def _transcribe_sync(self, ctx: int, audio_path: str, lang: str) -> TranscribeResult:
        capi = _CAPI.get()
        # Use the lang-aware non-JSON path for explicit language selection on
        # prompt-conditioned models (nemotron). For models without prompt
        # conditioning the C-API ignores `target_lang` and behaves identically
        # to the non-lang path. We always go through the JSON path to get
        # per-word timestamps + confidence regardless of language.
        if lang and lang != "auto":
            # No path_json_lang exists in the published API (v3/v4): we get
            # plain text via *_path_lang and JSON via the non-lang JSON path
            # if we wanted timestamps with non-auto language. For nemotron with
            # explicit lang we trade timestamps for language correctness here.
            text_ptr = capi.lib.parakeet_capi_transcribe_path_lang(
                ctypes.c_void_p(ctx),
                audio_path.encode("utf-8"),
                _DECODER_DEFAULT,
                lang.encode("utf-8"),
            )
            if not text_ptr:
                err = (capi.lib.parakeet_capi_last_error(ctypes.c_void_p(ctx))
                       or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"parakeet_capi_transcribe_path_lang failed: {err}"
                )
            try:
                text = ctypes.string_at(text_ptr).decode("utf-8", errors="replace")
            finally:
                capi.lib.parakeet_capi_free_string(text_ptr)
            return TranscribeResult(
                text=_strip_lang_tag(text),
                segments=[],
                words=[],
                language=lang,
                duration=None,
                supports_timestamps=False,
            )

        # auto / unset language → JSON path with timestamps + confidence.
        json_ptr = capi.lib.parakeet_capi_transcribe_path_json(
            ctypes.c_void_p(ctx),
            audio_path.encode("utf-8"),
            _DECODER_DEFAULT,
        )
        if not json_ptr:
            err = (capi.lib.parakeet_capi_last_error(ctypes.c_void_p(ctx))
                   or b"").decode("utf-8", errors="replace")
            raise RuntimeError(
                f"parakeet_capi_transcribe_path_json failed: {err}"
            )
        try:
            doc = ctypes.string_at(json_ptr).decode("utf-8", errors="replace")
        finally:
            capi.lib.parakeet_capi_free_string(json_ptr)
        parsed = json.loads(doc)
        # Drop nemotron's trailing <lang> token from both the flat text and the
        # per-word list (it shows up as the last word entry too).
        raw_words = parsed.get("words", []) or []
        if raw_words and _LANG_TAG_RE.fullmatch((raw_words[-1].get("w") or "").strip()):
            raw_words = raw_words[:-1]
        words = [
            {
                "word": w.get("w", ""),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            }
            for w in raw_words
        ]
        return TranscribeResult(
            text=_strip_lang_tag(str(parsed.get("text", ""))),
            segments=_segments_from_words(words),
            words=words,
            language=None,  # parakeet.cpp doesn't echo back the detected lang
            duration=None,
            supports_timestamps=bool(words),
        )


def _segments_from_words(words: list[dict]) -> list[dict]:
    """Group ``words`` into Whisper-shape segments by silence gap.

    The C-API doesn't expose segment boundaries — parakeet.cpp's offline
    decoder only finalizes per-token / per-word units. To match OpenAI's
    verbose_json shape we synthesize segments here: walk the word list,
    open a new segment whenever the gap between the previous word's end
    and the next word's start exceeds ``_SEGMENT_GAP_THRESHOLD_S``.
    Whisper-only fields (``tokens``, ``avg_logprob``, ``no_speech_prob``,
    ``compression_ratio``, ``temperature``) are filled with null/empty
    values by the server's verbose_json serializer.
    """
    if not words:
        return []
    segments: list[dict] = []
    current_words: list[dict] = [words[0]]
    seg_start = float(words[0].get("start", 0.0))
    prev_end = float(words[0].get("end", 0.0))
    for w in words[1:]:
        start = float(w.get("start", 0.0))
        end = float(w.get("end", 0.0))
        if start - prev_end > _SEGMENT_GAP_THRESHOLD_S:
            segments.append(_pack_segment(len(segments), seg_start, prev_end, current_words))
            current_words = []
            seg_start = start
        current_words.append(w)
        prev_end = end
    if current_words:
        segments.append(_pack_segment(len(segments), seg_start, prev_end, current_words))
    return segments


def _pack_segment(seg_id: int, start: float, end: float, words: list[dict]) -> dict:
    text = " ".join((w.get("word") or "").strip() for w in words if (w.get("word") or "").strip())
    return {
        "id": seg_id,
        "start": start,
        "end": end,
        "text": text,
    }
