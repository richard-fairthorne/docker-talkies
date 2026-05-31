"""Qwen3-TTS backend (Qwen/Qwen3-TTS-12Hz-*, Apache-2.0).

CUDA-only TTS via ``faster_qwen3_tts.FasterQwen3TTS``. Three operational
modes, picked per-model via the ``qwen3_mode`` field in ``models.json``:

* ``base`` (default) — voice cloning from a reference WAV under
  ``$BUILTIN_VOICES_DIR`` (baked-in samples) or ``$CUSTOM_VOICES_DIR``
  (user-mounted). The ``voice`` field on /v1/audio/speech is the path of
  the WAV relative to its parent dir with ``.wav`` stripped. Optional
  sibling ``<name>.txt`` (reference transcript — required for ICL
  cloning; absent triggers x-vector-only fallback with a warning) and
  ``<name>.lang`` (language label, default "English").
* ``custom_voice`` — 9 preset speakers, no reference WAV needed. The
  ``voice`` field is the speaker name (Vivian / Serena / Uncle_Fu / Dylan
  / Eric / Ryan / Aiden / Ono_Anna / Sohee). The OpenAI ``instructions``
  field carries emotion / style cues ("Speak angrily"). 1.7B honours it;
  0.6B silently ignores it (upstream limitation).
* ``voice_design`` — single-model NL voice description. The ``voice``
  field is ignored (sentinel ``"design"``); the OpenAI ``instructions``
  field carries the voice description ("A warm, friendly young female
  voice with a cheerful tone"). Empty ``instructions`` returns 400.

The first generation after load captures CUDA graphs (~30-60 s on a
mid-range GPU); subsequent generations are sub-second.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .. import config
from .base import SynthesisResult


DEFAULT_LANGUAGE = "English"

# Sibling-file conventions next to ``<name>.wav`` for ``base`` mode.
_REF_TEXT_EXT = ".txt"
_LANG_EXT = ".lang"

# Mode constants — kept in sync with faster_qwen3_tts.model.tts_model_type
# strings ("base" / "custom_voice" / "voice_design").
MODE_BASE = "base"
MODE_CUSTOM_VOICE = "custom_voice"
MODE_VOICE_DESIGN = "voice_design"
_VALID_MODES = (MODE_BASE, MODE_CUSTOM_VOICE, MODE_VOICE_DESIGN)

# Hardcoded preset speakers for ``custom_voice`` mode — must match
# ``faster_qwen3_tts.model.get_supported_speakers()`` for the Qwen3-TTS
# CustomVoice checkpoint. Hardcoded so /v1/audio/voices works without
# loading the ~3 GB checkpoint just to list strings.
_CUSTOM_VOICE_SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)

# Sentinel voice for ``voice_design`` mode — the API has no preset voices;
# this single sentinel makes the standard route-layer voice validation
# pass without a backend-mode branch in server.py.
_VOICE_DESIGN_SENTINEL = "design"


class Qwen3TTSBackend:
    def __init__(
        self,
        model_id: str,
        repo: str,
        model_path: Path,
        device: str,
        mode: str = MODE_BASE,
        default_language: str = DEFAULT_LANGUAGE,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"qwen3_tts mode={mode!r} must be one of {_VALID_MODES}"
            )
        self.model_id = model_id
        self.repo = repo
        self.model_path = model_path
        # Device check is deferred to load time — same pattern as the other
        # CUDA-only backends. Constructing on CPU is fine; the first request
        # triggers _load_sync which surfaces a clear RuntimeError.
        self._device = device
        self._mode = mode
        self._default_language = default_language or DEFAULT_LANGUAGE
        self._lock = asyncio.Lock()
        self._model: Any = None
        self._last_used: float | None = None
        self._log = logging.getLogger(f"talkies.qwen3_tts.{model_id}")

    @property
    def mode(self) -> str:
        return self._mode

    def loaded(self) -> bool:
        return self._model is not None

    def last_used_secs_ago(self) -> float | None:
        if self._last_used is None:
            return None
        return time.monotonic() - self._last_used

    # ── voice catalog dispatch ───────────────────────────────────────────

    def voices(self) -> list[str]:
        if self._mode == MODE_CUSTOM_VOICE:
            return list(_CUSTOM_VOICE_SPEAKERS)
        if self._mode == MODE_VOICE_DESIGN:
            return [_VOICE_DESIGN_SENTINEL]
        return sorted(self._scan_voices().keys())

    def voice_origins(self) -> dict[str, str]:
        if self._mode == MODE_CUSTOM_VOICE:
            return {name: "builtin" for name in _CUSTOM_VOICE_SPEAKERS}
        if self._mode == MODE_VOICE_DESIGN:
            return {_VOICE_DESIGN_SENTINEL: "builtin"}

        out: dict[str, str] = {}
        builtin_resolved: Path | None = None
        custom_resolved: Path | None = None
        try:
            if config.BUILTIN_VOICES_DIR.is_dir():
                builtin_resolved = config.BUILTIN_VOICES_DIR.resolve(strict=True)
        except (OSError, RuntimeError):
            builtin_resolved = None
        try:
            if config.CUSTOM_VOICES_DIR.is_dir():
                custom_resolved = config.CUSTOM_VOICES_DIR.resolve(strict=True)
        except (OSError, RuntimeError):
            custom_resolved = None
        for name, wav in self._scan_voices().items():
            try:
                resolved = wav.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if custom_resolved is not None and resolved.is_relative_to(custom_resolved):
                out[name] = "custom"
                continue
            if builtin_resolved is not None and resolved.is_relative_to(builtin_resolved):
                out[name] = "builtin"
                continue
            out[name] = "builtin"
        return out

    def default_voice(self) -> str:
        if self._mode == MODE_CUSTOM_VOICE:
            return _CUSTOM_VOICE_SPEAKERS[0]  # Vivian
        if self._mode == MODE_VOICE_DESIGN:
            return _VOICE_DESIGN_SENTINEL
        catalog = self._scan_voices()
        if "alloy" in catalog:
            return "alloy"
        if not catalog:
            raise RuntimeError(
                f"no qwen3-tts voices found under {config.BUILTIN_VOICES_DIR} "
                f"or {config.CUSTOM_VOICES_DIR}; drop a .wav into "
                f"{config.CUSTOM_VOICES_DIR}/ to enable voice cloning"
            )
        return sorted(catalog)[0]

    def _scan_voices(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for base in (config.BUILTIN_VOICES_DIR, config.CUSTOM_VOICES_DIR):
            if not base.is_dir():
                continue
            try:
                base_resolved = base.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            for wav in sorted(base.rglob("*.wav")):
                try:
                    resolved = wav.resolve(strict=True)
                except (OSError, RuntimeError):
                    continue
                if not resolved.is_relative_to(base_resolved):
                    self._log.warning(
                        "skipping voice wav %s — resolves outside %s",
                        wav,
                        base,
                    )
                    continue
                name = wav.relative_to(base).with_suffix("").as_posix()
                if not name or name.startswith("."):
                    continue
                out[name] = wav
        return out

    def _voice_config(self, wav_path: Path) -> dict[str, Any]:
        ref_text = ""
        language = self._default_language
        ref_text_path = wav_path.with_suffix(_REF_TEXT_EXT)
        lang_path = wav_path.with_suffix(_LANG_EXT)
        if ref_text_path.is_file():
            ref_text = ref_text_path.read_text(encoding="utf-8").strip()
        if lang_path.is_file():
            lang_raw = lang_path.read_text(encoding="utf-8").strip()
            if lang_raw:
                language = lang_raw
        return {
            "ref_audio": str(wav_path),
            "ref_text": ref_text,
            "language": language,
        }

    # ── load / unload ────────────────────────────────────────────────────

    async def get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            self._log.info(
                "loading %s (mode=%s) onto %s", self.repo, self._mode, self._device
            )
            self._model = await asyncio.to_thread(self._load_sync)
            loaded_type = getattr(
                getattr(self._model, "model", None), "model", None
            )
            loaded_type = getattr(loaded_type, "tts_model_type", None)
            if loaded_type is not None and loaded_type != self._mode:
                self._log.warning(
                    "registry says qwen3_mode=%s but loaded checkpoint reports "
                    "tts_model_type=%s — synthesis will follow the registry mode",
                    self._mode,
                    loaded_type,
                )
            self._log.info("loaded %s", self.repo)
            return self._model

    def _load_sync(self) -> Any:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        if not self._device.startswith("cuda"):
            raise RuntimeError(
                f"qwen3_tts backend requires CUDA; got device={self._device!r}. "
                "Run the CUDA image with --gpus all, or exclude "
                f"{self.model_id!r} via TALKIES_ENABLED_MODELS."
            )
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"qwen3-tts snapshot missing at {self.model_path}"
            )
        return FasterQwen3TTS.from_pretrained(
            str(self.model_path),
            device=self._device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            max_seq_len=2048,
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz. 24000 is the Qwen3-TTS fixed rate."""
        if self._model is not None:
            return int(getattr(self._model, "sample_rate", 24000))
        return 24000

    # ── synthesis dispatch ───────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        instructions: str | None = None,
        language: str | None = None,
        sampling: dict | None = None,
    ) -> SynthesisResult:
        if not text.strip():
            raise ValueError("input text is empty")
        if speed != 1.0:
            self._log.debug(
                "qwen3_tts has no speed control — ignoring speed=%.2f", speed
            )
        lang = (language or self._default_language).strip() or DEFAULT_LANGUAGE
        sampling = sampling or {}

        if self._mode == MODE_BASE:
            cfg = self._resolve_base_voice(voice)
            model = await self.get_model()
            async with self._lock:
                result = await asyncio.to_thread(
                    self._synthesize_base_sync,
                    model, text, cfg, instructions, lang, sampling,
                )
                self._last_used = time.monotonic()
                return result

        if self._mode == MODE_CUSTOM_VOICE:
            if voice not in _CUSTOM_VOICE_SPEAKERS:
                raise ValueError(
                    f"unknown speaker {voice!r} for model {self.model_id!r}; "
                    f"available: {list(_CUSTOM_VOICE_SPEAKERS)}"
                )
            model = await self.get_model()
            async with self._lock:
                result = await asyncio.to_thread(
                    self._synthesize_custom_sync,
                    model, text, voice, lang, instructions, sampling,
                )
                self._last_used = time.monotonic()
                return result

        # voice_design
        if not instructions or not instructions.strip():
            raise ValueError(
                f"model {self.model_id!r} (voice_design) requires the "
                "`instructions` field to describe the voice "
                "(e.g. 'A warm, friendly young female voice')"
            )
        model = await self.get_model()
        async with self._lock:
            result = await asyncio.to_thread(
                self._synthesize_design_sync,
                model, text, instructions, lang, sampling,
            )
            self._last_used = time.monotonic()
            return result

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        instructions: str | None = None,
        language: str | None = None,
        sampling: dict | None = None,
        chunk_size: int = 8,
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            raise ValueError("input text is empty")
        if speed != 1.0:
            self._log.debug(
                "qwen3_tts has no speed control — ignoring speed=%.2f", speed
            )
        lang = (language or self._default_language).strip() or DEFAULT_LANGUAGE
        sampling_kwargs = self._sampling_kwargs(sampling)

        worker_kwargs: dict[str, Any]
        if self._mode == MODE_BASE:
            cfg = self._resolve_base_voice(voice)
            ref_text = cfg["ref_text"]
            x_vector_only = not ref_text
            if x_vector_only:
                self._log.warning(
                    "no reference transcript (.txt) found for voice %s — "
                    "falling back to x-vector-only mode (lower fidelity).",
                    cfg["ref_audio"],
                )
            worker_kwargs = {
                "method": "voice_clone",
                "text": text,
                "language": cfg["language"] or lang,
                "ref_audio": cfg["ref_audio"],
                "ref_text": ref_text,
                "xvec_only": x_vector_only,
                "instruct": instructions or None,
                "chunk_size": chunk_size,
                **sampling_kwargs,
            }
        elif self._mode == MODE_CUSTOM_VOICE:
            if voice not in _CUSTOM_VOICE_SPEAKERS:
                raise ValueError(
                    f"unknown speaker {voice!r} for model {self.model_id!r}; "
                    f"available: {list(_CUSTOM_VOICE_SPEAKERS)}"
                )
            worker_kwargs = {
                "method": "custom_voice",
                "text": text,
                "speaker": voice,
                "language": lang,
                "instruct": instructions or None,
                "chunk_size": chunk_size,
                **sampling_kwargs,
            }
        else:
            if not instructions or not instructions.strip():
                raise ValueError(
                    f"model {self.model_id!r} (voice_design) requires the "
                    "`instructions` field"
                )
            worker_kwargs = {
                "method": "voice_design",
                "text": text,
                "instruct": instructions,
                "language": lang,
                "chunk_size": chunk_size,
                **sampling_kwargs,
            }

        model = await self.get_model()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(maxsize=4)
        cancel = threading.Event()

        def _stream_worker() -> None:
            import numpy as np

            try:
                method = worker_kwargs.pop("method")
                if method == "voice_clone":
                    gen = model.generate_voice_clone_streaming(**worker_kwargs)
                elif method == "custom_voice":
                    gen = model.generate_custom_voice_streaming(**worker_kwargs)
                else:
                    gen = model.generate_voice_design_streaming(**worker_kwargs)

                for audio_chunk, _sr, _timing in gen:
                    if cancel.is_set():
                        break
                    chunk = audio_chunk.astype(np.float32, copy=False)
                    np.clip(chunk, -1.0, 1.0, out=chunk)
                    pcm = (chunk * 32767.0).astype(np.int16).tobytes()
                    fut = asyncio.run_coroutine_threadsafe(queue.put(pcm), loop)
                    fut.result()
                    if cancel.is_set():
                        break
            except Exception as exc:  # noqa: BLE001
                if not cancel.is_set():
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
                    return
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        async with self._lock:
            thread_task = asyncio.create_task(asyncio.to_thread(_stream_worker))
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item  # type: ignore[misc]
            finally:
                cancel.set()
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                await thread_task
                self._last_used = time.monotonic()

    # ── per-mode sync workers ────────────────────────────────────────────

    # Allowed sampling knobs passed through to faster-qwen3-tts. Names match
    # the upstream generate_* method signatures exactly so we can ** them in.
    _SAMPLING_KEYS = (
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "max_new_tokens",
        "do_sample",
    )

    def _sampling_kwargs(self, sampling: dict | None) -> dict[str, Any]:
        if not sampling:
            return {}
        return {k: sampling[k] for k in self._SAMPLING_KEYS if k in sampling}

    def _resolve_base_voice(self, voice: str) -> dict[str, Any]:
        catalog = self._scan_voices()
        wav_path = catalog.get(voice)
        if wav_path is None:
            raise ValueError(
                f"unknown voice {voice!r} for model {self.model_id!r}; "
                f"{len(catalog)} voice(s) available — call "
                "GET /v1/audio/voices to list them"
            )
        return self._voice_config(wav_path)

    def _synthesize_base_sync(
        self,
        model: Any,
        text: str,
        cfg: dict[str, Any],
        instructions: str | None,
        language: str,
        sampling: dict,
    ) -> SynthesisResult:
        ref_text = cfg["ref_text"]
        x_vector_only = not ref_text
        if x_vector_only:
            self._log.warning(
                "no reference transcript (.txt) found for voice %s — "
                "falling back to x-vector-only mode (lower fidelity).",
                cfg["ref_audio"],
            )
        # .lang sibling (if any) wins over the per-request language for
        # base mode — the reference transcript is in that language so the
        # model expects them to match.
        effective_language = cfg["language"] or language
        audio_arrays, sample_rate = model.generate_voice_clone(
            text=text,
            language=effective_language,
            ref_audio=cfg["ref_audio"],
            ref_text=ref_text,
            xvec_only=x_vector_only,
            instruct=instructions or None,
            **self._sampling_kwargs(sampling),
        )
        return self._pack_pcm(audio_arrays, sample_rate)

    def _synthesize_custom_sync(
        self,
        model: Any,
        text: str,
        speaker: str,
        language: str,
        instructions: str | None,
        sampling: dict,
    ) -> SynthesisResult:
        audio_arrays, sample_rate = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instructions or None,
            **self._sampling_kwargs(sampling),
        )
        return self._pack_pcm(audio_arrays, sample_rate)

    def _synthesize_design_sync(
        self,
        model: Any,
        text: str,
        instructions: str,
        language: str,
        sampling: dict,
    ) -> SynthesisResult:
        audio_arrays, sample_rate = model.generate_voice_design(
            text=text,
            instruct=instructions,
            language=language,
            **self._sampling_kwargs(sampling),
        )
        return self._pack_pcm(audio_arrays, sample_rate)

    def _pack_pcm(self, audio_arrays: list, sample_rate: int) -> SynthesisResult:
        import numpy as np

        if not audio_arrays:
            return SynthesisResult(pcm_int16=b"", sample_rate=int(sample_rate))
        full = np.concatenate(audio_arrays).astype(np.float32, copy=False)
        np.clip(full, -1.0, 1.0, out=full)
        int16 = (full * 32767.0).astype(np.int16)
        return SynthesisResult(pcm_int16=int16.tobytes(), sample_rate=int(sample_rate))

    async def unload(self) -> None:
        async with self._lock:
            if self._model is None:
                return
            self._log.info("unloading %s", self.repo)
            model = self._model
            self._model = None
            self._last_used = None
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
