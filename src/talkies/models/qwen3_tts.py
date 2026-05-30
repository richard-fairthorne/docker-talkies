"""Qwen3-TTS backend (Qwen/Qwen3-TTS-12Hz-0.6B-Base, Apache-2.0).

CUDA-only voice-cloning TTS via ``faster_qwen3_tts.FasterQwen3TTS``.
Voices are sourced from two on-disk dirs and merged into a single catalog:

* ``$BUILTIN_VOICES_DIR`` (``/opt/talkies/qwen3-voices``) — baked into the
  image at build time. A handful of curated samples to give the model
  something to clone out-of-the-box.
* ``$CUSTOM_VOICES_DIR`` (``/data/custom-voices``) — user-owned, mounted
  via the data volume. Drop ``foo/bar/me.wav`` in and the voice
  ``foo/bar/me`` shows up in ``GET /v1/audio/voices``.

Each ``<name>.wav`` may have sibling ``<name>.txt`` (reference transcript
for ICL voice cloning — required by the model; without it the backend
falls back to x-vector-only mode which is lower fidelity) and
``<name>.lang`` (language label string passed through to the model —
defaults to "English").

Custom voices shadow builtin voices that share the same name. Voice
names are the path of the wav relative to its parent voices dir with the
``.wav`` suffix stripped — nested subdirs are preserved (so a file at
``custom-voices/foo/bar/me.wav`` is voice ``foo/bar/me``). A
path-traversal guard refuses any wav whose resolved path escapes the
voices dir, so a hostile symlink can't read arbitrary host files as a
voice prompt.

The first ``generate_voice_clone`` call after load captures the CUDA
graphs (~30-60s on a mid-range GPU); subsequent generations are
sub-second.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from pathlib import Path
from typing import Any

from .. import config
from .base import SynthesisResult


DEFAULT_LANGUAGE = "English"

# Sibling-file conventions next to ``<name>.wav``.
_REF_TEXT_EXT = ".txt"
_LANG_EXT = ".lang"


class Qwen3TTSBackend:
    def __init__(
        self, model_id: str, repo: str, model_path: Path, device: str
    ) -> None:
        self.model_id = model_id
        self.repo = repo
        self.model_path = model_path
        # Device check is deferred to load time — same pattern as the other
        # CUDA-only ASR backends. Constructing on CPU is fine; the first
        # request triggers _load_sync which surfaces the fail-loud
        # ValueError from FasterQwen3TTS upstream.
        self._device = device
        self._lock = asyncio.Lock()
        self._model: Any = None
        self._last_used: float | None = None
        self._default_voice = "alloy"
        self._log = logging.getLogger(f"talkies.qwen3_tts.{model_id}")

    def loaded(self) -> bool:
        return self._model is not None

    def last_used_secs_ago(self) -> float | None:
        if self._last_used is None:
            return None
        return time.monotonic() - self._last_used

    def voices(self) -> list[str]:
        # Re-scan every call — custom-voices/ is a live host mount, users
        # drop new wavs in at runtime and expect them visible immediately.
        # Scanning two dirs of <100 wavs is negligible vs synth cost.
        return sorted(self._scan_voices().keys())

    def voice_origins(self) -> dict[str, str]:
        """Return ``{voice_name: "builtin" | "custom"}`` for this backend.

        Used by ``/v1/audio/voices`` to let API consumers tell baked-in
        sample voices from user-mounted clones at a glance.
        """
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
        catalog = self._scan_voices()
        if self._default_voice in catalog:
            return self._default_voice
        if not catalog:
            raise RuntimeError(
                f"no qwen3-tts voices found under {config.BUILTIN_VOICES_DIR} "
                f"or {config.CUSTOM_VOICES_DIR}; drop a .wav into "
                f"{config.CUSTOM_VOICES_DIR}/ to enable voice cloning"
            )
        return sorted(catalog)[0]

    def _scan_voices(self) -> dict[str, Path]:
        """Return ``{voice_name: absolute_wav_path}``. Custom shadows builtin."""
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
                # Path-traversal guard — a symlink under base could resolve
                # outside; refuse those so a hostile mount can't exfiltrate.
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
        """Read sibling .txt / .lang metadata for a voice wav."""
        ref_text = ""
        language = DEFAULT_LANGUAGE
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
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        if not self._device.startswith("cuda"):
            # Surface the CUDA-only constraint with a clearer message than
            # the raw ValueError from FasterQwen3TTS upstream.
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

    async def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        instructions: str | None = None,
    ) -> SynthesisResult:
        if not text.strip():
            raise ValueError("input text is empty")
        catalog = self._scan_voices()
        wav_path = catalog.get(voice)
        if wav_path is None:
            raise ValueError(
                f"unknown voice {voice!r} for model {self.model_id!r}; "
                f"{len(catalog)} voice(s) available — call "
                "GET /v1/audio/voices to list them"
            )
        cfg = self._voice_config(wav_path)
        # qwen3-tts has no speed control; log + ignore non-default speed so
        # callers don't get a 4xx for a parameter the wire format requires.
        if speed != 1.0:
            self._log.debug(
                "qwen3_tts has no speed control — ignoring speed=%.2f", speed
            )
        model = await self.get_model()
        async with self._lock:
            result = await asyncio.to_thread(
                self._synthesize_sync, model, text, cfg, instructions
            )
            self._last_used = time.monotonic()
            return result

    def _synthesize_sync(
        self, model: Any, text: str, cfg: dict[str, Any], instructions: str | None
    ) -> SynthesisResult:
        import numpy as np

        ref_text = cfg["ref_text"]
        x_vector_only = not ref_text
        if x_vector_only:
            self._log.warning(
                "no reference transcript (.txt) found for voice %s — "
                "falling back to x-vector-only mode (lower fidelity). "
                "Add a sibling .txt file with the spoken content of the "
                "reference audio to enable ICL cloning.",
                cfg["ref_audio"],
            )
        audio_arrays, sample_rate = model.generate_voice_clone(
            text=text,
            language=cfg["language"],
            ref_audio=cfg["ref_audio"],
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            instruct=instructions or None,
        )
        if not audio_arrays:
            return SynthesisResult(pcm_int16=b"", sample_rate=int(sample_rate))
        full = np.concatenate(audio_arrays).astype(np.float32, copy=False)
        # Defensive clamp — the speech tokenizer emits well-formed audio in
        # [-1, 1] but rare edge cases produce slight overshoot.
        np.clip(full, -1.0, 1.0, out=full)
        int16 = (full * 32767.0).astype(np.int16)
        return SynthesisResult(
            pcm_int16=int16.tobytes(), sample_rate=int(sample_rate)
        )

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
