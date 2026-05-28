"""SALM backend — canary-qwen-2.5b (NeMo speechlm2.SALM, ASR via prompt).

Unlike `EncDecMultiTaskModel`, SALM has no `.transcribe()`. Use `.generate()`
with a chat-style prompt:

    prompts=[[{
        "role": "user",
        "content": f"Transcribe the following: {model.audio_locator_tag}",
        "audio": [audio_path],
    }]]

Returns decoded transcript via `model.tokenizer.ids_to_text(...)`. Plain text,
not chat-format JSON.
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
from .base import TranscribeResult


_PROMPT_PREFIX = "Transcribe the following:"


class SalmBackend:
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
        self._log = logging.getLogger(f"talkies.salm.{model_id}")

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
            self._log.info("loading SALM %s onto %s", self.repo, self._device)
            self._model = await asyncio.to_thread(self._load_sync)
            self._log.info("loaded SALM %s", self.repo)
            return self._model

    def _load_sync(self) -> Any:
        import torch

        from nemo.collections.speechlm2.models import SALM
        from nemo.collections.speechlm2.models import salm as _salm_mod

        # canary-qwen-2.5b ships HF-native (config.json + model.safetensors,
        # no .nemo archive) because SALM wraps a Qwen3 LLM under the hood.
        # from_pretrained accepts a local dir → no network when offline.
        #
        # nemo_toolkit 2.7.3's `load_pretrained_hf` hard-codes dtype=fp32
        # when it calls AutoModelForCausalLM.from_pretrained for the Qwen3
        # backbone. SALM.__init__ doesn't forward a dtype either. Net result
        # on a 12 GB card: ~6.8 GB just for the 1.7B Qwen weights in fp32,
        # plus the speech encoder → OOM during `.to(cuda)`. The model config
        # itself declares `torch_dtype: bfloat16` for the SALM safetensors,
        # so it's safe (and was the training-time precision) to load the
        # LLM backbone in bf16 too.
        #
        # SALM imports the helper by name (`from ...pretrained import
        # load_pretrained_hf`), so we have to patch the binding inside the
        # `nemo.collections.speechlm2.models.salm` module namespace, not on
        # the source module — patching the source module would leave SALM
        # holding the stale reference. Restored in finally so future
        # callers aren't affected.
        _orig = _salm_mod.load_pretrained_hf

        def _bf16_load(
            model_path: str,
            pretrained_weights: bool = True,
            dtype: Any = torch.bfloat16,
        ) -> Any:
            return _orig(
                model_path,
                pretrained_weights=pretrained_weights,
                dtype=dtype,
            )

        self._log.info(
            "loading SALM %s from %s", self.model_id, self.model_path
        )
        _salm_mod.load_pretrained_hf = _bf16_load
        try:
            model = SALM.from_pretrained(
                str(self.model_path), map_location="cpu"
            )
        finally:
            _salm_mod.load_pretrained_hf = _orig
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
        del source_lang, target_lang, task, with_timestamps  # SALM is text-only
        model = await self.get_model()
        async with self._lock:
            text, duration = await asyncio.to_thread(
                self._transcribe_dispatch, model, audio_path
            )
            self._last_used = time.monotonic()
            return TranscribeResult(
                text=text, duration=duration, supports_timestamps=False
            )

    def _transcribe_dispatch(self, model: Any, audio_path: str) -> tuple[str, float]:
        """One-shot when audio fits, VAD-chunked when it doesn't.

        SALM concatenates the audio into the Qwen3 LLM context. A 13-minute
        clip at 16kHz blows past the kv cache and OOMs the GPU. Falling
        through to the same VAD chunker the other backends use keeps each
        SALM forward pass bounded by VAD_MAX_SPEECH_SECONDS — text is
        concatenated across chunks (no timestamps; SALM has no alignment
        head anyway).
        """
        audio = vad_mod.load_wav_16k_mono(audio_path)
        duration = audio.shape[0] / vad_mod.SAMPLE_RATE

        if duration <= config.VAD_CHUNK_THRESHOLD_SECONDS:
            return self._generate_sync(model, audio_path), duration

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
            return "", duration

        self._log.info(
            "vad chunked %.1fs into %d region(s) for %s",
            duration,
            len(chunks),
            self.model_id,
        )

        tmpdir = tempfile.mkdtemp(prefix="talkies-salm-")
        parts: list[str] = []
        try:
            for i, region in enumerate(chunks):
                chunk_path = os.path.join(tmpdir, f"chunk-{i:04d}.wav")
                vad_mod.write_chunk_wav(audio, region, chunk_path)
                parts.append(self._generate_sync(model, chunk_path))
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

        joined = " ".join(p for p in parts if p).strip()
        return joined, duration

    def _generate_sync(self, model: Any, audio_path: str) -> str:
        audio_tag = getattr(model, "audio_locator_tag", "<audio>")
        prompt = [
            [
                {
                    "role": "user",
                    "content": f"{_PROMPT_PREFIX} {audio_tag}",
                    "audio": [audio_path],
                }
            ]
        ]
        tokens = model.generate(prompts=prompt, max_new_tokens=512)
        ids = tokens[0].tolist() if hasattr(tokens[0], "tolist") else list(tokens[0])
        text = model.tokenizer.ids_to_text(ids)
        if not isinstance(text, str):
            text = str(text)
        return text.strip()

    async def unload(self) -> None:
        async with self._lock:
            if self._model is None:
                return
            self._log.info("unloading SALM %s", self.repo)
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
        self._log.info("unloaded SALM %s", self.repo)
