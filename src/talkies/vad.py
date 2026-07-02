"""Silero-VAD speech region detection + chunk merging.

Two-stage pipeline (mirrors speaches):

  1. `detect_speech_regions(audio)` runs silero-vad over a 16 kHz mono PCM
     stream and returns raw speech spans (samples).
  2. `merge_speech_regions(regions, ...)` glues adjacent spans up to
     `max_speech_samples` so each merged chunk fits inside one model
     forward pass (~30s for whisper, similar for canary/parakeet).

The whisper backend feeds the merged chunks straight to faster-whisper as
`clip_timestamps=[start_s, end_s, ...]` — faster-whisper handles slicing
internally and returns timestamps on the absolute timeline.

The NeMo backends (canary multitask, parakeet) don't accept clip_timestamps;
they need actual sliced WAVs and the wrapper adds the chunk-start offset
to every returned segment/word.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
_WINDOW_SIZE_SAMPLES = 512  # silero-vad v5 requires exactly 512 at 16 kHz

log = logging.getLogger("talkies.vad")


@dataclass
class SpeechRegion:
    """Half-open [start, end) range in samples at 16 kHz."""

    start: int
    end: int

    @property
    def start_seconds(self) -> float:
        return self.start / SAMPLE_RATE

    @property
    def end_seconds(self) -> float:
        return self.end / SAMPLE_RATE

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start) / SAMPLE_RATE


class _SileroVAD:
    """Lazy-loaded silero-vad ONNX wrapper. One instance per process."""

    _instance: "_SileroVAD | None" = None

    def __init__(self) -> None:
        import onnxruntime as ort  # type: ignore[import-not-found]

        model_path = self._resolve_model_path()
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        log.info("silero-vad loaded from %s", model_path)

    @staticmethod
    def _resolve_model_path() -> str:
        """Find silero-vad ONNX in the silero-vad python package."""
        import importlib.util

        spec = importlib.util.find_spec("silero_vad")
        if spec is None or spec.origin is None:
            raise RuntimeError("silero_vad package not installed")
        from pathlib import Path

        pkg_dir = Path(spec.origin).parent
        for candidate in ("data/silero_vad.onnx", "silero_vad.onnx"):
            p = pkg_dir / candidate
            if p.is_file():
                return str(p)
        raise FileNotFoundError(f"silero_vad.onnx not found under {pkg_dir}")

    @classmethod
    def get(cls) -> "_SileroVAD":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        """Return one speech-probability per 512-sample window."""
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)

        n_windows = audio.shape[0] // _WINDOW_SIZE_SAMPLES
        probs = np.empty(n_windows, dtype=np.float32)
        for i in range(n_windows):
            start = i * _WINDOW_SIZE_SAMPLES
            window = audio[start : start + _WINDOW_SIZE_SAMPLES].astype(np.float32)
            out, state = self.session.run(
                None,
                {"input": window[np.newaxis, :], "sr": sr, "state": state},
            )
            probs[i] = float(out[0][0])
        return probs


def detect_speech_regions(
    audio: np.ndarray,
    *,
    threshold: float = 0.5,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 200,
) -> list[SpeechRegion]:
    """Return contiguous speech regions over a 16 kHz mono float32 audio array.

    Adjacent windows with prob >= threshold form regions. Regions separated by
    < min_silence_ms of low-prob windows are joined. Each region is padded by
    speech_pad_ms on both ends (clipped to audio bounds).
    """
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")

    if audio.shape[0] < _WINDOW_SIZE_SAMPLES:
        return []

    # Pad to a window multiple so the last partial window is included.
    pad = (-audio.shape[0]) % _WINDOW_SIZE_SAMPLES
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])

    probs = _SileroVAD.get().probabilities(audio)
    speech_mask = probs >= threshold

    min_silence_windows = max(
        1, (min_silence_ms * SAMPLE_RATE) // (1000 * _WINDOW_SIZE_SAMPLES)
    )
    pad_samples = (speech_pad_ms * SAMPLE_RATE) // 1000

    regions: list[SpeechRegion] = []
    in_speech = False
    region_start = 0
    silence_run = 0

    for i, is_speech in enumerate(speech_mask):
        sample_pos = i * _WINDOW_SIZE_SAMPLES
        if is_speech:
            if not in_speech:
                in_speech = True
                region_start = sample_pos
            silence_run = 0
        else:
            if not in_speech:
                continue
            silence_run += 1
            if silence_run < min_silence_windows:
                continue
            region_end = sample_pos - silence_run * _WINDOW_SIZE_SAMPLES
            regions.append(
                _pad_clip(region_start, region_end, audio.shape[0], pad_samples)
            )
            in_speech = False
            silence_run = 0

    if in_speech:
        region_end = speech_mask.shape[0] * _WINDOW_SIZE_SAMPLES
        regions.append(_pad_clip(region_start, region_end, audio.shape[0], pad_samples))

    return regions


def _pad_clip(start: int, end: int, audio_len: int, pad: int) -> SpeechRegion:
    return SpeechRegion(
        start=max(0, start - pad),
        end=min(audio_len, end + pad),
    )


def merge_speech_regions(
    regions: list[SpeechRegion],
    *,
    max_speech_seconds: float = 28.0,
) -> list[SpeechRegion]:
    """Greedily concatenate adjacent regions until adding the next would
    exceed max_speech_seconds. Yields the chunk list fed to backends.

    If a single region exceeds the cap, emit it as-is — backends must accept
    it and clip internally. (Whisper accepts up to its own context limit;
    NeMo Canary/Parakeet may truncate.)
    """
    if not regions:
        return []
    max_samples = int(max_speech_seconds * SAMPLE_RATE)
    out: list[SpeechRegion] = []
    cur = SpeechRegion(start=regions[0].start, end=regions[0].end)
    for r in regions[1:]:
        if (r.end - cur.start) <= max_samples:
            cur = SpeechRegion(start=cur.start, end=r.end)
            continue
        out.append(cur)
        cur = SpeechRegion(start=r.start, end=r.end)
    out.append(cur)
    return out


def load_wav_16k_mono(wav_path: str) -> np.ndarray:
    """Read a 16 kHz mono PCM-16 WAV into a float32 [-1, 1] numpy array.

    The wrapper guarantees this shape via ffmpeg preprocessing — we don't
    handle arbitrary inputs here.
    """
    import wave

    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"expected mono, got {wf.getnchannels()} channels")
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz, got {wf.getframerate()}")
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit")
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def write_chunk_wav(audio: np.ndarray, region: SpeechRegion, out_path: str) -> None:
    """Write [region.start, region.end) of `audio` as a 16 kHz mono PCM WAV."""
    import wave

    slice_ = audio[region.start : region.end]
    pcm16 = np.clip(slice_ * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def _validate_segments(segments: list[dict]) -> list[dict]:
    """Drop malformed segments (missing start/end) so callers don't crash."""
    out: list[dict] = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        if "start" not in s or "end" not in s:
            continue
        out.append(s)
    return out


def offset_segments(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Return a copy of `segments` with start/end shifted by `offset_seconds`."""
    out: list[dict] = []
    for s in _validate_segments(segments):
        out.append(
            {
                **s,
                "start": float(s["start"]) + offset_seconds,
                "end": float(s["end"]) + offset_seconds,
            }
        )
    return out


def offset_words(words: list[dict], offset_seconds: float) -> list[dict]:
    out: list[dict] = []
    for w in words:
        if not isinstance(w, dict) or "start" not in w or "end" not in w:
            continue
        out.append(
            {
                **w,
                "start": float(w["start"]) + offset_seconds,
                "end": float(w["end"]) + offset_seconds,
            }
        )
    return out


def stitch_results(
    parts: list[tuple[float, Any]],
) -> tuple[str, list[dict], list[dict]]:
    """Combine per-chunk (offset_seconds, TranscribeResult) into one transcript.

    Returns (combined_text, combined_segments, combined_words). Segments are
    renumbered 0..N-1 across the full timeline.
    """
    texts: list[str] = []
    all_segments: list[dict] = []
    all_words: list[dict] = []
    for offset, result in parts:
        if result.text:
            texts.append(result.text.strip())
        all_segments.extend(offset_segments(result.segments, offset))
        all_words.extend(offset_words(result.words, offset))
    for idx, seg in enumerate(all_segments):
        seg["id"] = idx
    return " ".join(t for t in texts if t), all_segments, all_words
