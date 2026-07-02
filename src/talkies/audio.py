"""Audio upload preprocessing — convert any container/codec to 16kHz mono WAV.

NeMo Canary's `.transcribe()` accepts file paths (any format soundfile/librosa
recognise) plus numpy arrays. The safe lowest-common-denominator is 16kHz mono
WAV on disk. Conversion via ffmpeg subprocess — librosa-soxr is faster but
ffmpeg covers a broader codec matrix (webm/m4a/opus/etc.) without extra deps.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile


class AudioConversionError(Exception):
    pass


class NotStereoError(AudioConversionError):
    """Raised when channel-split was requested on a non-stereo source."""


def _write_temp_input(raw_bytes: bytes, original_filename: str) -> str:
    if not raw_bytes:
        raise AudioConversionError("upload is empty")
    suffix = ""
    if "." in original_filename:
        ext = original_filename.rsplit(".", 1)[-1].lower()
        if ext and len(ext) <= 8:
            suffix = "." + ext
    in_fd, in_path = tempfile.mkstemp(prefix="talkies-in-", suffix=suffix)
    try:
        with os.fdopen(in_fd, "wb") as fh:
            fh.write(raw_bytes)
    except Exception:
        os.unlink(in_path)
        raise
    return in_path


def _probe_channels(in_path: str) -> int:
    """Return channel count of `in_path` (>=1). Raises AudioConversionError
    on failure. Uses ffprobe (ffmpeg's metadata tool) which is bundled with
    every ffmpeg install we ship.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "json",
        in_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioConversionError(f"ffprobe failed: {stderr or 'unknown error'}")
    try:
        meta = json.loads(proc.stdout.decode("utf-8", errors="replace"))
        streams = meta.get("streams") or []
        if not streams:
            raise AudioConversionError("ffprobe: no audio stream in upload")
        ch = int(streams[0].get("channels", 0))
        if ch < 1:
            raise AudioConversionError("ffprobe: invalid channel count")
        return ch
    except (ValueError, KeyError) as exc:
        raise AudioConversionError(f"ffprobe: bad metadata: {exc}") from exc


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, timeout=600)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioConversionError(f"ffmpeg failed: {stderr or 'unknown error'}")


def to_wav_16k_mono(raw_bytes: bytes, original_filename: str) -> str:
    """Write `raw_bytes` to a temp file, convert to 16kHz mono WAV, return WAV path.

    Caller is responsible for deleting the returned WAV.
    """
    in_path = _write_temp_input(raw_bytes, original_filename)
    out_fd, out_path = tempfile.mkstemp(prefix="talkies-out-", suffix=".wav")
    os.close(out_fd)
    try:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                in_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                out_path,
            ]
        )
    except Exception:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise
    finally:
        try:
            os.unlink(in_path)
        except OSError:
            pass
    return out_path


def to_wav_16k_split_lr(raw_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """Split a stereo upload into two 16kHz-mono WAVs (L, R) and return their paths.

    Raises NotStereoError when the source is not exactly 2 channels — caller
    converts that into HTTP 400. Caller is responsible for deleting both
    returned WAVs.
    """
    in_path = _write_temp_input(raw_bytes, original_filename)
    try:
        channels = _probe_channels(in_path)
        if channels != 2:
            raise NotStereoError(
                f"diarization=true requires a stereo (2-channel) source; "
                f"got {channels} channel(s). Channel-split diarization splits "
                f"L/R into per-speaker streams — it can't run on mono."
            )
    except Exception:
        try:
            os.unlink(in_path)
        except OSError:
            pass
        raise

    l_fd, l_path = tempfile.mkstemp(prefix="talkies-out-L-", suffix=".wav")
    os.close(l_fd)
    r_fd, r_path = tempfile.mkstemp(prefix="talkies-out-R-", suffix=".wav")
    os.close(r_fd)
    try:
        # Single ffmpeg invocation produces both mono outputs via
        # `pan` filter: front-left -> L file, front-right -> R file.
        _run_ffmpeg(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                in_path,
                "-vn",
                "-filter_complex",
                "[0:a]channelsplit=channel_layout=stereo[L][R]",
                "-map",
                "[L]",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                l_path,
                "-map",
                "[R]",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                r_path,
            ]
        )
    except Exception:
        for p in (l_path, r_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        raise
    finally:
        try:
            os.unlink(in_path)
        except OSError:
            pass
    return l_path, r_path
