"""TTS output encoding — raw mono PCM int16 to the OpenAI ``response_format`` set.

Backends return raw mono PCM as ``SynthesisResult``. The server hands it
to ``encode_audio`` which transcodes via ffmpeg into the requested format
and surfaces the proper ``Content-Type`` for the HTTP response.

``pcm`` short-circuits — OpenAI's "pcm" format is the raw 16-bit signed
little-endian mono samples at the model's sample rate, no header. So we
just hand the bytes back as-is.
"""

from __future__ import annotations

import asyncio


class TTSEncodingError(Exception):
    pass


# Maps OpenAI response_format -> (ffmpeg muxer args, Content-Type).
# ``-f wav`` produces a standard RIFF/WAV header at the input sample rate.
# ``-f ogg`` is the standard opus container.
# ``-f adts`` is raw AAC frames (matches OpenAI's ``aac`` shape).
_FORMATS: dict[str, tuple[list[str], str]] = {
    "mp3":  (["-f", "mp3", "-acodec", "libmp3lame", "-b:a", "128k"], "audio/mpeg"),
    "opus": (["-f", "ogg", "-acodec", "libopus", "-b:a", "64k"], "audio/ogg"),
    "aac":  (["-f", "adts", "-acodec", "aac", "-b:a", "128k"], "audio/aac"),
    "flac": (["-f", "flac", "-acodec", "flac"], "audio/flac"),
    "wav":  (["-f", "wav", "-acodec", "pcm_s16le"], "audio/wav"),
}

SUPPORTED_FORMATS: tuple[str, ...] = (*_FORMATS.keys(), "pcm")


def content_type_for(fmt: str) -> str:
    """Return the Content-Type header value for an OpenAI response_format."""
    if fmt == "pcm":
        return "application/octet-stream"
    if fmt not in _FORMATS:
        raise TTSEncodingError(
            f"unsupported response_format {fmt!r}; supported: {SUPPORTED_FORMATS}"
        )
    return _FORMATS[fmt][1]


async def encode_audio(
    pcm_int16: bytes, sample_rate: int, fmt: str
) -> tuple[bytes, str]:
    """Encode raw mono PCM int16 into ``fmt``. Returns ``(bytes, content_type)``.

    Raises ``TTSEncodingError`` on unsupported format or ffmpeg failure.
    """
    if fmt == "pcm":
        return pcm_int16, "application/octet-stream"
    if fmt not in _FORMATS:
        raise TTSEncodingError(
            f"unsupported response_format {fmt!r}; supported: {SUPPORTED_FORMATS}"
        )
    if not pcm_int16:
        return b"", _FORMATS[fmt][1]

    args, content_type = _FORMATS[fmt]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        *args,
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(pcm_int16), timeout=300
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise TTSEncodingError(f"ffmpeg encode timed out for {fmt}") from exc

    if proc.returncode != 0:
        msg = stderr.decode("utf-8", errors="replace").strip()
        raise TTSEncodingError(
            f"ffmpeg encode failed for {fmt}: {msg or 'unknown error'}"
        )
    return stdout, content_type
