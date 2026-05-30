# PCM Streaming for Qwen3-TTS

Real-time audio streaming is available for the `qwen3_tts` backend when
`response_format="pcm"` is requested. Instead of buffering the full utterance
before responding, the server yields raw PCM chunks as the GPU decodes them —
first audio arrives in ~200–700 ms depending on GPU and `chunk_size`.

Non-`pcm` formats (mp3, wav, opus, aac, flac) and non-Qwen3 backends (Kokoro)
are unaffected and continue to use the fully-buffered path.

---

## Wire format

| Property | Value |
|---|---|
| HTTP method | `POST /v1/audio/speech` |
| `response_format` | `"pcm"` |
| Transfer-Encoding | `chunked` (HTTP/1.1) |
| Content-Type | `application/octet-stream` |
| `X-Sample-Rate` header | e.g. `24000` |
| Sample encoding | Signed 16-bit little-endian (int16 LE) |
| Channels | Mono |
| Sample rate | 24 000 Hz (Qwen3-TTS fixed rate) |

Each HTTP chunk is a raw contiguous block of int16 LE samples with no WAV
header or framing. Concatenating all chunks produces a valid raw PCM stream
that can be decoded with any tool that understands the parameters above.

---

## Quick start

```bash
# Play audio in real time as it arrives (Linux/WSL)
curl -s -N http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "qwen3-tts-0.6b",
        "input": "Streaming audio from Qwen3 TTS.",
        "voice": "alloy",
        "response_format": "pcm"
      }' \
  | aplay -f S16_LE -r 24000 -c 1

# Save to file (all platforms)
curl -s -N http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-0.6b","input":"Hello!","voice":"alloy","response_format":"pcm"}' \
  --output speech.raw

# Convert saved raw PCM to WAV with ffmpeg
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.raw speech.wav
```

Python example using `httpx`:

```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "qwen3-tts-0.6b",
        "input": "Hello from streaming Qwen3 TTS!",
        "voice": "alloy",
        "response_format": "pcm",
    },
    timeout=None,
) as r:
    r.raise_for_status()
    sample_rate = int(r.headers.get("x-sample-rate", 24000))
    print(f"Sample rate: {sample_rate} Hz")
    with open("speech.raw", "wb") as f:
        for chunk in r.iter_bytes():
            f.write(chunk)
```

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `TALKIES_QWEN3_STREAM_CHUNK_SIZE` | `8` | Codec steps per yielded chunk. 12 steps ≈ 1 s; 8 ≈ 667 ms. Smaller = lower time-to-first-audio but more decode overhead per chunk. |

```bash
# Example: trade slightly higher throughput for lower TTFA
docker run ... -e TALKIES_QWEN3_STREAM_CHUNK_SIZE=4 ...
```

Chunk size guidance (0.6B model, RTX 4090):

| `chunk_size` | Audio per chunk | Approx. TTFA |
|---|---|---|
| 4 | ~333 ms | ~156 ms |
| 8 | ~667 ms | ~156 ms |
| 12 | ~1 000 ms | ~156 ms |

TTFA is dominated by CUDA-graph warmup on the first call; subsequent calls are
much faster. Smaller chunks have more decode overhead but lower perceived
latency. `8` is the default as it stays real-time on all tested hardware
including Jetson AGX Orin.

---

## Cancellation

When a client disconnects mid-stream, the server detects it via Starlette's
`StreamingResponse` generator teardown (the async generator's `finally` block
is invoked). The implementation:

1. Sets a `threading.Event` to signal the GPU worker thread.
2. Drains the internal `asyncio.Queue` so the worker is not blocked on a
   full-queue `.put()`.
3. Awaits the worker thread task to join cleanly.

No zombie threads or unreleased GPU locks remain after a cancelled request.

---

## Architecture notes

```
POST /v1/audio/speech (response_format=pcm, qwen3 backend)
│
├─ server.py: speech()
│   ├─ validates model / voice / format (same as buffered path)
│   ├─ evicts sibling models
│   └─ returns StreamingResponse(_pcm_stream(), headers={"X-Sample-Rate": "24000"})
│
└─ _pcm_stream() async generator
    └─ backend.synthesize_stream(...) async generator
        ├─ pre-yield validation (text, voice) → HTTP 4xx if bad
        ├─ await get_model()  → lazy-loads + CUDA-graph warmup on first call
        └─ async with self._lock:  ← held for full stream duration
            ├─ asyncio.Queue(maxsize=4)   ← bounded backpressure
            ├─ threading.Event            ← cancellation signal
            └─ _stream_worker thread
                └─ model.generate_voice_clone_streaming(chunk_size=N)
                    yields (float32 ndarray, sample_rate, timing)
                    → np.clip + cast to int16 → bytes → queue.put()
                    → async generator yields bytes to StreamingResponse
```

The GPU lock (`self._lock`) is held for the entire stream, matching the
buffered path. Only one Qwen3-TTS synthesis (streaming or not) runs at a time;
concurrent requests queue behind the lock.
