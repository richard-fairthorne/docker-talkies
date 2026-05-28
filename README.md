# talkies

> Self-hosted `/v1/audio/transcriptions` that fronts seven open ASR models behind OpenAI's wire format. Point your existing OpenAI client at it, change the model slug, and you're done.

`POST /v1/audio/transcriptions` with a multipart `file` + a `model` slug → text back. Same wire shape as OpenAI. The same client you point at `api.openai.com/v1/audio/transcriptions` works here, you just change the base URL and the slug. That's the entire story.

Swap the slug — `whisper-large-v3`, `whisper-large-v3-turbo`, `distil-whisper-large-v3`, `parakeet-tdt-0.6b-v3`, `canary-180m-flash`, `canary-1b-flash`, `canary-qwen-2.5b` — and the contract stays identical. Behind the scenes the request is dispatched to the right backend (faster-whisper for the whisper family, NeMo for everything else), the audio is normalized to 16 kHz mono WAV, long files are sliced via Silero VAD into ≤28-second speech regions, results are stitched back into one Whisper-shape timeline. None of that leaks into the wire shape. You just get text.

Need stereo speaker diarization? Pass `diarization=true` and upload a stereo file — left channel = speaker L, right channel = speaker R, output gets per-segment `channel` tags and the text is split into chronological `L:` / `R:` turn lines. Two-mic setups (interview rigs, podcast splits, dual-track ham recordings) end up with a clean transcript without you having to bolt a separate diarization model onto your stack.

## Table of contents

- [Quick start](#quick-start)
- [Supported models](#supported-models)
- [What's NOT supported](#whats-not-supported)
- [API — `POST /v1/audio/transcriptions`](#api--post-v1audiotranscriptions)
  - [Request fields](#request-fields)
  - [Response formats](#response-formats)
    - [`json` (default)](#json-default)
    - [`verbose_json`](#verbose_json)
    - [`text`](#text)
    - [`srt`](#srt)
    - [`vtt`](#vtt)
  - [Stereo diarization](#stereo-diarization)
  - [Translation (Canary X→Y)](#translation-canary-xy)
  - [Long files + VAD chunking](#long-files--vad-chunking)
  - [Error contract](#error-contract)
- [Resource-management endpoints (Ollama-style)](#resource-management-endpoints-ollama-style)
- [Server-side file staging (`/v1/files`)](#server-side-file-staging-v1files)
- [MCP endpoint (`/v1/mcp`)](#mcp-endpoint-v1mcp)
- [Bearer-token auth](#bearer-token-auth)
- [Configuration (env vars)](#configuration-env-vars)
- [CPU vs CUDA images](#cpu-vs-cuda-images)
- [Architecture](#architecture)
- [Customizing the model registry](#customizing-the-model-registry)
- [Development](#development)
- [Security notes](#security-notes)
- [License](#license)

## Quick start

```bash
docker run -d --name talkies \
  -v $HOME/talkies-models:/data \
  -p 8000:8000 \
  psyb0t/talkies:latest

# On boot the entrypoint downloads every model in models.json into
# /data/models/<slug>/ as flat directories — no HF cache, no
# `models--org--repo/snapshots/<hash>` indirection. Each snapshot is
# ~75MB-3GB. Bind-mount /data so subsequent restarts are no-ops.
# To restrict the download set
# `-e TALKIES_ENABLED_MODELS=whisper-large-v3-turbo,canary-180m-flash`
# — only those slugs are pulled, and only those are queryable.
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file=@samples/hello.wav" \
  -F "model=whisper-large-v3-turbo" | jq

# Verbose JSON — full Whisper shape with per-segment + per-word timestamps.
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file=@samples/hello.wav" \
  -F "model=whisper-large-v3-turbo" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -F "timestamp_granularities[]=segment" | jq

# SRT subtitle output (drop straight into a video player).
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file=@samples/lecture.mp3" \
  -F "model=whisper-large-v3" \
  -F "response_format=srt" > lecture.srt

# Stereo diarization — left/right channels become speakers L/R.
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file=@samples/interview-stereo.wav" \
  -F "model=whisper-large-v3-turbo" \
  -F "diarization=true" \
  -F "response_format=verbose_json" | jq

# Which models are configured, which are loaded, evict one if you want.
curl -s http://localhost:8000/v1/models | jq
curl -s http://localhost:8000/api/ps | jq
curl -s -X DELETE "http://localhost:8000/api/ps/whisper-large-v3-turbo"
curl -s -X POST  http://localhost:8000/unload | jq    # evict everything
```

GPU variant: pull `psyb0t/talkies:latest-cuda` and add `--gpus all` to `docker run`. The CPU image only ships the four models that actually run reasonably without a GPU (the three Whisper variants + `canary-180m-flash`). The CUDA image adds Parakeet-TDT, Canary-1B-Flash, and Canary-Qwen-2.5B on top — they need VRAM to be anything other than a space heater.

## Supported models

All seven are publicly-available ASR foundation models with permissive licenses. They split into three engine families:

| Slug | HF repo | Family | Image | Languages | License |
|---|---|---|---|---|---|
| `whisper-large-v3` | `Systran/faster-whisper-large-v3` | faster-whisper (CTranslate2) | CPU + CUDA | 99 (auto-detect) | MIT |
| `whisper-large-v3-turbo` | `deepdml/faster-whisper-large-v3-turbo-ct2` | faster-whisper (CTranslate2) | CPU + CUDA | 99 (auto-detect) | MIT |
| `distil-whisper-large-v3` | `Systran/faster-distil-whisper-large-v3` | faster-whisper (CTranslate2) | CPU + CUDA | English | MIT |
| `parakeet-tdt-0.6b-v3` | `nvidia/parakeet-tdt-0.6b-v3` | NeMo (TDT) | CUDA only | English | CC-BY-4.0 |
| `canary-180m-flash` | `nvidia/canary-180m-flash` | NeMo Canary (multitask) | CPU + CUDA | English (ASR only on this size) | CC-BY-4.0 |
| `canary-1b-flash` | `nvidia/canary-1b-flash` | NeMo Canary (multitask) | CUDA only | en, de, fr, es (ASR + X→en / en→X translation) | CC-BY-4.0 |
| `canary-qwen-2.5b` | `nvidia/canary-qwen-2.5b` | NeMo Canary SALM (Qwen2 decoder) | CUDA only | English | CC-BY-4.0 |

The three Whisper variants are tokenized + executed through [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which is roughly 4× faster than the reference OpenAI implementation at the same accuracy on the same hardware. The four NVIDIA models go through NeMo's native inference path — Parakeet uses the TDT decoder, Canary models use the multitask transformer head, Canary-Qwen swaps the decoder for a Qwen2 LLM (the "speech-augmented language model" trick that lets you tack instructions onto the prompt).

You don't have to care about any of this from the client side. You pick the slug; we handle the engine.

## What's NOT supported

A short list of things that look like they might work but don't, so you don't waste an afternoon finding out the hard way.

| Thing | Status | Notes |
|---|---|---|
| **Streaming / partial results** | Not supported | The endpoint is request/response. The whole file is buffered, normalized, transcribed, and the full response is returned. No SSE, no websockets, no chunked streaming output. |
| **`prompt` request field** | Accepted, ignored | Present in the form schema for OpenAI compatibility. It's not threaded into any backend. |
| **`temperature` request field** | Accepted, ignored | Same — present for compatibility, not used. |
| **Mono file + `diarization=true`** | 400 error | Diarization requires a 2-channel input. Mono uploads get rejected with `NotStereoError`. |
| **>2 channels with diarization** | 400 error | Only stereo L/R is meaningful. 5.1 / 7.1 / multi-track uploads with `diarization=true` are rejected. (Without `diarization=true`, multi-channel inputs are downmixed to mono and transcribed normally.) |
| **Per-request translation task selection** | Not supported via API | The `task` (`asr` vs `s2t_translation`) and `target_lang` are baked into the model slug via `models.json`'s `default_task` / `default_target_lang`. To enable translation you add a custom slug — see [Translation](#translation-canary-xy). |
| **Multiple models resident at once** | Not supported in one container | Every transcription request evicts other loaded models (sibling eviction) so VRAM/RAM doesn't get split. If you genuinely need two models simultaneously, run two containers. |
| **arm64 / aarch64** | Not built | `linux/amd64` only. `nemo_toolkit[asr]` + the rest of the chain doesn't currently resolve cleanly on arm64 at the pinned versions. |
| **`canary-qwen-2.5b` timestamps** | Not produced | The SALM head has no alignment output, so `verbose_json` comes back with `segments: []` and `words: []`, and `srt` / `vtt` fall back to a single full-duration cue. Transcription itself still covers the whole file — long inputs are VAD-chunked and the per-chunk text is concatenated. |
| **Files > 100 MB** | 413 error by default | Configurable via `TALKIES_MAX_UPLOAD_BYTES`. Bump it for long lectures / podcasts. |
| **Custom Canary prompts** | Not supported | NeMo's Canary prompt format (`<\|spltoken\|>`, source/target tokens) isn't exposed to callers. You get the prompt the backend builds from `source_lang`/`target_lang`/`task`. |
| **Speaker identification beyond stereo channels** | Not supported | There's no voice clustering / speaker-embedding model in here. "Diarization" means "two-channel split", not "figure out who's talking from the audio". |
| **Real-time / live mic input** | Out of scope | Send a file. If you need live transcription, buffer a few seconds client-side and POST chunks. |
| **OpenAI-compatible translation endpoint (`/v1/audio/translations`)** | Not implemented | OpenAI's separate `/v1/audio/translations` (always-translate-to-English) isn't exposed. Use a Canary slug with `default_task=s2t_translation` instead. |

## API — `POST /v1/audio/transcriptions`

Multipart form. Same field names as OpenAI's transcription endpoint where they overlap. Extra fields are talkies-specific.

### Request fields

| Field | Required | Default | Notes |
|---|---|---|---|
| `file` | one of `file`/`file_path` | — | Audio file (any format that ffmpeg can decode — WAV, MP3, M4A, FLAC, OGG, WebM, Opus, MP4 audio track, etc.). Capped at `TALKIES_MAX_UPLOAD_BYTES` (default 100 MB). |
| `file_path` | one of `file`/`file_path` | — | Either (a) a server-side path of a file previously uploaded via `PUT /v1/files/{path}` — leading `/` is stripped, traversal segments are rejected — or (b) an `http://` / `https://` URL, which is downloaded once into `${TALKIES_DATA_DIR}/files/downloads/` and cached for subsequent requests (same URL = cache hit, no re-download). The `TALKIES_MAX_UPLOAD_BYTES` cap doesn't apply; URL downloads are capped separately via `TALKIES_MAX_DOWNLOAD_BYTES` (default 1 GiB). See [Server-side file staging](#server-side-file-staging-v1files). |
| `model` | yes | — | One of the configured slugs (see `GET /v1/models`). Unknown slug → 404. |
| `language` | no | model default | ISO-639-1 language code. Whisper auto-detects when omitted; Canary multilingual uses its `default_source_lang` from `models.json` (English unless overridden). |
| `response_format` | no | `json` | `json` / `text` / `verbose_json` / `srt` / `vtt`. See [Response formats](#response-formats). |
| `timestamp_granularities[]` | no | `[]` (segments only) | Repeat the field to enable extra granularities. `segment` is always returned in verbose_json; add `word` for per-word timestamps. |
| `prompt` | no | — | Accepted for OpenAI compatibility, **currently ignored**. |
| `temperature` | no | — | Accepted for OpenAI compatibility, **currently ignored**. |
| `diarization` | no | `false` | Stereo-channel diarization. Requires a 2-channel input file; mono uploads return 400. See [Stereo diarization](#stereo-diarization). |

### Response formats

The `response_format` field picks one of five wire shapes. The content-type and structure differ — pick based on whether you need a string, a structured object, a subtitle file, or full Whisper-shape segment data.

| `response_format` | Content-Type | Shape |
|---|---|---|
| `json` (default) | `application/json` | `{"text": "..."}` — just the transcript. |
| `text` (alias: `txt`) | `text/plain` | The transcript as plain text. No JSON envelope. |
| `verbose_json` | `application/json` | Full Whisper shape — `task`, `language`, `duration`, `text`, `segments`, `words`. |
| `srt` | `application/x-subrip` | SubRip subtitle file, one cue per segment. |
| `vtt` | `text/vtt` | WebVTT subtitle file, one cue per segment. |

#### `json` (default)

```json
{
  "text": " the full transcript as a single string"
}
```

The simplest case. One field. The leading space mirrors Whisper's tokenizer output and is preserved verbatim — strip it client-side if you don't want it.

#### `verbose_json`

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 6.42,
  "text": " the full transcript",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.31,
      "text": " the full transcript",
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": null,
      "compression_ratio": null,
      "no_speech_prob": null
    }
  ],
  "words": [
    {"word": " the", "start": 0.0, "end": 0.12},
    {"word": " full", "start": 0.12, "end": 0.34}
  ]
}
```

Both `segments` and `words` are always present in the envelope — backends that don't produce alignments (`canary-qwen-2.5b`) emit empty arrays for both rather than omitting the keys, so clients can read the fields unconditionally. `timestamp_granularities[]` is accepted for OpenAI compatibility but ignored — we emit segment-level and word-level timings in a single pass, so there's no cost to always sending both. `tokens` is always `[]` (the underlying token IDs only mean something in the context of each model's tokenizer, and most clients don't use them). The Whisper-only confidence fields (`avg_logprob`, `compression_ratio`, `no_speech_prob`) are emitted as `null` regardless of backend — they're null-filled rather than omitted so OpenAI clients reading them don't crash. Word entries carry `word`/`start`/`end` only; no `probability` field is emitted by any backend.

`canary-qwen-2.5b` (SALM) has no alignment head, so verbose_json comes back with `segments: []` and `words: []`. For `srt` / `vtt` against this model, the server falls back to a single cue spanning `[0, duration]` containing the full transcript — the file is valid and playable, just one cue long.

When `diarization=true`, every entry in `segments` and `words` carries an extra `"channel": "L"` or `"channel": "R"` field, and the top-level `text` is the alternating-turn-lines form (see [Stereo diarization](#stereo-diarization)).

#### `text`

```
 the full transcript as a single string
```

`text/plain`. Identical to `json`'s `text` field, just without the JSON envelope. Useful when you're piping the output straight into another tool.

With `diarization=true`:

```
L: hi how's it going
R: not bad you
L: cool man
```

#### `srt`

```
1
00:00:00,000 --> 00:00:02,310
 the full transcript

2
00:00:02,310 --> 00:00:05,780
 continuing here on the next segment
```

Standard SubRip. Drop straight into a video player or burn into a video with `ffmpeg -vf subtitles=foo.srt`. One cue per VAD-segmented chunk. Timestamps are end-to-end in the source file (not per-chunk-relative).

With `diarization=true`, each cue is prefixed with the channel:

```
1
00:00:00,000 --> 00:00:01,420
L: hi how's it going

2
00:00:01,500 --> 00:00:02,310
R: not bad you
```

#### `vtt`

```
WEBVTT

00:00:00.000 --> 00:00:02.310
 the full transcript

00:00:02.310 --> 00:00:05.780
 continuing here on the next segment
```

Standard WebVTT. Same content as SRT but with the `WEBVTT` header, `.` as the decimal separator instead of `,`, and no cue indices. Use this for HTML5 `<track kind="subtitles">`.

With `diarization=true`, cue payloads are prefixed with `L:` / `R:` exactly like the SRT variant.

### Stereo diarization

Pass `diarization=true` and upload a 2-channel audio file. Left channel = speaker `L`, right channel = speaker `R`. Each channel is transcribed independently through the chosen backend, then the two timelines are merged chronologically by segment start time.

What changes in the output:
- `verbose_json` — every segment and every word gets a `"channel": "L"` or `"R"` field. Segments are re-numbered after the merge so `id` is contiguous across channels.
- `text` (top-level in JSON, or as the body in `response_format=text`) — rebuilt as alternating turn lines: `L: <whole L segment>\nR: <whole R segment>\n...`. Consecutive same-channel segments are collapsed into one line per turn so you don't get one line per breath.
- `srt` / `vtt` — each cue's payload is prefixed with `L:` / `R:`.

Caveats:
- Requires exactly **2 channels**. Mono → 400. >2 channels → 400.
- Both channels go through the same backend instance sequentially (model only sits resident once). Latency is ~2× the mono case for the same audio.
- The backend processes each channel as if it were a standalone mono recording — there's no acoustic separation logic between channels. If your "stereo" recording has both speakers on both channels at different gains (e.g. a single-mic recording rendered to stereo), diarization won't magically split them. The technique is exact for true two-mic setups, useless otherwise.

### Translation (Canary X→Y)

The Canary multitask models (`canary-180m-flash`, `canary-1b-flash`) can do speech-to-text translation natively — `canary-1b-flash` covers EN/DE/FR/ES in both directions (`X→en` and `en→X`).

**However**: the `task` field (`asr` vs `s2t_translation`) and `target_lang` aren't request-time parameters. They come from the model registry entry's `default_task` and `default_target_lang`. The shipped `models.json` uses `default_task=asr` for every slug, so out of the box the API only transcribes.

To enable translation, add a translation-specific slug to a custom `models.json` and bind-mount it (see [Customizing the model registry](#customizing-the-model-registry)). Example — German speech → English text:

```json
{
  "models": {
    "canary-1b-flash-de2en": {
      "repo": "nvidia/canary-1b-flash",
      "executor": "canary_multitask",
      "default_source_lang": "de",
      "default_target_lang": "en",
      "default_task": "s2t_translation",
      "languages": ["de"]
    }
  }
}
```

Then call it normally:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file=@samples/german-clip.wav" \
  -F "model=canary-1b-flash-de2en" | jq
```

The output shape is identical — `text` carries the English translation, `language` reflects the source language. You can request multiple directions by adding multiple slugs (`canary-1b-flash-en2de`, `canary-1b-flash-fr2en`, etc.) all pointing at the same HF repo with different `default_task` / `default_source_lang` / `default_target_lang` triples. talkies loads the underlying weights once and just changes the prompt format per slug.

`canary-180m-flash` is English-ASR-only by design — it doesn't have the multilingual head. Don't point a translation slug at it. `canary-qwen-2.5b` does English ASR only; the SALM head isn't a translator.

### Long files + VAD chunking

Anything longer than `TALKIES_VAD_CHUNK_THRESHOLD_SECONDS` (default 30s) gets sliced through [Silero VAD](https://github.com/snakers4/silero-vad) into ≤`TALKIES_VAD_MAX_SPEECH_SECONDS` (default 28s) speech regions before being handed to the model. Whisper's own internal long-form path is bypassed because:

1. We need consistent chunking behavior across **all** backends (Whisper, Parakeet, Canary multitask, Canary SALM) — the whisper-internal sliding window doesn't apply to the NeMo backends.
2. VAD-aligned cuts produce noticeably better segment boundaries on real-world audio than fixed 30-second window slides.
3. Timestamps are re-assembled by offsetting each chunk's segment/word timings by the chunk's start in the source timeline, so you get one continuous `segments` list spanning the whole file.

Canary SALM (`canary-qwen-2.5b`) is the partial exception — same VAD chunker, but because the SALM head has no alignment output, the per-chunk results are concatenated as plain text (with a single space) instead of being stitched into a `segments` timeline. You still get the full transcript on long files; you just don't get per-segment timestamps for this one model.

### Error contract

Two response shapes — application errors return `{"detail": "..."}` with a human-readable string; Pydantic validation errors (422) return the FastAPI default structured array.

**App errors** (`400`, `404`, `413`):

```json
{ "detail": "human-readable error string" }
```

**Validation errors** (`422`):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "model"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

| Status | Shape | When |
|---|---|---|
| 200 | per `response_format` | success |
| 400 | string | bad audio (ffmpeg conversion failure, unsupported codec, corrupted file), mono input with `diarization=true`, >2 channels with `diarization=true`, neither or both of `file`/`file_path` set, invalid `/v1/files` path (null bytes, backslashes, `.` / `..` segments, double slashes), URL download failure (DNS, HTTP 4xx/5xx, unsupported scheme, no host, too many redirects, size exceeded `TALKIES_MAX_DOWNLOAD_BYTES`, blocked by SSRF guard when `TALKIES_BLOCK_PRIVATE_DOWNLOADS=true`) |
| 401 | string | only emitted when `TALKIES_AUTH_TOKEN` is set: missing / malformed / wrong bearer token. Response includes `WWW-Authenticate: Bearer`. |
| 404 | string | unknown model slug in `model` field, unknown model in `DELETE /api/ps/{model_id}`, model in DELETE path is configured but not currently loaded, `file_path` references a missing file, `/v1/files/{path}` GET or DELETE on a non-existent file |
| 413 | string | upload exceeded `TALKIES_MAX_UPLOAD_BYTES` (applies to `POST /v1/audio/transcriptions` multipart `file` and `PUT /v1/files/{path}` body; **not** to `file_path`-driven transcribe) |
| 422 | array | Pydantic validation (missing required fields, wrong field types, malformed `timestamp_granularities[]`) |
| 500 | string | unhandled backend exception (NeMo / faster-whisper / torch internal failure) |

Auth: set `TALKIES_AUTH_TOKEN` to require a bearer token on every route (see [Bearer-token auth](#bearer-token-auth)). Without it, every endpoint is open — stick the container behind a reverse proxy (Caddy, Traefik, nginx, your VPN's auth gateway) if you don't want the built-in token. There's no built-in rate limiting either; that's a reverse-proxy concern.

## Resource-management endpoints (Ollama-style)

talkies mirrors a subset of [speaches](https://github.com/speaches-ai/speaches) and Ollama's resource-management surface, so a single LiteLLM-style proxy can drive both:

| Endpoint | Behavior |
|---|---|
| `GET /healthz` | Unauthenticated liveness. Returns `{ok, device, models}` where `models` is the configured slug list. |
| `GET /v1/models` | OpenAI-style model list. `{"object": "list", "data": [{"id": slug, ...}]}`. |
| `GET /api/ps` | Currently-loaded models, with per-model `idle_seconds` (seconds since last use). |
| `DELETE /api/ps/{model_id}` | Evict one model from RAM/VRAM. `model_id` can be URL-encoded (`whisper%2Flarge` → `whisper/large`) — LiteLLM's resource manager does this on slashes. Returns 404 if not loaded. |
| `POST /unload` | Evict every loaded model. Returns the list that was actually unloaded. |

Behind these endpoints there's an **idle sweeper** that runs on a `TALKIES_SWEEPER_INTERVAL` cadence (default 60s) and unloads any backend that hasn't been called in `TALKIES_MODEL_TTL` seconds (default 600s = 10min). Set `TALKIES_MODEL_TTL=0` to disable auto-unload entirely.

There's also **sibling eviction at transcription time**: when a request arrives and a model that isn't the requested one is currently loaded, the other model gets unloaded first. All seven models compete for the same VRAM (or the same fat slice of RAM on CPU), so loading two at once on a 12GB card OOMs you. Ollama does this implicitly via its scheduler; we do it explicitly per-request. If you genuinely want two models resident simultaneously, you want two containers.

## Server-side file staging (`/v1/files`)

If you're going to transcribe the same recording multiple times (different `response_format`, different model, re-runs while you tweak something else) it gets annoying re-uploading the same bytes on every call. The `/v1/files` API lets you stage a file on the server once, then reference it by relative path in `/v1/audio/transcriptions` via the `file_path` form field.

Files land under `${TALKIES_DATA_DIR}/files/<path>`. The path you supply in the URL is treated as relative to that root — `/foo/bar/clip.mp3` and `foo/bar/clip.mp3` both end up at `${TALKIES_DATA_DIR}/files/foo/bar/clip.mp3`. Parent directories are created on PUT and pruned (only the empty ones, only up to but not including the root) on DELETE.

| Endpoint | Behavior |
|---|---|
| `GET /v1/files` | List every staged file. Returns `{"files": [{"path": "...", "size": N, "modified": "...Z"}]}`, sorted by path. |
| `PUT /v1/files/{path}` | Upload raw bytes (no multipart wrapper — `--data-binary @local-file`). Capped at `TALKIES_MAX_UPLOAD_BYTES`. Written atomically (`.part` tmp file → rename). Overwrites any existing file at the same path. Returns 201 with `{"path": "...", "size": N}`. |
| `GET /v1/files/{path}` | Streams the file back. Content-Type guessed from the extension (`.mp3` → `audio/mpeg`, `.wav` → `audio/wav`, etc.); falls back to `application/octet-stream`. 404 if missing. |
| `DELETE /v1/files/{path}` | Removes the file and prunes empty parent directories up to the root. 404 if missing. |

```bash
# Stage the file once.
curl -X PUT --data-binary @lecture.mp3 \
  -H "Content-Type: audio/mpeg" \
  http://localhost:8000/v1/files/lectures/2026-03-15/lecture.mp3

# Reuse it across multiple transcribe calls — no re-upload.
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file_path=lectures/2026-03-15/lecture.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "response_format=verbose_json" | jq

curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file_path=lectures/2026-03-15/lecture.mp3" \
  -F "model=canary-1b-flash" \
  -F "response_format=srt" > lecture.srt

# List what's there.
curl -s http://localhost:8000/v1/files | jq

# Delete when done.
curl -X DELETE http://localhost:8000/v1/files/lectures/2026-03-15/lecture.mp3
```

Path safety rules: null bytes, backslashes, `.` segments, `..` segments and double slashes are all rejected with 400. After lexical validation the resolved absolute path is required to remain inside `${TALKIES_DATA_DIR}/files/` — symlinks pointing outside the root are caught here and refused. Symlinks themselves are not followed for GET / DELETE (a symlink at a request path returns 404 as if no file is there).

Transcribe requests must specify exactly one of `file` or `file_path`. Passing both or neither returns 400. The `TALKIES_MAX_UPLOAD_BYTES` cap does **not** apply to `file_path` — the file is already on disk, you put it there.

### Pulling from a URL

`file_path` also accepts an `http://` or `https://` URL. First request downloads the bytes into `${TALKIES_DATA_DIR}/files/downloads/<sha256(url)[:16]>-<safe-basename>` and runs the transcription off that cached file. Subsequent requests with the same URL skip the download entirely. Two concurrent requests for the same URL won't double-fetch — the second waiter sees the cache hit after the first finishes.

```bash
# First call downloads, transcribes off the cached copy.
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file_path=https://example.com/podcasts/ep-042.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "response_format=verbose_json" | jq

# Second call hits the cache — same URL, no re-download.
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file_path=https://example.com/podcasts/ep-042.mp3" \
  -F "model=canary-1b-flash" \
  -F "response_format=srt" > ep-042.srt
```

Downloads land in `downloads/` under the files root, so `GET /v1/files` lists them alongside your uploads and `DELETE /v1/files/downloads/<key>` invalidates a single cached entry. The cache key is a 16-char prefix of `sha256(url)`, suffixed with a safe basename from the URL path so listings stay readable.

Constraints applied during the download:

- Size: streamed to disk with a per-download cap from `TALKIES_MAX_DOWNLOAD_BYTES` (default 1 GiB). Exceeding the cap aborts and removes the partial file.
- Redirects: followed manually, capped at 5 hops, with the SSRF guard re-applied at every hop.
- Timeouts: 10 s connect, 300 s read per response chunk.
- SSRF: off by default (LAN-fetch is the common self-hosted case). Set `TALKIES_BLOCK_PRIVATE_DOWNLOADS=true` to reject URLs whose hostname resolves to private / loopback / link-local / multicast / reserved IPs — handy if you're exposing the server to untrusted clients on a network where it can reach metadata endpoints or internal services.

## MCP endpoint (`/v1/mcp`)

talkies speaks the [Model Context Protocol](https://modelcontextprotocol.io) over a Streamable HTTP transport at `/v1/mcp`. Point an MCP-aware agent (Claude Code, Claude Desktop, MCP Inspector, anything that supports the streamable-http transport) at `http://<host>:8000/v1/mcp` and it gets six tools for free:

| Tool | What it does |
|---|---|
| `list_models` | Discover available ASR slugs (returns `[{slug, executor, default_source_lang, default_target_lang, default_task, loaded}]`). |
| `transcribe` | Run ASR on a `file_path` — either an `http(s)://` URL (downloaded + cached server-side) or a path under the staging area. Args: `model`, `language?`, `response_format?` (`json`/`verbose_json`/`text`/`srt`/`vtt`), `diarization?`. JSON formats return a JSON-encoded string; text/srt/vtt return raw. |
| `list_files` | Same payload as `GET /v1/files`. |
| `put_file` | Upload to the staging area. Body is base64-encoded (`content_base64`), decoded size capped at `TALKIES_MAX_UPLOAD_BYTES`. For big files, prefer `PUT /v1/files/{path}` over HTTP — JSON-RPC + base64 chews token budget. |
| `get_file` | Read a staged file as base64. Same size cap. Same advice — for big bytes, hit `GET /v1/files/{path}` over HTTP instead. |
| `delete_file` | Remove a staged file, prune empty parents up to (but not including) the root. |

Wire it into Claude Code:

```bash
claude mcp add --transport http talkies http://localhost:8000/v1/mcp
```

If `TALKIES_AUTH_TOKEN` is set, the client must send `Authorization: Bearer <token>` — Claude Code supports this via `--header`:

```bash
claude mcp add --transport http talkies http://localhost:8000/v1/mcp \
  --header "Authorization: Bearer <your-token>"
```

The MCP server runs over the same FastAPI process, shares `BACKENDS` / `REGISTRY` with the HTTP routes, and goes through the same auth middleware. Sibling-eviction and idle-unload work identically — a model loaded by the MCP `transcribe` tool is the same instance the HTTP endpoint sees.

## Bearer-token auth

Set `TALKIES_AUTH_TOKEN` to gate every route — `/v1/audio/transcriptions`, `/v1/files/*`, `/v1/mcp`, the resource-management endpoints. Requests without `Authorization: Bearer <token>` get 401 with `WWW-Authenticate: Bearer`. `/healthz` and CORS preflights (`OPTIONS`) are exempt so probes + browser clients keep working.

```bash
# Server side:
docker run -p 8000:8000 -e TALKIES_AUTH_TOKEN=$(openssl rand -hex 32) \
  -v $PWD/data:/data psyb0t/talkies:latest

# Client side:
curl -H "Authorization: Bearer <token>" http://localhost:8000/v1/models
```

If you don't set the env var (or set it to an empty string), talkies stays wide open — that's the historical default and matches what self-hosted deployments behind a private network expect. The token is checked with `hmac.compare_digest`, so timing-side-channel leak is bounded. Keep the token out of URLs, query strings, and logs (talkies doesn't log it; your reverse proxy might — check there).

## Configuration (env vars)

| Var | Default | What it does |
|---|---|---|
| `TALKIES_AUTH_TOKEN` | (empty = no auth) | Bearer token required on every route except `/healthz`. Unset / empty leaves the server wide open (existing behaviour). When set, every HTTP request and every MCP call must include `Authorization: Bearer <token>` or it returns 401. |
| `TALKIES_DEVICE` | `auto` (in entrypoint) / `cpu` / `cuda` (per-image default) | `auto` picks `cuda` if available else `cpu`. Pin to a specific GPU with `cuda:N`. |
| `TALKIES_MODELS_FILE` | `/app/models.json` | Path to the model registry JSON. Override to ship a custom subset (e.g. only Whisper-turbo if you only care about that one model). |
| `TALKIES_DATA_DIR` | `/data` | Base data dir. Model snapshots land in `$TALKIES_DATA_DIR/models/<slug>/` as flat per-model directories (no HF cache layout); staged uploads from `/v1/files` land under `$TALKIES_DATA_DIR/files/`. Bind-mount this to persist both across restarts. |
| `TALKIES_MODEL_TTL` | `600` (10 min) | Idle time before a loaded backend is unloaded by the sweeper. Bare number = seconds; also accepts Go-style `3h30m5s`, `45m`, `90s`. `0` disables auto-unload. |
| `TALKIES_SWEEPER_INTERVAL` | `60` | How often the sweeper checks for idle models (seconds; same Go-style parsing). |
| `TALKIES_LOAD_TIMEOUT` | `300` | Per-model load timeout (seconds; same Go-style parsing). Initial weights download + warmup runs inside this budget. |
| `TALKIES_MAX_UPLOAD_BYTES` | `104857600` (100 MB) | Reject uploads larger than this with 413. Bump for long lectures / podcasts. Applies to `POST /v1/audio/transcriptions` (`file` field) and `PUT /v1/files/{path}` only. |
| `TALKIES_MAX_DOWNLOAD_BYTES` | `1073741824` (1 GiB) | Abort URL downloads (when `file_path` is an http(s) URL) larger than this. Bigger default than the upload cap because downloads stream straight to disk, no in-memory buffering. |
| `TALKIES_BLOCK_PRIVATE_DOWNLOADS` | `false` | Set to `true` to refuse URL downloads whose hostname resolves to private / loopback / link-local / multicast / reserved IPs. Default `false` because the typical self-hosted deployment is a LAN box fetching from another LAN box. Flip to `true` if the server's exposed to untrusted clients. |
| `TALKIES_ENABLED_MODELS` | (empty = all from models.json) | Comma-separated slugs whitelist. Restricts both the boot-time snapshot download and the queryable surface of `/v1/models`. Unknown slugs fail fast on startup. Leave empty to enable every model in `models.json` (heavy on first boot — the CUDA image's full set is ~12 GB on disk). |
| `TALKIES_PRELOAD` | (empty) | Comma-separated slugs to load into RAM/VRAM at boot, before uvicorn accepts requests. Skips the cold-load penalty on the first transcription. Must be a subset of `TALKIES_ENABLED_MODELS` (or any slug from `models.json` when that's empty). |
| `TALKIES_VAD_CHUNK_THRESHOLD` | `30.0` | Audio longer than this (seconds) goes through VAD chunking. Shorter clips are sent to the backend whole. |
| `TALKIES_VAD_MAX_SPEECH` | `28.0` | Max length of a single VAD-detected speech region (seconds). Anything longer gets split. Should stay under Whisper's 30s internal window. |
| `TALKIES_VAD_MIN_SILENCE_MS` | `500` | Silero VAD param — minimum gap (ms) to consider a region break. |
| `TALKIES_VAD_SPEECH_PAD_MS` | `200` | Silero VAD param — how much silence padding (ms) to add around each detected speech region. |
| `TALKIES_VAD_THRESHOLD` | `0.5` | Silero VAD speech-probability threshold. Lower = more aggressive (catches quiet speech, more false positives). |
| `HF_HUB_OFFLINE` | `1` (in image) | Refuse network calls from HuggingFace Hub. The entrypoint transparently unsets this for the one-shot prefetch step so the initial download still works; the server process itself runs with the image default (offline). You shouldn't need to touch this — it's an internal escape hatch. |

## CPU vs CUDA images

| Image | Tag | Platforms | Models served | Image size (approx) |
|---|---|---|---|---|
| CPU | `psyb0t/talkies:latest` | `linux/amd64` | 3× Whisper, 1× Canary-180m-Flash | ~3 GB |
| CUDA | `psyb0t/talkies:latest-cuda` | `linux/amd64` | all seven | ~9 GB |

Why split the model list? Whisper and the tiny Canary work fine on CPU. Parakeet-TDT, Canary-1B-Flash, and Canary-Qwen-2.5B don't — Parakeet-TDT is awkward on CPU because its decoder is autoregressive and slow without batched-attention kernels, Canary-1B and Canary-Qwen are flat-out too big to be useful in software-only inference. Rather than ship a CPU image that *technically* serves models nobody would use on CPU, the CPU image only lists what'll actually finish in a sane time.

Both images are amd64-only — `nemo_toolkit[asr]` and `faster-whisper` have aarch64 wheels for some of the chain but the full stack doesn't currently resolve cleanly on arm64 at the pinned versions. If you need arm64, file an issue with your specific use case.

The CUDA image also runs on CPU if `--gpus all` isn't passed — it'll bind to CPU, ignore the CUDA env vars, and refuse the GPU-only slugs at first call. Useful for debugging without a GPU host.

## Architecture

```
        client (curl / openai-py / litellm / whatever)
                          │
                  POST /v1/audio/transcriptions
                          │
                          ▼
              ┌─────────────────────┐
              │   FastAPI server    │
              │   (talkies/server)  │
              └──────────┬──────────┘
                         │ ffmpeg → 16 kHz mono WAV
                         ▼
              ┌─────────────────────┐
              │   Silero VAD        │  (if duration > 30s)
              │   (talkies/vad)     │
              └──────────┬──────────┘
                         │  list of (start, end) speech regions
                         ▼
              ┌─────────────────────┐
              │   Backend dispatch  │
              │   (talkies/models)  │
              └────┬────┬────┬─────┘
                   │    │    │
        ┌──────────┘    │    └──────────┐
        ▼               ▼               ▼
  faster-whisper     NeMo TDT      NeMo Canary
  (CTranslate2)     (parakeet)     (multitask / SALM)
  whisper-*         parakeet-*     canary-*
```

- **`talkies/audio.py`** — uses ffmpeg under the hood (`subprocess.run`, no python-ffmpeg overhead) to normalize any input format to 16 kHz mono WAV. Stereo diarization mode splits to two mono WAVs, one per channel.
- **`talkies/vad.py`** — wraps Silero VAD. Returns merged speech regions capped at `TALKIES_VAD_MAX_SPEECH` seconds (regions longer than the cap get re-split at the longest silence inside).
- **`talkies/models/`** — one module per engine family. Each implements the `Backend` ABC: `get_model()` (lazy load), `transcribe(wav_path, source_lang, target_lang, task, with_timestamps)` (return a `TranscribeResult`), `unload()` (free RAM/VRAM), `loaded()`, `last_used_secs_ago()`.
  - `whisper.py` — drives faster-whisper.
  - `parakeet.py` — drives NeMo Parakeet-TDT.
  - `multitask.py` — drives Canary-180M-Flash + Canary-1B-Flash.
  - `salm.py` — drives Canary-Qwen-2.5B (SALM head with Qwen2 decoder).
  - `base.py` — common types (`TranscribeResult`).
  - `__init__.py` — `build_backends(registry, device)` factory.
- **`talkies/config.py`** — env-driven config, parsed at import time. Bad input fails the container, doesn't ship a half-broken service.

All seven backends compete for the same VRAM/RAM. The server enforces "one model loaded at a time" via sibling eviction on every transcription request; the idle sweeper unloads anything that hasn't been used in `TALKIES_MODEL_TTL`. This matches the "single-GPU host, multiple-model registry, one model resident at a time" assumption that Ollama makes and that most self-hosted ASR setups actually want.

## Customizing the model registry

The image ships with `models.json` (CUDA) or `models-cpu.json` (CPU) baked in. You can override the registry without rebuilding by bind-mounting your own:

```bash
docker run -d --name talkies \
  -v $HOME/talkies-models:/data \
  -v $PWD/my-models.json:/app/models.json:ro \
  -p 8000:8000 \
  psyb0t/talkies:latest
```

Or point `TALKIES_MODELS_FILE` at a different path inside the container. The file structure:

```json
{
  "models": {
    "your-slug": {
      "repo": "huggingface-org/repo-name",
      "executor": "whisper",
      "default_source_lang": "en",
      "default_target_lang": "en",
      "default_task": "asr",
      "languages": ["en"]
    }
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `repo` | yes | HuggingFace repo id. talkies pulls via `snapshot_download(local_dir=$TALKIES_DATA_DIR/models/<slug>)` so each model lives as a flat directory keyed by its slug. |
| `executor` | yes | One of `whisper`, `parakeet`, `canary_multitask`, `canary_salm`. Other values fail startup. |
| `default_source_lang` | no | Used when the request omits `language`. |
| `default_target_lang` | no | Used by Canary multitask for translation tasks. |
| `default_task` | no | `asr` (transcribe) or `s2t_translation` (Canary multitask only). Default `asr`. |
| `languages` | no | Informational only — listed in error messages, not enforced. |
| `dependencies` | no | List of extra HuggingFace repo ids the executor needs at load time (e.g. `canary-qwen-2.5b` instantiates a Qwen3 tokenizer separately from its own snapshot). Each is `snapshot_download`'d at entrypoint time into the standard HF cache (`HF_HOME`) so `transformers`/`AutoTokenizer` find it offline. |

Adding a new slug pointing at a new repo "just works" if the repo follows the same conventions as the executor expects (a faster-whisper CT2 dir for `whisper`, a NeMo `.nemo` checkpoint for the others). Adding a brand-new executor family means editing `talkies/models/__init__.py` to register the dispatch.

A common reason to ship a custom `models.json`: enabling translation directions on Canary-1B-Flash. See [Translation](#translation-canary-xy).

## Development

Everything runs in a sandboxed dev container — your host needs only `docker`, `make`, `git`, and a shell.

```bash
make help            # list all targets
make dev-image       # build the dev container (run once, rebuilt on changes)
make lint            # flake8 + mypy inside the dev container
make format          # isort + black inside the dev container
make test            # unit tests inside the dev container (fast, offline, no GPU)

make build           # build CPU production image
make build-cuda      # build CUDA production image
make build-all       # both

make run             # build + run CPU image, weights cache to ~/.talkies-models
make run-cuda        # build + run CUDA image with --gpus all

make test-integration  # CUDA integration suite — builds + boots talkies, hits the HTTP surface
```

The dev image is intentionally light — it has the lightweight runtime deps (`fastapi`, `pydantic`, etc.) plus lint/format/test tools, but no torch, no nemo_toolkit, no faster-whisper. Those are multi-GB and CPU/CUDA-variant-specific; pulling them just to lint would be silly. The full ML stack lives only in the production images.

### Unit tests (`make test`)

Pure-python coverage of `talkies.config` — `TALKIES_ENABLED_MODELS` parsing + filtering, schema validation in `load_registry()`, env-var coercion (durations, device strings). No model loading, no HTTP, runs in sub-second inside the dev container.

### Integration tests (`make test-integration`)

CUDA-only end-to-end suite that builds `psyb0t/talkies:local-cuda`, spawns a fresh container with `--gpus all`, waits for `/healthz`, and runs every `tests/integration/test_*.sh` against the running service:

- Endpoint smoke (`test_endpoints.sh`): `/healthz`, `/v1/models`, `/api/ps`, `/unload`, 404/422 paths.
- Per-model transcription (`test_transcribe.sh`): every enabled model goes through `json`, `verbose_json`, `srt`, and `vtt` against a fixture audio file. Also asserts `/api/ps` reflects loads and that `DELETE /api/ps/<slug>` actually unloads.

Drop a short clip (a few seconds is plenty) at `tests/integration/.fixtures/audio.<wav|mp3|m4a|flac|ogg>` — the harness picks it up automatically; the transcription tests skip if it's missing.

Env knobs:

| Var | Default | Effect |
|---|---|---|
| `TALKIES_TEST_PORT` | `18000` | Host port to publish. |
| `TALKIES_TEST_CACHE` | `~/.talkies-models` | Bind-mounted to `/data` so weights persist across runs. |
| `TALKIES_TEST_IMAGE` | `psyb0t/talkies:local-cuda` | Image under test. |
| `TALKIES_SKIP_BUILD` | (unset) | Set to `1` to skip `make build-cuda` and reuse what's tagged. |
| `TALKIES_TEST_KEEP` | (unset) | Set to `1` to leave the test container running on exit (for `docker logs` / manual poking). |
| `TALKIES_ENABLED_MODELS` | (unset → all) | Comma slugs to restrict the test surface (also what gets downloaded on first boot). |
| `TALKIES_READY_TIMEOUT` | `1800` | Seconds to wait for `/healthz` (the default tolerates a cold cache pulling every model on a single GPU). |

You can pass test names as args to run a subset, e.g.: `bash tests/integration/run.sh test_talkies_healthz test_talkies_models_list`.

CPU isn't supported as a test target on purpose — whisper-large-v3 on a desktop CPU is a half-hour-per-clip operation, useless as a regression gate. If you only have a CPU host, run `make test` (unit) and exercise the service manually via the [Quick start](#quick-start) curl loop.

## Security notes

- Every Python dependency is exactly pinned in the Dockerfiles. No floating constraints.
- Base images pinned by `@sha256:...` digest (Python 3.12-slim-bookworm for CPU, nvidia/cuda:12.6.3-runtime-ubuntu24.04 for CUDA).
- `uv` itself is COPY'd from `ghcr.io/astral-sh/uv:0.11.15` by digest.
- `[tool.uv] exclude-newer` in `pyproject.toml` refuses to install package versions newer than the gate date — blocks same-day supply-chain attacks at lockfile generation time.
- Container runs as non-root user `talkies` (uid 1000). `/data` is the only writable mount target.
- `HF_HUB_OFFLINE=1` is the production default — once weights are cached on disk, the container has no reason to call out to HuggingFace. The entrypoint's prefetch step transparently unsets this for the snapshot-download sub-shell only; the server process itself runs offline. So in steady state (after the first boot) talkies never reaches the internet.
- Optional built-in bearer-token auth via `TALKIES_AUTH_TOKEN` (see [Bearer-token auth](#bearer-token-auth)). Default-off — set the env var to require `Authorization: Bearer <token>` on every route (HTTP API and MCP). The server binds to `0.0.0.0:8000` inside the container — control network exposure at `docker run` time (`-p 127.0.0.1:8000:8000` for loopback-only on the host, `-p 8000:8000` for all interfaces). For untrusted networks, combine the token with a reverse proxy doing TLS termination + rate limiting.

Open CVEs against the pinned `torch` / `transformers` / `nemo-toolkit` versions are threat-modelled in the Dockerfile.cuda header comments — short version, talkies never calls `torch.load()` on untrusted files (weights come from hardcoded HF org repos via `snapshot_download`), never instantiates `Trainer` (inference only), and never runs the per-model conversion paths flagged by the transformers advisories. If you point `TALKIES_DATA_DIR` at a directory containing **arbitrary user-provided model weights**, you're on your own — talkies' auto-fetch only writes to `$TALKIES_DATA_DIR/models/<slug>/` from the hardcoded HuggingFace repo ids in `models.json`.

Run `osv-scanner` against the image if you want a fresh advisory check before deploying.

## License

WTFPL — Do What The Fuck You Want To Public License. See `LICENSE`.
