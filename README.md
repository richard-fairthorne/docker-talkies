# talkies

[![Docker Pulls](https://img.shields.io/docker/pulls/psyb0t/talkies?style=flat-square)](https://hub.docker.com/r/psyb0t/talkies)
[![Docker Hub](https://img.shields.io/docker/v/psyb0t/talkies?sort=semver&label=Docker%20Hub&style=flat-square)](https://hub.docker.com/r/psyb0t/talkies)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg?style=flat-square)](http://www.wtfpl.net/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square)](https://www.python.org/downloads/)

> **Self-hosted, OpenAI-compatible speech server.** 7 ASR backends, 2 TTS engines, voice cloning, MCP — one Docker container, one wire format.

```python
# Drop-in: point your existing OpenAI client at it, change the slug.
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

c.audio.transcriptions.create(model="whisper-large-v3-turbo", file=open("a.mp3", "rb"))
c.audio.speech.create(model="qwen3-tts-0.6b", voice="alloy", input="hello").stream_to_file("out.mp3")
```

The same client you use against `api.openai.com` works here — only the base URL and the slug change. That's the entire story.

- **6 ASR backends** — Whisper (v3 / turbo), Parakeet-TDT, Canary-180M-Flash / 1B-Flash / Canary-Qwen-2.5B. Whisper-shape response across all of them; long files get sliced via Silero VAD and stitched back.
- **2 TTS engines, 3 backends** — Kokoro-82M (~41 voices across en/es/fr/hi/it/pt, sub-second on CPU) shipped in two flavors (`kokoro-82m` PyTorch and `kokoro-82m-nvidia` ONNX-via-ORT — NVIDIA's TensorRT-friendly export), plus Qwen3-TTS-0.6B (CUDA-only voice cloning).
- **Voice cloning** — drop a 10-30 s reference `.wav` into `/data/custom-voices/<name>.wav`, synth as `voice=<name>`. Nested paths preserved (`clients/acme/jane.wav` → `voice=clients/acme/jane`). Live re-scan, no restart.
- **Hot model swap + idle eviction** — one GPU pool serves both modalities, Ollama-style `/api/ps` for introspection, `DELETE /api/ps/<slug>` to evict.
- **MCP server built in** at `/v1/mcp` — Claude / Cursor / IDE-side LLMs can call transcribe + speak as tools.
- **Stereo diarization** without bolting on a separate model — left channel = speaker L, right = speaker R, chronological `L:` / `R:` turn lines.
- **CPU + CUDA images** — `psyb0t/talkies:latest` (CPU + Kokoro + 4 ASR models) and `:latest-cuda` (everything, ~11 GB VRAM at full load).

## Quick start

```bash
docker run -d --name talkies \
  -v $HOME/talkies-data:/data \
  -p 8000:8000 \
  psyb0t/talkies:latest

curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file=@samples/hello.wav" \
  -F "model=whisper-large-v3-turbo" | jq
```

First boot downloads every model in `models.json` into `/data/models/<slug>/` (75 MB-3 GB each — bind-mount `/data` so they survive restarts). Restrict the set with `-e TALKIES_ENABLED_MODELS=whisper-large-v3-turbo,canary-180m-flash`. GPU: pull `psyb0t/talkies:latest-cuda` and add `--gpus all`.

<details>
<summary><b>More <code>curl</code> examples</b> — verbose JSON, SRT, stereo diarization, TTS, model management</summary>

```bash
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

# Kokoro TTS — list the shipped voices, then synthesize an MP3.
curl -s http://localhost:8000/v1/audio/voices | jq
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-82m","input":"Hello from talkies.","voice":"af_heart","response_format":"mp3"}' \
  --output hello.mp3

# Which models are configured, which are loaded, evict one if you want.
curl -s http://localhost:8000/v1/models | jq
curl -s http://localhost:8000/api/ps | jq
curl -s -X DELETE "http://localhost:8000/api/ps/whisper-large-v3-turbo"
curl -s -X POST  http://localhost:8000/unload | jq    # evict everything
```

</details>

<details>
<summary><b>Table of contents</b></summary>

- [Quick start](#quick-start)
- [How it works](#how-it-works)
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
- [API — `POST /v1/audio/speech` (TTS)](#api--post-v1audiospeech-tts)
  - [Request body](#request-body)
  - [Voices (`GET /v1/audio/voices`)](#voices-get-v1audiovoices)
  - [Output formats](#output-formats)
  - [Error contract (TTS)](#error-contract-tts)
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
- [Credits](#credits)
- [License](#license)

</details>

## How it works

`POST /v1/audio/transcriptions` with a multipart `file` + a `model` slug → text back. `POST /v1/audio/speech` with a JSON body (`model` + `input` + `voice`) → audio bytes back. Same wire shape as OpenAI for both.

Swap the ASR slug — `whisper-large-v3`, `whisper-large-v3-turbo`, `parakeet-tdt-0.6b-v3`, `canary-180m-flash`, `canary-1b-flash`, `canary-qwen-2.5b` — and the transcription contract stays identical. Behind the scenes the request is dispatched to the right backend (faster-whisper for the whisper family, NeMo for everything else), audio is normalized to 16 kHz mono WAV, long files are sliced via Silero VAD into ≤28-second speech regions, results are stitched back into one Whisper-shape timeline. None of that leaks into the wire shape. You just get text.

For TTS there are three slugs:

- `kokoro-82m` (Kokoro-82M, Apache 2.0, ~41 voices across en/es/fr/hi/it/pt) — the fast in-process PyTorch pipeline (via the `kokoro` PyPI lib + `misaki` G2P). Sub-second synthesis on CPU and trivial on GPU.
- `kokoro-82m-nvidia` (nvidia/kokoro-82M-onnx-opt, Apache 2.0, same voice catalog) — same Kokoro weights served via ONNXRuntime against NVIDIA's TensorRT-friendly ONNX export. No PyTorch on the hot path; CUDA EP on the CUDA image, CPU EP on the CPU image. G2P via `espeak-ng` + `phonemizer` (no `misaki` dep). Drop-in for `kokoro-82m` — same `voice` names, same wire format, same defaults.
- `qwen3-tts-0.6b` (Qwen3-TTS-12Hz-0.6B-Base, Apache 2.0, CUDA only) — voice cloning. Bring your own reference `.wav` (10-30 s of clean speech is plenty), drop it into `/data/custom-voices/<your-name>.wav`, and synthesize in that speaker's voice via `voice=<your-name>`. Supports nested paths (`/data/custom-voices/clients/acme/jane.wav` → `voice=clients/acme/jane`). Three sample voices (`alloy`, `echo`, `fable`) ship baked into the image.

Pass `model=<slug>`, an `input` string, and a `voice` from `GET /v1/audio/voices` — the server runs the matching backend's pipeline, encodes the raw PCM into your requested `response_format` (`mp3` / `opus` / `aac` / `flac` / `wav` / `pcm`) via ffmpeg, and streams the bytes back with the matching `Content-Type`.

Need stereo speaker diarization on transcription? Pass `diarization=true` and upload a stereo file — left channel = speaker L, right channel = speaker R, output gets per-segment `channel` tags and the text is split into chronological `L:` / `R:` turn lines. Two-mic setups (interview rigs, podcast splits, dual-track ham recordings) end up with a clean transcript without you having to bolt a separate diarization model onto your stack.

GPU variant (`psyb0t/talkies:latest-cuda` + `--gpus all`) ships everything; the CPU image (`psyb0t/talkies:latest`) ships the four ASR models that actually run reasonably without a GPU (the three Whisper variants + `canary-180m-flash`) plus Kokoro TTS. Parakeet-TDT, Canary-1B-Flash, Canary-Qwen-2.5B, and Qwen3-TTS need VRAM to be anything other than a space heater, so they're CUDA-only. Kokoro is fast enough on CPU that it ships in both images.

## Supported models

Seven ASR models + three TTS slugs (two engines, since `kokoro-82m` and `kokoro-82m-nvidia` are the same weights via different runtimes), all publicly available with permissive licenses. They split into five engine families:

### ASR (`POST /v1/audio/transcriptions`)

| Slug | HF repo | Family | Image | Languages | License |
|---|---|---|---|---|---|
| `whisper-large-v3` | `Systran/faster-whisper-large-v3` | faster-whisper (CTranslate2) | CPU + CUDA | 99 (auto-detect) | MIT |
| `whisper-large-v3-turbo` | `deepdml/faster-whisper-large-v3-turbo-ct2` | faster-whisper (CTranslate2) | CPU + CUDA | 99 (auto-detect) | MIT |
| `parakeet-tdt-0.6b-v3` | `nvidia/parakeet-tdt-0.6b-v3` | NeMo (TDT) | CUDA only | English | CC-BY-4.0 |
| `canary-180m-flash` | `nvidia/canary-180m-flash` | NeMo Canary (multitask) | CPU + CUDA | English (ASR only on this size) | CC-BY-4.0 |
| `canary-1b-flash` | `nvidia/canary-1b-flash` | NeMo Canary (multitask) | CUDA only | en, de, fr, es (ASR + X→en / en→X translation) | CC-BY-4.0 |
| `canary-qwen-2.5b` | `nvidia/canary-qwen-2.5b` | NeMo Canary SALM (Qwen2 decoder) | CUDA only | English | CC-BY-4.0 |

Both Whisper variants are tokenized + executed through [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which is roughly 4× faster than the reference OpenAI implementation at the same accuracy on the same hardware. The four NVIDIA models go through NeMo's native inference path — Parakeet uses the TDT decoder, Canary models use the multitask transformer head, Canary-Qwen swaps the decoder for a Qwen2 LLM (the "speech-augmented language model" trick that lets you tack instructions onto the prompt).

### TTS (`POST /v1/audio/speech`)

| Slug | HF repo | Family | Image | Languages | License |
|---|---|---|---|---|---|
| `kokoro-82m` | `hexgrad/Kokoro-82M` | Kokoro (PyTorch in-process, 24 kHz) | CPU + CUDA | en (American + British), es, fr, hi, it, pt | Apache 2.0 |
| `kokoro-82m-nvidia` | `nvidia/kokoro-82M-onnx-opt` | Kokoro (ONNX via ORT, 24 kHz) | CPU + CUDA | en (American + British), es, fr, hi, it, pt | Apache 2.0 |
| `qwen3-tts-0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3-TTS (faster-qwen3-tts, CUDA graphs) | CUDA only | 17 languages (en, zh, ja, ko, fr, de, es, it, pt, ru, vi, th, id, ar, tr, pl, nl) | Apache 2.0 |

Kokoro-82M is an 82-million-parameter open-weight TTS model. It runs in-process via the [`kokoro`](https://pypi.org/project/kokoro/) PyPI package — no separate sidecar — and is fast enough on a 4-core CPU to be useful, so it ships in both images. The server exposes Kokoro's native voice naming (`af_heart`, `bm_george`, `ef_dora`, …) directly; there's no OpenAI alias mapping for that slug. Discover voices via `GET /v1/audio/voices`.

`kokoro-82m-nvidia` is the same Kokoro-82M weights served via NVIDIA's [TensorRT-friendly ONNX export](https://huggingface.co/nvidia/kokoro-82M-onnx-opt) (Apache 2.0, released by the NVIDIA NeMo Speech team in May 2026). It uses ONNXRuntime with the CUDA execution provider on the CUDA image and the CPU EP on the CPU image — no PyTorch on the inference hot path. G2P is `espeak-ng` via `phonemizer` instead of `misaki`. Same 40-voice catalog as `kokoro-82m`, same wire format, same defaults (`af_heart`). Pick this slug when you want the ONNX/ORT execution path; pick `kokoro-82m` when you want the misaki-driven G2P quality (slightly better pronunciation on tricky English words).

Qwen3-TTS-0.6B is a 600-million-parameter voice-cloning model. It runs via [`faster-qwen3-tts`](https://pypi.org/project/faster-qwen3-tts/) (MIT-licensed wrapper that captures CUDA graphs around the talker + predictor heads for sub-second synthesis after a one-time ~30-60s warmup). The voice catalog comes from two on-disk dirs that both contribute `.wav` files (each one becomes a clone-target voice):

- `/opt/talkies/qwen3-voices/` (baked into the image) — ships `alloy`, `echo`, `fable` as a starter set.
- `/data/custom-voices/` (your data volume) — drop your own `.wav` files in. Nested subdirs are preserved: `/data/custom-voices/clients/acme/jane.wav` becomes voice `clients/acme/jane`. Custom voices shadow builtins with the same name.

Each `<name>.wav` should have a sibling `<name>.txt` (transcript of what the speaker says in the reference) for in-context-learning (ICL) clone mode — the model produces noticeably better fidelity with a faithful transcript. If the `.txt` is missing the backend falls back to x-vector-only mode automatically (lower quality, but still produces audio) and logs a warning. Optional sibling `<name>.lang` is the language label, defaults to `English`. `GET /v1/audio/voices` returns an `origin: "builtin" | "custom"` field for each Qwen3 voice so a UI can tell baked-in samples from user-supplied clones at a glance.

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
| **OpenAI voice aliases for Kokoro (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`)** | 400 | Kokoro exposes its native voice names only. (Note: `alloy` / `echo` / `fable` exist as `qwen3-tts-0.6b` voices — different model, different catalog. They're not aliases for Kokoro voices.) Discover voices via `GET /v1/audio/voices`. Map client-side if your stack hard-codes the OpenAI names against Kokoro. |
| **Japanese (`j*`) and Chinese (`z*`) Kokoro voices** | Filtered out | Those voices need the optional `misaki[ja]` / `misaki[zh]` extras, which pull large MeCab / pypinyin chains. The voice catalog only exposes the 40 voices whose lang codes work with the lightweight `espeak-ng`-based G2P shipped in the image. Same filter applies to `kokoro-82m-nvidia` — the NVIDIA snapshot ships zh-specific lexicons + FSTs that would unlock those voices, but the bundled frontend isn't yet wired into the backend. (Qwen3-TTS does support Japanese / Chinese / Korean — pick that slug instead.) |
| **TTS streaming output** | Not supported | The whole utterance is synthesized + encoded, then the full response body is returned. No SSE, no chunked audio. For long inputs split client-side. |
| **TTS `instructions` field on Kokoro** | Accepted, ignored | Both kokoro slugs (PyTorch + ONNX) take no instruction-prompt input — `voice` is the only style control. The field is accepted for OpenAI parity and silently dropped. Qwen3-TTS *does* honour `instructions` — see the row below. |
| **TTS `instructions` field on Qwen3-TTS** | Honoured | Passed through as the `instruct` parameter on `generate_voice_clone`. Works best when the reference voice has a sibling `.txt` transcript (ICL mode); with no transcript the backend falls back to x-vector-only synthesis and logs a warning telling you to add the `.txt` for better fidelity. |
| **TTS `speed` on Qwen3-TTS** | Accepted, ignored | Qwen3-TTS has no speed-control parameter. The field is still validated against `[0.25, 4.0]` for compatibility but the backend ignores anything other than `1.0`. Kokoro applies `speed` as documented. |
| **TTS `speed` outside `[0.25, 4.0]`** | Clamped | Values outside the OpenAI-documented range are silently clamped (applies to Kokoro; Qwen3-TTS ignores `speed` regardless). |
| **Qwen3-TTS on CPU** | 400 / startup error | `faster-qwen3-tts` captures CUDA graphs at load time; there's no CPU path. The CPU image (`psyb0t/talkies:latest`) doesn't include the Qwen3-TTS dependencies at all — only the CUDA image (`psyb0t/talkies:latest-cuda`) does. |

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

## API — `POST /v1/audio/speech` (TTS)

JSON body. Same field names as OpenAI's speech endpoint. Returns the encoded audio bytes in the body with the matching `Content-Type` (no JSON envelope).

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "kokoro-82m",
        "input": "The quick brown fox jumps over the lazy dog.",
        "voice": "af_heart",
        "response_format": "mp3",
        "speed": 1.0
      }' \
  --output fox.mp3
```

### Request body

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | TTS model slug — one of `kokoro-82m`, `kokoro-82m-nvidia`, `qwen3-tts-0.6b`. Unknown slug → 404. ASR slug → 400 (wrong endpoint). |
| `input` | yes | — | Text to synthesize. Empty / whitespace-only → 400. No fixed length cap; for very long inputs split client-side and concatenate the resulting audio. |
| `voice` | no | model `default_voice` (`af_heart` for either kokoro slug, `alloy` for qwen3) | Voice name from `GET /v1/audio/voices` for the chosen model. Unknown → 400 with the catalog listed. Voices are not interchangeable across models — each engine owns its own catalog. |
| `response_format` | no | `mp3` | One of `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`. See [Output formats](#output-formats). |
| `speed` | no | `1.0` | Playback rate. Clamped to `[0.25, 4.0]`. Kokoro supports speed control; Qwen3-TTS does not — non-1.0 values are silently ignored for that backend. |
| `instructions` | no | — | Free-form style prompt. **Honoured by Qwen3-TTS** (threaded into the model as `instruct`; best with a `.txt` reference transcript for ICL mode). **Ignored by both kokoro slugs** — Kokoro takes no instruction-prompt input. Accepted on every TTS slug for OpenAI parity. |

### Voices (`GET /v1/audio/voices`)

Returns the catalog of voices the server can synthesize, across all loaded-or-loadable TTS models:

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

```json
{
  "voices": [
    {"voice": "af_heart",  "model": "kokoro-82m", "default": true},
    {"voice": "af_alloy",  "model": "kokoro-82m", "default": false},
    {"voice": "am_adam",   "model": "kokoro-82m", "default": false},
    {"voice": "bf_emma",   "model": "kokoro-82m", "default": false},
    {"voice": "bm_george", "model": "kokoro-82m", "default": false},
    {"voice": "ef_dora",   "model": "kokoro-82m", "default": false},
    {"voice": "ff_siwis",  "model": "kokoro-82m", "default": false},
    {"voice": "hf_alpha",  "model": "kokoro-82m", "default": false},
    {"voice": "if_sara",   "model": "kokoro-82m", "default": false},
    {"voice": "pf_dora",   "model": "kokoro-82m", "default": false},
    {"voice": "alloy",                "model": "qwen3-tts-0.6b", "default": true,  "origin": "builtin"},
    {"voice": "echo",                 "model": "qwen3-tts-0.6b", "default": false, "origin": "builtin"},
    {"voice": "fable",                "model": "qwen3-tts-0.6b", "default": false, "origin": "builtin"},
    {"voice": "clients/acme/jane",    "model": "qwen3-tts-0.6b", "default": false, "origin": "custom"}
  ]
}
```

Kokoro voice names encode `<lang_code><gender>_<name>`:

| Prefix | Language |
|---|---|
| `af_` / `am_` | American English (female / male) |
| `bf_` / `bm_` | British English (female / male) |
| `ef_` / `em_` | Spanish |
| `ff_` | French |
| `hf_` / `hm_` | Hindi |
| `if_` / `im_` | Italian |
| `pf_` / `pm_` | Portuguese (Brazilian) |

41 Kokoro voices ship in the image. The Japanese (`jf_*` / `jm_*`) and Chinese (`zf_*` / `zm_*`) voices in Kokoro's upstream voice pack are filtered out because they require the optional `misaki[ja]` / `misaki[zh]` extras (MeCab + pypinyin chains) which would add hundreds of MB to the image for languages most users don't need.

Qwen3-TTS voice names come from the on-disk catalog (see [Supported models → TTS](#tts-post-v1audiospeech)). Three builtin voices (`alloy`, `echo`, `fable`) ship in the image. Drop your own `.wav` reference samples into `/data/custom-voices/` (the host mount) and they show up tagged `origin: "custom"`. The path of the wav relative to that dir, with the `.wav` stripped, is the voice name — so `/data/custom-voices/clients/acme/jane.wav` becomes `clients/acme/jane`. A custom voice with the same name as a builtin shadows the builtin.

To improve clone fidelity, drop a sibling `<name>.txt` (transcript of what the speaker is saying in the reference audio) and optionally `<name>.lang` (one of `English`, `Chinese`, `Japanese`, `Korean`, `French`, `German`, `Spanish`, `Italian`, `Portuguese`, `Russian`, `Vietnamese`, `Thai`, `Indonesian`, `Arabic`, `Turkish`, `Polish`, `Dutch`; defaults to `English`). Reference audio should be 10-30 seconds of clean speech in the target speaker's voice — no music, minimal background noise, single speaker.

### Output formats

`response_format` picks the encoder applied to the raw 24 kHz PCM Kokoro emits. ffmpeg does the conversion in-process; no temp files.

| `response_format` | Content-Type | Codec / container | Notes |
|---|---|---|---|
| `mp3` (default) | `audio/mpeg` | libmp3lame, 128 kbps CBR | Most universal. Plays everywhere. |
| `opus` | `audio/ogg` | libopus, 64 kbps VBR, Ogg container | Best quality-per-byte for speech. |
| `aac` | `audio/aac` | AAC-LC, 128 kbps, ADTS framing | iOS-friendly. |
| `flac` | `audio/flac` | FLAC | Lossless. ~3-5× the size of opus. |
| `wav` | `audio/wav` | PCM s16le, 24 kHz, mono, RIFF header | Lossless, largest. |
| `pcm` | `application/octet-stream` | Raw PCM s16le, 24 kHz, mono — no container, no header | For real-time chaining into another encoder. Caller is expected to know the sample rate / format. |

### Error contract (TTS)

Same envelope as the transcription endpoint — application errors as `{"detail": "..."}`, Pydantic validation as the structured 422 array.

| Status | When |
|---|---|
| 200 | success (audio bytes in body) |
| 400 | empty `input`, unknown `voice`, unsupported `response_format`, model isn't a TTS backend (e.g. someone POSTed `whisper-large-v3` here) |
| 401 | `TALKIES_AUTH_TOKEN` set, missing / wrong bearer |
| 404 | unknown `model` slug |
| 422 | Pydantic validation (missing required fields, wrong types) |
| 500 | unhandled ffmpeg / kokoro / qwen3-tts internal failure |
| 503 | snapshot files missing under `${TALKIES_DATA_DIR}/models/<slug>/` (the model was excluded from `TALKIES_ENABLED_MODELS` at boot but is still being called); or `qwen3-tts-0.6b` requested with no voices on disk |

## Resource-management endpoints (Ollama-style)

talkies mirrors a subset of [speaches](https://github.com/speaches-ai/speaches) and Ollama's resource-management surface, so a single LiteLLM-style proxy can drive both:

| Endpoint | Behavior |
|---|---|
| `GET /healthz` | Unauthenticated liveness. Returns `{ok, device, models}` where `models` is the configured slug list. |
| `GET /v1/models` | OpenAI-style model list. `{"object": "list", "data": [{"id": slug, "modality": "asr"\|"tts", ...}]}`. The `modality` field is talkies-specific so clients can filter ASR vs TTS slugs. |
| `GET /api/ps` | Currently-loaded models, with per-model `idle_seconds` (seconds since last use). |
| `DELETE /api/ps/{model_id}` | Evict one model from RAM/VRAM. `model_id` can be URL-encoded (`whisper%2Flarge` → `whisper/large`) — LiteLLM's resource manager does this on slashes. Returns 404 if not loaded. |
| `POST /unload` | Evict every loaded model. Returns the list that was actually unloaded. |

Behind these endpoints there's an **idle sweeper** that runs on a `TALKIES_SWEEPER_INTERVAL` cadence (default 60s) and unloads any backend that hasn't been called in `TALKIES_MODEL_TTL` seconds (default 600s = 10min). Set `TALKIES_MODEL_TTL=0` to disable auto-unload entirely.

There's also **sibling eviction at request time**: when a transcription or speech request arrives and a model that isn't the requested one is currently loaded, the other model gets unloaded first — regardless of modality. ASR and TTS share the same pool; loading Kokoro evicts a resident Whisper and vice versa. All models compete for the same VRAM (or the same fat slice of RAM on CPU), so loading two at once on a 12GB card OOMs you. Ollama does this implicitly via its scheduler; we do it explicitly per-request. If you genuinely want two models resident simultaneously, you want two containers.

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
| CPU | `psyb0t/talkies:latest` | `linux/amd64` | 3× Whisper, 1× Canary-180m-Flash, Kokoro-82M | ~3 GB |
| CUDA | `psyb0t/talkies:latest-cuda` | `linux/amd64` | all seven ASR + Kokoro-82M + Qwen3-TTS-0.6B | ~11 GB |

Why split the model list? Whisper and the tiny Canary work fine on CPU. Parakeet-TDT, Canary-1B-Flash, Canary-Qwen-2.5B, and Qwen3-TTS-0.6B don't — Parakeet-TDT is awkward on CPU because its decoder is autoregressive and slow without batched-attention kernels, Canary-1B and Canary-Qwen are flat-out too big to be useful in software-only inference, and Qwen3-TTS via `faster-qwen3-tts` captures CUDA graphs at load time (no CPU code path exists). Rather than ship a CPU image that *technically* serves models nobody would use on CPU, the CPU image only lists what'll actually finish in a sane time. Kokoro-82M ships in both images — at 82M params it synthesizes faster than real-time on a 4-core CPU.

Both images are amd64-only — `nemo_toolkit[asr]` and `faster-whisper` have aarch64 wheels for some of the chain but the full stack doesn't currently resolve cleanly on arm64 at the pinned versions. If you need arm64, file an issue with your specific use case.

The CUDA image also runs on CPU if `--gpus all` isn't passed — it'll bind to CPU, ignore the CUDA env vars, and refuse the GPU-only slugs at first call. Useful for debugging without a GPU host (but `qwen3-tts-0.6b` will hard-fail at load time without CUDA).

## Architecture

```
        client (curl / openai-py / litellm / whatever)
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
  POST /v1/audio/    POST /v1/audio/    GET /v1/audio/
   transcriptions       speech              voices
       │                  │                  │
       ▼                  ▼                  ▼
              ┌─────────────────────┐
              │   FastAPI server    │
              │   (talkies/server)  │
              └──────────┬──────────┘
                         │
        ┌────────────────┴────────────────┐
        │ ASR path                        │ TTS path
        ▼                                 ▼
  ffmpeg → 16 kHz mono WAV         kokoro pipeline (24 kHz PCM)
        │                                 │
        ▼                                 │
  Silero VAD (if dur > 30s)               │
  (talkies/vad)                           │
        │                                 ▼
        ▼                          ffmpeg encode →
  Backend dispatch                 mp3 / opus / aac /
  (talkies/models)                 flac / wav / pcm
        │                          (talkies/tts)
   ┌────┼────┬─────┐                      │
   ▼    ▼    ▼     ▼                      ▼
 fw   TDT  Canary  Canary             Kokoro
 *    *    multitask SALM             (kokoro PyPI)
```

- **`talkies/audio.py`** — uses ffmpeg under the hood (`subprocess.run`, no python-ffmpeg overhead) to normalize any input format to 16 kHz mono WAV. Stereo diarization mode splits to two mono WAVs, one per channel.
- **`talkies/vad.py`** — wraps Silero VAD. Returns merged speech regions capped at `TALKIES_VAD_MAX_SPEECH` seconds (regions longer than the cap get re-split at the longest silence inside).
- **`talkies/tts.py`** — pipes Kokoro's raw 24 kHz mono int16 PCM through ffmpeg to produce the requested `response_format`. `pcm` short-circuits and returns the raw bytes verbatim.
- **`talkies/models/`** — one module per engine family. Each implements the duck-typed `BackendBase` Protocol: `get_model()` (lazy load), `unload()` (free RAM/VRAM), `loaded()`, `last_used_secs_ago()`. ASR backends additionally implement `transcribe(...)` returning a `TranscribeResult`; TTS backends implement `synthesize(...)` returning a `SynthesisResult`, plus `voices()` / `default_voice()`.
  - `whisper.py` — drives faster-whisper.
  - `parakeet.py` — drives NeMo Parakeet-TDT.
  - `multitask.py` — drives Canary-180M-Flash + Canary-1B-Flash.
  - `salm.py` — drives Canary-Qwen-2.5B (SALM head with Qwen2 decoder).
  - `kokoro.py` — drives Kokoro-82M (one shared `KModel`, per-lang `KPipeline`; reads voice tensors directly off the snapshot dir so the runtime stays `HF_HUB_OFFLINE=1`).
  - `base.py` — Protocols + result dataclasses (`TranscribeResult`, `SynthesisResult`).
  - `__init__.py` — `build_backends(registry, device)` factory + `is_asr_backend` / `is_tts_backend` duck-type guards.
- **`talkies/config.py`** — env-driven config, parsed at import time. Bad input fails the container, doesn't ship a half-broken service.

All backends compete for the same VRAM/RAM, ASR and TTS together. The server enforces "one model loaded at a time" via sibling eviction on every request (transcribe or synthesize); the idle sweeper unloads anything that hasn't been used in `TALKIES_MODEL_TTL`. This matches the "single-GPU host, multiple-model registry, one model resident at a time" assumption that Ollama makes and that most self-hosted speech setups actually want.

## Customizing the model registry

The image ships with `models.json` (CUDA) or `models-cpu.json` (CPU) baked in. You can override the registry without rebuilding by bind-mounting your own:

```bash
docker run -d --name talkies \
  -v $HOME/talkies-data:/data \
  -v $PWD/my-models.json:/app/models.json:ro \
  -p 8000:8000 \
  psyb0t/talkies:latest
```

Or point `TALKIES_MODELS_FILE` at a different path inside the container. The file structure:

```json
{
  "models": {
    "your-asr-slug": {
      "repo": "huggingface-org/repo-name",
      "executor": "whisper",
      "default_source_lang": "en",
      "default_target_lang": "en",
      "default_task": "asr",
      "languages": ["en"]
    },
    "your-tts-slug": {
      "repo": "huggingface-org/tts-repo-name",
      "executor": "kokoro",
      "modality": "tts",
      "default_voice": "af_heart",
      "languages": ["en"]
    }
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `repo` | yes | HuggingFace repo id. talkies pulls via `snapshot_download(local_dir=$TALKIES_DATA_DIR/models/<slug>)` so each model lives as a flat directory keyed by its slug. |
| `executor` | yes | One of `whisper`, `parakeet`, `canary_multitask`, `canary_salm`, `kokoro`. Other values fail startup. |
| `modality` | no | `asr` (default) or `tts`. Used by `/v1/models` filtering and by the endpoint guards (`/v1/audio/transcriptions` requires ASR; `/v1/audio/speech` requires TTS). The `kokoro` executor implies `tts`; the four ASR executors imply `asr`. |
| `default_source_lang` | no | ASR only. Used when the request omits `language`. |
| `default_target_lang` | no | ASR only. Used by Canary multitask for translation tasks. |
| `default_task` | no | ASR only. `asr` (transcribe) or `s2t_translation` (Canary multitask only). Default `asr`. |
| `default_voice` | no | TTS only. Used when the request omits `voice`. Defaults to the first voice the backend reports. |
| `languages` | no | Informational only — listed in error messages, not enforced. |
| `dependencies` | no | List of extra HuggingFace repo ids the executor needs at load time (e.g. `canary-qwen-2.5b` instantiates a Qwen3 tokenizer separately from its own snapshot). Each is `snapshot_download`'d at entrypoint time into the standard HF cache (`HF_HOME`) so `transformers`/`AutoTokenizer` find it offline. |

Adding a new slug pointing at a new repo "just works" if the repo follows the same conventions as the executor expects (a faster-whisper CT2 dir for `whisper`, a NeMo `.nemo` checkpoint for `parakeet`/`canary_*`, a Kokoro-style `config.json` + `kokoro-v*.pth` + `voices/*.pt` layout for `kokoro`). Adding a brand-new executor family means editing `talkies/models/__init__.py` to register the dispatch.

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

make run             # build + run CPU image, /data persisted at ~/.talkies-data
make run-cuda        # build + run CUDA image with --gpus all

make test-integration  # CUDA integration suite — builds + boots talkies, hits the HTTP surface

# Dependency management (bumps [tool.uv] exclude-newer to today first, then
# runs the uv operation inside the dev container — see "Security notes" below)
make pkg-lock                 # refresh uv.lock honouring the current gate
make pkg-add PKG=name[==ver]  # add a package
make pkg-update PKG=name      # upgrade ONE package to its latest allowed version
make pkg-upgrade              # upgrade EVERYTHING to its latest allowed version
make pkg-remove PKG=name      # remove a package
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
| `TALKIES_TEST_CACHE` | `~/.talkies-data` | Bind-mounted to `/data` so models / voices / files persist across runs. |
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
- `[tool.uv] exclude-newer` in `pyproject.toml` refuses to install package versions newer than the gate date — blocks same-day supply-chain attacks at lockfile generation time. Every `make pkg-*` dep mutation (`pkg-add`, `pkg-update`, `pkg-upgrade`, `pkg-remove`) bumps the gate to today's UTC midnight FIRST via `scripts/bump_exclude_newer.sh`, so the age window stays anchored to the moment of the change instead of silently drifting.
- Container runs as non-root user `talkies` (uid 1000). `/data` is the only writable mount target.
- `HF_HUB_OFFLINE=1` is the production default — once weights are cached on disk, the container has no reason to call out to HuggingFace. The entrypoint's prefetch step transparently unsets this for the snapshot-download sub-shell only; the server process itself runs offline. So in steady state (after the first boot) talkies never reaches the internet.
- Optional built-in bearer-token auth via `TALKIES_AUTH_TOKEN` (see [Bearer-token auth](#bearer-token-auth)). Default-off — set the env var to require `Authorization: Bearer <token>` on every route (HTTP API and MCP). The server binds to `0.0.0.0:8000` inside the container — control network exposure at `docker run` time (`-p 127.0.0.1:8000:8000` for loopback-only on the host, `-p 8000:8000` for all interfaces). For untrusted networks, combine the token with a reverse proxy doing TLS termination + rate limiting.

Open CVEs against the pinned `torch` / `transformers` / `nemo-toolkit` versions are threat-modelled in the Dockerfile.cuda header comments — short version, talkies never calls `torch.load()` on untrusted files (weights come from hardcoded HF org repos via `snapshot_download`), never instantiates `Trainer` (inference only), and never runs the per-model conversion paths flagged by the transformers advisories. If you point `TALKIES_DATA_DIR` at a directory containing **arbitrary user-provided model weights**, you're on your own — talkies' auto-fetch only writes to `$TALKIES_DATA_DIR/models/<slug>/` from the hardcoded HuggingFace repo ids in `models.json`.

Run `osv-scanner` against the image if you want a fresh advisory check before deploying.

## Credits

Inspired by [speaches](https://github.com/speaches-ai/speaches) — the OpenAI-compatible wire shape, the `/v1/models` + `/api/ps` resource-management surface, and the "one container, multiple speech backends" packaging idea all come from there. talkies is a sibling project, not a fork: different backend mix (NeMo Canary/Parakeet + faster-whisper for ASR, Kokoro-82M for TTS), different model-loading strategy (flat per-slug snapshot directories vs HF cache), CPU + CUDA images, and a few extras (stereo diarization, MCP endpoint, bearer auth, URL `file_path` fetching).

TTS uses [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via the [`kokoro`](https://pypi.org/project/kokoro/) PyPI package — Apache 2.0, 82M params, in-process. No sidecar.

## License

WTFPL — Do What The Fuck You Want To Public License. See `LICENSE`.
