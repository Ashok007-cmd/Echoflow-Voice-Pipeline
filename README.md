# EchoFlow Voice Pipeline

> Real-time streaming voice assistant — Microphone → ASR → LLM → TTS → Speaker

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

EchoFlow is a production-grade, asyncio-based voice assistant pipeline that chains speech recognition, large language models, and speech synthesis into a single low-latency streaming loop. LLM tokens stream directly into TTS as they arrive — audio begins playing before the full response is generated.

---

## Architecture

```
Microphone ──► ASR Engine ──► LLM Client ──► TTS Engine ──► Speaker
                  │               │               │
                  ▼               ▼               ▼
             Transcription    Streaming       Audio Chunks
             (utterance)      Tokens          (sentence-split)
                  │               │               │
                  └───────────────┴───────────────┘
                                  │
                          Latency Tracker
                        (per-stage metrics,
                         SQLite + matplotlib)
```

Three concurrent async tasks run per conversational turn — an **LLM worker**, a **TTS synthesizer worker**, and an **audio playback worker** — connected via `asyncio.Queue`. Each stage begins as soon as the prior stage produces output; no stage waits for the previous one to finish.

---

## Features

### Streaming Pipeline
- Concurrent ASR → LLM → TTS execution via asyncio task queues
- Sentence-boundary splitting: TTS begins on the first complete sentence while LLM is still generating tokens
- Dynamic VAD (Voice Activity Detection) using RMS energy + Zero Crossing Rate
- Interrupt support: new speech cancels in-flight playback immediately
- Backend pre-warming on startup to reduce first-turn latency

### Multiple Backends

| Stage | Backend | Notes |
|-------|---------|-------|
| ASR | Google Web Speech | Free, requires internet |
| ASR | OpenAI Whisper API | `whisper-1`, requires `OPENAI_API_KEY` |
| ASR | Local Whisper | HuggingFace `openai/whisper-tiny` (offline) |
| ASR | Mock | Instant, no dependencies |
| LLM | OpenAI | `gpt-4o-mini` default, requires `OPENAI_API_KEY` |
| LLM | Anthropic Claude | `claude-3-5-haiku-latest`, requires `ANTHROPIC_API_KEY` |
| LLM | Mock | Instant, no dependencies |
| TTS | Microsoft Edge TTS | Neural voices, `en-US-JennyNeural` default |
| TTS | pyttsx3 | Fully offline |
| TTS | Mock | Instant, no dependencies |

### Resilience
- Per-service circuit breakers with CLOSED → OPEN → HALF_OPEN state machine
- Configurable per-stage timeouts (ASR: 10s, LLM: 15s, TTS: 10s)
- Graceful degradation with automatic fallback chains (ASR → Google → Mock, TTS → pyttsx3 → Mock)
- Configurable degradation message for canned responses under total failure
- Client errors (auth, 404) excluded from circuit breaker failure counts

### Observability
- Nanosecond-precision latency tracking per stage: ASR, LLM TTFT, LLM total, TTS first-chunk, playback, E2E
- Rolling window statistics: mean, median, p50/p90/p95/p99, min, max, std dev
- SQLite persistence for cross-session metrics (`latency_metrics.db`)
- matplotlib chart generation (`latency_report.png`)
- Health check endpoint exposing circuit breaker states and degradation level

---

## Quickstart

### Prerequisites

- Python 3.11+
- PortAudio (for microphone capture)
  ```bash
  # Debian/Ubuntu
  sudo apt install portaudio19-dev

  # macOS
  brew install portaudio
  ```

### Install

```bash
git clone https://github.com/Ashok007-cmd/echoflow-voice-pipeline.git
cd echoflow-voice-pipeline

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

For local Whisper inference (downloads ~75 MB model on first run):

```bash
pip install -e ".[local]"
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Run the offline benchmark (no API keys or microphone required)

```bash
voice-assistant benchmark --turns 10 --mock
```

### Run with real backends

```bash
voice-assistant interactive \
  --asr-backend google \
  --llm-backend openai \
  --tts-backend edge_tts
```

---

## Configuration

All settings load from environment variables or `.env`. CLI flags override env vars for a single run.

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_BACKEND` | `google` | `google` · `openai` · `local_whisper` · `mock` |
| `LLM_BACKEND` | `openai` | `openai` · `anthropic` · `mock` |
| `TTS_BACKEND` | `edge_tts` | `edge_tts` · `pyttsx3` · `mock` |
| `OPENAI_API_KEY` | — | Required for `openai` ASR and LLM backends |
| `ANTHROPIC_API_KEY` | — | Required for `anthropic` LLM backend |
| `SYSTEM_PROMPT` | *(built-in)* | Override the assistant system prompt |

---

## CLI Reference

### `interactive` — live voice assistant loop

```
voice-assistant interactive [OPTIONS]

Options:
  --asr-backend TEXT    ASR backend: google, openai, local_whisper, mock
  --llm-backend TEXT    LLM backend: openai, anthropic, mock
  --tts-backend TEXT    TTS backend: edge_tts, pyttsx3, mock
  --mock                Force all backends to mock mode (offline)
  --timeout FLOAT       Per-stage timeout in seconds (overrides all three)
  --report              Write latency_report.png on exit
  -v, --verbose         Debug logging
```

Example output per turn:
```
User: what's the weather like today?
Assistant: I don't have access to real-time data, but I can help with...
Latency: ASR=312ms | LLM TTFT=187ms | TTS=94ms | E2E=593ms
```

### `benchmark` — synthetic latency measurement

```
voice-assistant benchmark [OPTIONS]

Options:
  -n, --turns INTEGER   Number of turns to run (default: 10)
  --mock                All-mock backends (offline, no API keys needed)
  --mock-asr            Mock ASR only
  --mock-llm            Mock LLM only
  --mock-tts            Mock TTS only
  --output PATH         Chart output path (default: latency_report.png)
  -v, --verbose         Debug logging
```

### `report` — visualise existing metrics

```
voice-assistant report [METRICS_FILE] [OPTIONS]

Options:
  --output PATH         Chart output path (default: latency_report.png)
```

---

## Docker

### Pull from GitHub Container Registry (recommended)

```bash
docker pull ghcr.io/ashok007-cmd/echoflow-voice-pipeline:latest

# Run the offline benchmark
docker run --rm ghcr.io/ashok007-cmd/echoflow-voice-pipeline:latest benchmark --turns 5 --mock

# Run with API keys
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  ghcr.io/ashok007-cmd/echoflow-voice-pipeline:latest \
  benchmark --turns 10
```

### Build locally

```bash
docker build -t echoflow .
docker run --rm echoflow benchmark --turns 5 --mock
```

The default `CMD` runs `benchmark --turns 5 --mock` — no environment setup required.

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full suite
.venv/bin/python -m pytest

# With coverage report
.venv/bin/python -m pytest --cov=src --cov-report=term-missing
```

All tests use mock backends — no API keys, microphone, or audio hardware required. Tests cover:
- All ASR backends (Google, OpenAI Whisper, Local Whisper) with patched network calls
- All LLM backends (OpenAI streaming, Anthropic streaming) with mock chunk iterators
- All TTS backends (Edge TTS, pyttsx3) including subprocess playback and temp-file cleanup
- ASR streaming with VAD, timeout, and exception paths
- TTS streaming with sentence-splitting
- Circuit breaker state machine (CLOSED → OPEN → HALF_OPEN → CLOSED)

---

## Project Structure

```
src/
├── main.py                  # CLI entry point (interactive / benchmark / report)
├── config.py                # Dataclass config with env-var loading and validation
├── pipeline/
│   ├── orchestrator.py      # Async pipeline coordinator (concurrent task queues)
│   ├── asr.py               # ASR engines + Dynamic VAD
│   ├── llm_client.py        # LLM streaming clients (OpenAI / Anthropic / Mock)
│   ├── tts.py               # TTS engines (Edge TTS / pyttsx3 / Mock)
│   └── models.py            # Shared dataclasses (AudioChunk, TurnMetrics, …)
├── monitoring/
│   ├── latency_tracker.py   # Nanosecond-precision stage timers + rolling stats
│   ├── visualizer.py        # matplotlib chart generation
│   └── database.py          # Async SQLite persistence for turn metrics
└── resilience/
    ├── circuit_breaker.py   # CLOSED / OPEN / HALF_OPEN state machine
    ├── timeout.py           # Per-stage timeout wrappers (asyncio.wait_for)
    └── degradation.py       # Degradation level tracking + alert callbacks
tests/
├── test_backends.py         # ASR / LLM / TTS backend unit tests
├── test_pipeline.py         # End-to-end pipeline and orchestrator tests
└── test_cli.py              # CLI command tests
```

---

## How the Pipeline Works

1. **Audio capture** — PyAudio streams 16 kHz PCM chunks from the microphone.
2. **VAD** — `DynamicVAD` classifies each chunk using RMS energy and Zero Crossing Rate. Once silence exceeds `silence_threshold` (0.5 s default) after speech, the utterance boundary is emitted.
3. **ASR** — The utterance is transcribed by the configured engine, wrapped with a circuit breaker and per-stage timeout.
4. **LLM streaming** — The transcript is sent to the LLM. Tokens stream in and are buffered into sentences using punctuation/word-count splitting.
5. **TTS** — Each sentence is synthesized as soon as it is ready. The TTS worker runs concurrently with the LLM worker via `asyncio.Queue`.
6. **Playback** — Audio chunks queue into a third worker that calls `engine.play()` as they arrive.
7. **Interrupt** — When new speech is detected, `orchestrator.interrupt()` cancels the active playback task and drains both queues before the next turn begins.

---

## License

[MIT](LICENSE)
