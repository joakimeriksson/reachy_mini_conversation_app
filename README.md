---
title: Reachy Local Assistant
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini — fully local (Ollama + Kokoro voice server)
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Local Assistant (local / on-prem)

A **fully local** conversational app for the Reachy Mini robot: speech, reasoning,
and vision run on **Ollama (Gemma)** and the voice on a **self-hosted voice
server** (Kokoro TTS + Whisper STT) — no cloud, no API keys. Originally derived
from [Pollen's conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app),
now an independent project built around a **local, on-prem stack**; Pollen's app
(which went the cloud-realtime route) is still watched for ideas worth porting.

> [!IMPORTANT]
> Two services must be running before the app can hold a conversation: **Ollama**
> and the **voice server** (`voice-server/`). Without the voice server Reachy
> hears and thinks but cannot speak. The app probes both at startup and reports
> what is missing — see [Bringing up the stack](#bringing-up-the-stack).

## Table of contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Bringing up the stack](#bringing-up-the-stack) — or the full [SETUP.md](SETUP.md) walkthrough
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [Deploying to a robot](#deploying-to-a-robot)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Advanced features](#advanced-features)
- [License](#license)

## Overview
- **Local pipeline:** mic → VAD → **Gemma** (audio straight into the chat model) → **Ollama chat** (+ tools / MCP) → **voice server TTS** → speaker, with the head wobbler reacting to the spoken audio.
- **Vision via the LLM:** the camera tool hands a frame to the multimodal model (Gemma); no separate vision model.
- **Long-term memory:** `remember` / `forget` tools persist facts that are injected into the prompt across sessions.
- **External tool servers:** built-in MCP client connects to remote MCP servers (token / API-key auth).
- **Self-hosted voice:** an **OpenAI-compatible `/v1/audio/speech`** server keeps the robot thin. The included `voice-server/` runs Kokoro (incl. a Swedish fine-tune, 10 voices) plus Whisper transcription.
- **Barge-in:** talk over Reachy to interrupt it, with WebRTC echo cancellation so it works on open speakers.
- **Layered motion:** dances, emotions, head-tracking and speech-reactive wobble.

Everything heavy (Gemma and the voice generator) can run on a separate **on-prem
server**, keeping the robot/client thin.

## Architecture

```
   mic ──▶ VAD ──▶ Ollama chat (Gemma: hears audio, +tools/MCP) ──▶ voice server ──▶ speaker
                                    │                                (/v1/audio/speech)
                                    └──▶ Whisper (/v1/audio/transcriptions)
                                         transcribes each turn while the model
                                         thinks: noise gate, language evidence,
                                         and text (not audio) in history
```

By default (`OLLAMA_DIRECT_AUDIO=1`) the user's audio goes **straight into the chat
model** — one call instead of a separate STT pass, roughly halving latency, and the
model hears tone. Set `OLLAMA_DIRECT_AUDIO=0` for a classic transcribe-then-chat
pipeline.

On-prem topology (see [DEPLOY.md](DEPLOY.md)):

```
   Reachy Mini (body) ──WiFi──▶ client app ──OLLAMA_URL──▶ Ollama (Gemma)
   mic/cam/speaker              VAD · turn loop ──TTS_URL/STT_URL──▶ voice server
```

## Installation

> [!IMPORTANT]
> Install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) first.
> You also need a reachable **[Ollama](https://ollama.com)** server with an
> audio+vision+tools model pulled (e.g. `ollama pull gemma4:latest`).

```bash
# macOS (Homebrew): the reachy-mini SDK needs these system libs
brew install pkg-config gobject-introspection cairo
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/opt/libffi/lib/pkgconfig"

uv venv --python python3.12 .venv && source .venv/bin/activate
uv sync
```

Optional extras:
```bash
uv sync --extra mediapipe_vision   # MediaPipe head-tracking
uv sync --extra localdev           # standalone "fake robot" runner (sounddevice + webcam)
uv sync --extra aec                # echo cancellation, so barge-in works on open speakers
uv sync --group dev                # dev tooling (pytest/ruff/mypy)
```

| Extra | Purpose |
|-------|---------|
| `mediapipe_vision` | CLIENT-side mediapipe head-tracker (`--head-tracker mediapipe`, dev only) — the robot tracks faces daemon-side without it |
| `localdev` | `scripts/local_chat.py` — talk via your computer's mic/speaker/webcam (no robot). Not shipped to the robot. |
| `aec` | WebRTC echo cancellation (livekit) — required for barge-in on open speakers. |
| `silero` | Neural VAD backend (`VAD_BACKEND=silero`); better at speech-vs-noise than the default. |

> The conversation **vision** is the multimodal LLM itself — there is no separate
> local vision model (no torch/transformers in the app environment).

## Bringing up the stack

**→ Full walkthrough with verification at each step: [SETUP.md](SETUP.md).**

Three processes. The voice server lives in **its own environment**
(`voice-server/`) because its Kokoro/torch stack conflicts with the app's pins.

```bash
# 1. Ollama — must listen on all interfaces if the robot/app is on another machine
OLLAMA_HOST=0.0.0.0 ollama serve
ollama pull gemma4:latest

# 2. Voice server (TTS + Whisper STT) — its own uv project
cd voice-server && uv sync
uv run python ../scripts/voice_server.py \
    --engine kokoro --voice af_heart --port 8880 --whisper base

# 3. The app
reachy-local-assistant
```

Or start all three with the helper, which waits for each to come up:

```bash
./scripts/start_stack.sh          # --check to just report status, --no-app for backends only
```

Check what is running at any time:

```bash
curl -s localhost:8880/health          # {"status":"ok","tts":true,"stt":true,...}
curl -s localhost:11434/api/tags       # Ollama's pulled models
```

The app probes both backends at startup and logs exactly what is unreachable; the
same status is shown on the settings page under **Ollama & voice server**, with a
**Check connection** button.

> **Voice names are engine-specific.** `--engine kokoro` uses `af_heart`-style
> names; the Swedish `--engine kokoro-svml` uses named packs like `Stina` and
> additionally needs the separate `swedish-kokoro` checkout. See
> [voice-server/README.md](voice-server/README.md).

## Configuration

Copy `.env.example` to `.env` (or `.env.robot.example` for a robot deployment) and edit.
No API key is required. Key variables:

| Variable | Description |
|----------|-------------|
| `OLLAMA_URL` | Ollama server (local or remote). Default `http://localhost:11434`. |
| `OLLAMA_MODEL` | Conversation model (audio + chat + vision + tools), e.g. `gemma4:latest`. |
| `OLLAMA_DIRECT_AUDIO` | `1` (default): feed audio straight to the chat model. `0`: separate STT pass. |
| `OLLAMA_STT_MODEL` | STT model when `OLLAMA_DIRECT_AUDIO=0` (defaults to `OLLAMA_MODEL`). |
| `OLLAMA_TEMPERATURE` / `OLLAMA_NUM_CTX` / `OLLAMA_KEEP_ALIVE` | Generation + model-load tuning. |
| `TTS_URL` | **Required.** Voice server `/v1/audio/speech`, e.g. `http://voicehost:8880/v1/audio/speech`. |
| `TTS_VOICE` / `TTS_MODEL` / `TTS_FORMAT` | Voice name — must match the engine (`af_heart` for `kokoro`, `Stina` for `kokoro-svml`). |
| `STT_URL` | Whisper `/v1/audio/transcriptions`. Defaults to the `TTS_URL` host; used for the text history and the noise gate. |
| `NOISE_GATE` | `1` (default): drop a turn when Whisper hears no words in it, so room noise never gets an answer. Needs `STT_URL`. |
| `VAD_AGGRESSIVENESS` / `VAD_SILENCE_MS` | Voice-activity detection (raise aggressiveness if it over-listens). |
| `VAD_BACKEND` / `VAD_THRESHOLD` | `webrtc` (default) or `silero` (neural, needs the `silero` extra). |
| `BARGE_IN` / `AEC` | Interrupt Reachy mid-reply; echo cancellation so that works on open speakers. |
| `MCP_SERVER_URLS` | External MCP tool servers (comma-separated; append `token=` / `api_key=`). |
| `REACHY_MINI_CUSTOM_PROFILE` | Personality profile (folder under `profiles/`). |

See `.env.example` for the fully annotated list. `TTS_URL`, `OLLAMA_URL` and the
voice can also be set from the app's settings page and are persisted there.

## Running the app

```bash
reachy-local-assistant
```
The app is **headless**: audio and camera flow through the robot's media pipeline
(`robot.media`). When launched by the Reachy Mini Apps runtime it also serves a
settings page — backend URLs and their health, personality studio, MCP servers,
and the live conversation transcript — at http://localhost:7860. That page is
this app's own server (declared via `custom_app_url`); the **Reachy Mini Control**
desktop app opens it, and you can also browse to it directly. It is unrelated to
the robot's old web dashboard, which was removed in SDK 1.9.0.

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {mediapipe}` | `None` | Head-tracking backend (requires the matching extra). |
| `--no-camera` | `False` | Run without the camera. |
| `--local-webcam` | `False` | Dev only: use the computer's webcam (OpenCV) when `robot.media` has no camera. |
| `--webcam-index` | `0` | OpenCV webcam device index. |
| `--robot-name` | `None` | Connect to a specific robot by name (must match the daemon's). |
| `--mcp-servers` | `None` | Override `MCP_SERVER_URLS` from the CLI. |
| `--debug` | `False` | Verbose logging. |

**Talk without a robot** (great for tuning prompts/voice/VAD):
```bash
python scripts/local_chat.py            # mic/speaker via sounddevice
python scripts/local_chat.py --vision   # + webcam; ask "what do you see?"
```

## Deploying to a robot

See **[DEPLOY.md](DEPLOY.md)** for the full guide. In short, for a Reachy Mini
Wireless: run the app on your computer (with Ollama local or remote), the robot
provides the body over WiFi. Use `.env.robot.example` as your config template.

## LLM tools exposed to the assistant

| Tool | Action |
|------|--------|
| `camera` | Capture a frame; the multimodal LLM (Gemma) sees it directly. |
| `remember` / `forget` | Persist / remove a long-term fact about the user. |
| `move_head` | Queue a head pose (left/right/up/down/front). |
| `head_tracking` | Toggle face tracking — daemon-side on the robot (no client mediapipe needed). |
| `dance` / `stop_dance` | Play / clear a dance from `reachy_mini_dances_library`. |
| `play_emotion` / `stop_emotion` | Play / clear a recorded emotion (open HF dataset). |
| `task_status` / `task_cancel` | Inspect / cancel a long-running background tool. |
| `go_to_sleep` | Sleep pose + stop the app ("Reachy, go to sleep"). Also fires after `INACTIVITY_TIMEOUT_MIN` of silence. |

External tools from configured `MCP_SERVER_URLS` are exposed automatically.

## Advanced features

Built-in motion content is published as open Hugging Face datasets:
- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary><b>Custom profiles</b></summary>

Set `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `profiles/<name>/`. If unset, the
`default` profile is used. Each profile includes `instructions.txt` (prompt) and
recommended `tools.txt` (allowed tools); an optional `voice.txt` overrides
`TTS_VOICE` for that personality (e.g. a different Kokoro voice per character).
Profiles may include custom tool implementations (Python files subclassing
`reachy_local_assistant.tools.core_tools.Tool`; see `profiles/example/`).

Reuse shared prompt fragments via `[name]` placeholders, which pull matching files
under `src/reachy_local_assistant/prompts/` (nested paths allowed).

</details>

<details>
<summary><b>Locked profile mode</b></summary>

Set `LOCKED_PROFILE` in `src/reachy_local_assistant/config.py` to pin the app
to one profile and disable switching — useful for fixed-personality clones.

</details>

<details>
<summary><b>External profiles and tools</b></summary>

Store profiles/tools outside the repo via
`REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` and `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`
(see `.env.example`). With `AUTOLOAD_EXTERNAL_TOOLS=1`, all `*.py` tools in the
external tools directory are auto-registered. An external profile with no
`tools.txt` falls back to `profiles/default/tools.txt`.

</details>

<details>
<summary><b>Multiple robots on the same subnet</b></summary>

```bash
reachy-local-assistant --robot-name <name>
```
`<name>` must match the daemon's `--robot-name`.

</details>

## License

Apache 2.0
