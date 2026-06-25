---
title: Reachy Local Assistant
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini — fully local (Ollama + Piper)
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Local Assistant (local / on-prem)

A **fully local** conversational app for the Reachy Mini robot: speech, reasoning,
and vision run on **Ollama (Gemma)** and the voice on **Piper** — no cloud, no API
keys. A fork of Pollen's conversation app with the OpenAI Realtime backend
**replaced by a local, on-prem stack**.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Table of contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [Deploying to a robot](#deploying-to-a-robot)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Advanced features](#advanced-features)
- [License](#license)

## Overview
- **Local pipeline:** mic → VAD → **Gemma STT** → **Ollama chat** (+ tools / MCP) → **Piper TTS** → speaker, with the head wobbler reacting to the spoken audio.
- **Vision via the LLM:** the camera tool hands a frame to the multimodal model (Gemma); no separate vision model.
- **Long-term memory:** `remember` / `forget` tools persist facts that are injected into the prompt across sessions.
- **External tool servers:** built-in MCP client connects to remote MCP servers (token / API-key auth).
- **Pluggable voice:** Piper locally, or an external **OpenAI-compatible `/v1/audio/speech`** voice server for a thin client.
- **Layered motion:** dances, emotions, head-tracking and speech-reactive wobble.

Everything heavy (Gemma, and optionally the voice generator) can run on a separate
**on-prem server**, keeping the robot/client thin.

## Architecture

```
   mic ──▶ VAD ──▶ Gemma STT ──▶ Ollama chat (+tools/MCP) ──▶ Piper TTS ──▶ speaker
                     (Ollama)         (Ollama)                   (local or
                                                                  remote /v1/audio/speech)
```

On-prem topology (see [DEPLOY.md](DEPLOY.md)):

```
   Reachy Mini (body) ──WiFi──▶ client app ──OLLAMA_URL──▶ Ollama (Gemma)
   mic/cam/speaker              VAD · turn loop ──TTS_URL──▶ voice generator (optional)
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
uv sync --extra yolo_vision        # YOLO head-tracking backend
uv sync --extra mediapipe_vision   # MediaPipe head-tracking
uv sync --extra localdev           # standalone "fake robot" runner (sounddevice + webcam)
uv sync --group dev                # dev tooling (pytest/ruff/mypy)
```

| Extra | Purpose |
|-------|---------|
| `yolo_vision` | YOLO face detection for the `yolo` head-tracker (aiming the head) |
| `mediapipe_vision` | MediaPipe landmarks for the `mediapipe` head-tracker |
| `all_vision` | Both head-tracking backends |
| `localdev` | `scripts/local_chat.py` — talk via your computer's mic/speaker/webcam (no robot). Not shipped to the robot. |

> The conversation **vision** is the multimodal LLM itself — there is no separate
> local vision model (no torch/transformers).

## Configuration

Copy `.env.example` to `.env` (or `.env.robot.example` for a robot deployment) and edit.
No API key is required. Key variables:

| Variable | Description |
|----------|-------------|
| `OLLAMA_URL` | Ollama server (local or remote). Default `http://localhost:11434`. |
| `OLLAMA_MODEL` | Conversation model (audio STT + chat + vision + tools), e.g. `gemma4:latest`. |
| `OLLAMA_STT_MODEL` | STT model (defaults to `OLLAMA_MODEL`). |
| `OLLAMA_TEMPERATURE` / `OLLAMA_NUM_CTX` / `OLLAMA_KEEP_ALIVE` | Generation + model-load tuning. |
| `TTS_BACKEND` | `piper` (local) or `remote` (external `/v1/audio/speech`). |
| `PIPER_VOICE` / `PIPER_DATA_DIR` | Piper voice (auto-downloads). `PIPER_LENGTH_SCALE` = speed. |
| `TTS_URL` / `TTS_MODEL` / `TTS_VOICE` | Remote voice generator (when `TTS_BACKEND=remote`). |
| `VAD_AGGRESSIVENESS` / `VAD_SILENCE_MS` | Voice-activity detection (raise aggressiveness if it over-listens). |
| `MCP_SERVER_URLS` | External MCP tool servers (comma-separated; append `token=` / `api_key=`). |
| `REACHY_MINI_CUSTOM_PROFILE` | Personality profile (folder under `profiles/`). |

See `.env.example` for the fully annotated list.

## Running the app

```bash
reachy-local-assistant
```
On a real robot the app auto-selects **console mode** (audio/camera through the
robot). In **simulation** it auto-enables a Gradio web UI at http://localhost:7860.

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {yolo,mediapipe}` | `None` | Head-tracking backend (requires the matching extra). |
| `--no-camera` | `False` | Run without the camera. |
| `--local-webcam` | `False` | Dev only: use the computer's webcam (OpenCV) when `robot.media` has no camera. |
| `--webcam-index` | `0` | OpenCV webcam device index. |
| `--gradio` | `False` | Launch the Gradio web UI (auto-on in simulation). |
| `--robot-name` | `None` | Connect to a specific robot by name. |
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
| `head_tracking` | Toggle head-tracking offsets (position only, no recognition). |
| `dance` / `stop_dance` | Play / clear a dance from `reachy_mini_dances_library`. |
| `play_emotion` / `stop_emotion` | Play / clear a recorded emotion (open HF dataset). |
| `do_nothing` | Explicitly remain idle. |

External tools from configured `MCP_SERVER_URLS` are exposed automatically.

## Advanced features

Built-in motion content is published as open Hugging Face datasets:
- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary><b>Custom profiles</b></summary>

Set `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `profiles/<name>/`. If unset, the
`default` profile is used. Each profile includes `instructions.txt` (prompt) and
recommended `tools.txt` (allowed tools); an optional `voice.txt` selects the Piper
voice. Profiles may include custom tool implementations (Python files subclassing
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
