# Setup — from clone to talking robot

A first-run guide for the fully local stack. Every step ends with a command that
tells you whether it worked, so a failure is caught where it happens rather than
showing up later as a robot that listens and never answers.

**Three processes** must be running:

| # | Process | Port | Provides |
|---|---------|------|----------|
| 1 | Ollama | 11434 | Hearing + reasoning + vision (Gemma) |
| 2 | Voice server | 8880 | Speech (Kokoro TTS) + transcription (Whisper) |
| 3 | The app | 7860 | VAD, turn loop, robot motion, settings page |

They can live on different machines. Nothing leaves your network.

---

## 0. Prerequisites

- **Python 3.11+** and [uv](https://docs.astral.sh/uv/).
- A machine that can run Gemma. A Raspberry Pi **cannot** — neither can it run
  Kokoro. On a Reachy Mini Wireless, steps 1 and 2 run on your computer and the
  robot is just the body.
- macOS only: the Reachy SDK needs system libs.
  ```bash
  brew install pkg-config gobject-introspection cairo
  export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/opt/libffi/lib/pkgconfig"
  ```

```bash
git clone <your fork> reachy_mini_conversation_app
cd reachy_mini_conversation_app
uv venv --python python3.12 .venv && source .venv/bin/activate
uv sync
```

---

## 1. Ollama

```bash
# OLLAMA_HOST=0.0.0.0 is required if the app or robot is on another machine
OLLAMA_HOST=0.0.0.0 ollama serve

# In another shell — must support audio + vision + tools
ollama pull gemma4:latest
```

**Verify:**
```bash
curl -s localhost:11434/api/tags | grep gemma4
```
Nothing printed → the model is not pulled, and the app will refuse to answer.

---

## 2. Voice server

This is the [kokoro-voice-server](https://github.com/joakimeriksson/kokoro-voice-server)
repo, included as the `voice-server/` **submodule** — a separate uv project,
since its Kokoro/torch stack conflicts with the app's pinned dependencies.

```bash
git submodule update --init   # first time only
cd voice-server
uv sync            # downloads torch + kokoro + faster-whisper; takes a few minutes
```

### Which engine?

| Engine | Languages | Extra requirement |
|--------|-----------|-------------------|
| `kokoro` **(default, start here)** | en, es, fr, it, pt, hi, zh, ja | none |
| `kokoro-svml` | Swedish (10 named voices) + all of the above | the separate [`swedish-kokoro`](https://github.com/joakimeriksson/swedish-kokoro) checkout, for its g2p module |
| `kokoro-sv` | Swedish only, one voice, espeak g2p | same checkout |

> **Voice names must match the engine.** Base Kokoro voices look like `af_heart`,
> `bm_george`. The Swedish packs are names like `Stina`, `Björn`. Asking the base
> engine for `Stina` logs a warning and falls back to its default voice — it will
> speak, just not in the voice you asked for.

```bash
# Default engine — nothing outside this repo needed
uv run python voice_server.py \
    --engine kokoro --voice af_heart --host 0.0.0.0 --port 8880 --whisper base

# Swedish (needs the swedish-kokoro checkout)
PYTORCH_ENABLE_MPS_FALLBACK=1 SWEDISH_KOKORO_PATH=/abs/path/to/swedish-kokoro \
  uv run python voice_server.py \
      --engine kokoro-svml --voice Stina --host 0.0.0.0 --port 8880 --whisper base
```

First start downloads models and takes 30-60s.

**Verify:**
```bash
curl -s localhost:8880/health
# {"status":"ok","engine":"KokoroEngine","default_voice":"af_heart","tts":true,"stt":true}
```

`"stt":false` means you omitted `--whisper`. Conversations still work, but the
transcript panel shows only a mic glyph and history keeps raw audio instead of
text. Prefer `--whisper base`.

**Hear it for yourself:**
```bash
curl -s -X POST localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, I am Reachy Mini.","voice":"af_heart"}' -o /tmp/t.wav
afplay /tmp/t.wav          # macOS   (Linux: aplay /tmp/t.wav)
```

---

## 3. Configure the app

```bash
cp .env.example .env
```

The three settings that matter — everything else has a working default:

```dotenv
OLLAMA_URL="http://localhost:11434"
TTS_URL="http://localhost:8880/v1/audio/speech"
TTS_VOICE="af_heart"                  # or Stina, with --engine kokoro-svml
```

Use **LAN IPs, not `localhost`**, for any service on another machine — and note
that on the robot `localhost` means the robot's own Pi.

> `.env` overrides real environment variables, not the other way round. If a
> setting seems to be ignored, check `.env` first.

---

## 4. Run it

```bash
reachy-local-assistant                      # add --robot-name <name> for a specific robot
```

Or start all three together — it waits for each service to answer before starting
the next, and auto-selects the Swedish engine only if `SWEDISH_KOKORO_PATH` is set:

```bash
./scripts/start_stack.sh                    # everything
./scripts/start_stack.sh --check            # just report what is up
./scripts/start_stack.sh --no-app           # backends only
```

**A healthy start logs:**
```
Backend OK — Ollama: Up, 4 model(s), gemma4:latest available.
Backend OK — Voice server (TTS): Up at http://localhost:8880. Engine: KokoroEngine.
Backend OK — Voice server (STT): Up at http://localhost:8880. Engine: KokoroEngine.
Ollama conversation handler ready (model=gemma4:latest)
```

Any `Backend DOWN` line names the service and what to do. The same status, with a
**Check connection** button and a live transcript of the conversation, is on the
settings page at <http://localhost:7860>.

### No robot?

```bash
uv sync --extra localdev
python scripts/local_chat.py            # your computer's mic and speaker
python scripts/local_chat.py --vision   # + webcam; ask "what do you see?"
```

Measure the pipeline without talking at all:
```bash
python scripts/latency_bench.py         # per-stage latency, needs both backends up
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Reachy hears you, then silence | Voice server down or `TTS_URL` unset | `curl localhost:8880/health`; check the app's startup log |
| `Backend DOWN — Ollama` | Not running, or bound to loopback on another host | Start with `OLLAMA_HOST=0.0.0.0` |
| `Model 'gemma4:latest' is not pulled` | — | `ollama pull gemma4:latest` |
| Wrong voice, warning in the voice-server log | Voice name doesn't match the engine | Use `af_heart`-style for `kokoro`, `Stina`-style for `kokoro-svml` |
| Transcript shows only 🎤 | Server started without `--whisper` | Restart it with `--whisper base` |
| Reachy interrupts itself | Its own voice re-entering the mic | Use headphones, or set `AEC=1` (needs `uv sync --extra aec`) |
| Answers when nobody spoke | Whisper off, so the noise gate can't run | Start the voice server with `--whisper base` (the gate drops turns Whisper hears no words in) |
| Never stops listening | VAD too lenient | `VAD_AGGRESSIVENESS=3`, lower `VAD_SILENCE_MS` |
| Settings changes lost on restart | — | Save from the settings page; it persists to the instance `.env` |

Deploying onto a real Reachy Mini (including publishing to a Hugging Face Space
and installing it from the **Reachy Mini Control** desktop app — the robot's web
dashboard was removed in SDK 1.9.0) is covered in **[DEPLOY.md](DEPLOY.md)**.
