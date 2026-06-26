# Deploying to a real Reachy Mini Wireless

This is the **local-first** deployment: the **app runs on your Mac/workstation**,
the **robot (onboard Pi) provides the body** (mic, speaker, camera, motors via its
own daemon), and the **heavy models run on a remote Ollama** server.

```
   ┌──────────────┐   WiFi / mDNS    ┌─────────────────────┐
   │ Reachy Mini  │◀───robot.media──▶│  Your Mac           │
   │ Wireless     │  (audio/camera,  │  reachy-mini-        │
   │ (onboard Pi  │   motor commands)│  conversation-app    │
   │  + daemon)   │   :8000          │  (VAD·Piper·turn loop)│
   └──────────────┘                  └──────────┬──────────┘
                                                 │ OLLAMA_URL
                                                 ▼
                                       ┌─────────────────────┐
                                       │ Ollama server       │
                                       │ gemma4 (STT + chat) │
                                       │  :11434             │
                                       └─────────────────────┘
```

Nothing about the conversation pipeline changes — only *where* audio/camera come
from (the real robot instead of sim/sounddevice). The Mac-sim headaches
(GStreamer daemon crash, browser WebRTC, cv2/`av`) **do not apply here**: the
daemon runs on the robot's Linux, and the camera comes from `robot.media`.

---

## 1. Network

- Mac and robot on the **same WiFi**.
- Confirm the robot's daemon is reachable (it runs on the Pi, port 8000):
  ```bash
  curl -s -o /dev/null -w "%{http_code}\n" http://reachy-mini.local:8000/    # expect a code, not "could not resolve"
  ```
  If `reachy-mini.local` doesn't resolve, use the robot's IP, or pass
  `--robot-name <name>` to the app if the robot has a custom name.

## 2. Ollama (remote)

On the **Ollama host** (server or another machine):
```bash
# Make Ollama listen on all interfaces so the Mac can reach it
OLLAMA_HOST=0.0.0.0 ollama serve          # or set OLLAMA_HOST in its service config
ollama pull gemma4:latest                  # must expose caps: audio + vision + tools
```
Open port **11434** in the host firewall. Verify from the Mac:
```bash
curl -s http://<OLLAMA_HOST>:11434/api/tags | head -c 200
```

## 3. App config (`.env` on the Mac)

Copy `.env.robot.example` to `.env` and set:
```dotenv
OLLAMA_URL="http://<OLLAMA_HOST>:11434"     # the remote Ollama
OLLAMA_MODEL="gemma4:latest"
PIPER_VOICE="en_US-lessac-medium"
PIPER_DATA_DIR="piper_voices"               # auto-downloads on first run if missing
REACHY_MINI_CUSTOM_PROFILE="default"        # your personality
# MCP_SERVER_URLS="https://your-host/mcp api_key=..."   # optional external tools
```

## 4. Run the app (on the Mac)

```bash
uv run reachy-local-assistant
#   add --robot-name <name> if the daemon was started with one
```

**Do NOT pass** `--mockup-sim`, `--no-media`, `--local-webcam`, or `--gradio` —
those are sim/dev-only. On the real robot the app auto-selects the headless
console path: audio is the **robot's mic/speaker**, the camera is the **robot's
camera** (so "what do you see?" uses the real camera via `robot.media`).

You should see:
```
Connection mode selected: ...           # connected to the robot daemon
Ollama conversation handler ready (model=gemma4:latest)
```
Then just talk to the robot.

## 5. Tuning

- **VAD** (if it captures too long / never stops): set `VAD_AGGRESSIVENESS=3` and/or
  lower `VAD_SILENCE_MS` in `.env`. The handler re-reads these live.
- **Echo / self-listening**: the half-duplex gate holds the mic for the full reply
  + a 0.6 s tail. If a tail of speech still leaks in, we can raise that.
- **Latency**: two network hops (robot↔Mac, Mac↔Ollama). Keep Ollama close
  (same LAN). gemma4 STT+chat dominates; a faster Ollama host helps most.

---

# Self-hosted voice server (Piper or Kokoro)

By default the app speaks with **local Piper** (`TTS_BACKEND=piper`). To offload
TTS to a server (thin client), run the included OpenAI-compatible voice server
(`scripts/voice_server.py`, exposes `/v1/audio/speech`) and point the app at it:

```dotenv
TTS_BACKEND=remote
TTS_URL=http://<voice-host>:8880/v1/audio/speech
TTS_VOICE=en_US-lessac-medium     # a voice for the chosen engine
```

Two interchangeable engines — pick at launch:

```bash
# Piper engine (light: reuses this repo's Piper voices)
uv sync --extra voiceserver
uv run python scripts/voice_server.py --engine piper --voice en_US-lessac-medium

# Kokoro engine (hexgrad/Kokoro-82M; needs torch + the model download).
# The voice server is a SEPARATE service, so install kokoro into ITS environment
# only — it is deliberately NOT a project dependency (its huggingface-hub needs
# would conflict with the app's pin). Run it with the venv python directly so
# `uv run` doesn't re-sync it away:
uv pip install kokoro
.venv/bin/python scripts/voice_server.py --engine kokoro --voice af_heart --port 8881
```

**Swedish** (`kokoro-sv` engine): Kokoro ships no Swedish, so this uses a
fine-tuned voice from the sibling **swedish-kokoro** project — a torch-free ONNX
model + espeak `sv` g2p (the model/voicepack auto-download from its HF repo). Point
at the project with `--svml-path` or `$SWEDISH_KOKORO_PATH`:
```bash
uv pip install onnxruntime misaki phonemizer-fork espeakng_loader torch
SWEDISH_KOKORO_PATH=../ai-smarthome/swedish-kokoro \
  .venv/bin/python scripts/voice_server.py --engine kokoro-sv --port 8882
# app: TTS_BACKEND=remote  TTS_URL=http://<host>:8882/v1/audio/speech
```
**`kokoro-svml` engine (Torch, neural g2p)** — the higher-quality path: the same
fine-tuned voice but driven by `KokoroSVML`, whose `SwedishG2P` uses the **neural
NST g2p** (correct loanword/name pronunciation) instead of espeak, and which also
speaks the other Kokoro languages. It needs `kokoro` + `torch` **and the neural
g2p model present** (the `swedish-tts/g2p` project, via `SV_NEURAL_G2P`/`SV_G2P_DIR`).
Run it where that model lives — typically the GPU box:
```bash
SV_NEURAL_G2P=nst_g2p SV_G2P_DIR=/abs/path/to/swedish-tts/g2p \
SWEDISH_KOKORO_PATH=/abs/path/to/swedish-kokoro \
  python scripts/voice_server.py --engine kokoro-svml --host 0.0.0.0 --port 8880
# startup log must say `swedish g2p backend=neural` (not espeak)
```
> Note: `KokoroSVML` defaults to the neural g2p and only checks that the *adapter*
> imports, not the model — so on a machine **without** the g2p model it reports
> `neural` then errors at synth. On espeak-only machines (Mac/robot) use `kokoro-sv`.

(For plain Swedish without the neural voice, use Piper `sv_SE-nst-medium` instead.)

All engines return WAV that `RemoteTTS` decodes via `soundfile`. Verified
end-to-end: Piper @ 22.05 kHz, Kokoro @ 24 kHz, Swedish Kokoro @ 24 kHz. Bind the
host to `0.0.0.0` and open the port so the robot/client can reach it.
`scripts/voice_server.py` lives outside `src/`, so it never ships in the robot wheel.

---

# Deploy as a standalone on-robot app

Reachy Mini's "app store" is **HuggingFace Spaces**. You publish the app to a
Space under your account, then install it from the robot's dashboard, which
pip-installs it onto the Pi and runs it via the `reachy_mini_apps` entry point.

> **Hard constraint:** gemma4 (and ideally the STT model) **cannot run on the
> Pi**. The Pi runs the app + Piper + VAD only, and `OLLAMA_URL` must point at a
> remote Ollama (your Mac or a server) reachable from the robot over WiFi. Keep
> it always-on, bound to `0.0.0.0`, with `gemma4:latest` pulled. (OpenCV is not
> needed on the Pi — the camera tool uses Pillow and the robot's own camera.)

## 1. Validate the app package
```bash
uv run reachy-mini-app-assistant check .
```
This checks the `reachy_mini_apps` entry point and the README metadata
(title + `reachy_mini` / `reachy_mini_python_app` tags) — already present here.

## 2. Log in to HuggingFace (one-time)
```bash
uv run huggingface-cli login        # or export HF_TOKEN=hf_...
```

## 3. Publish to a Space (your account, your fork)
```bash
uv run reachy-mini-app-assistant publish . "Local Ollama+Piper conversation app" --public
```
- Omit `--official` — that requests inclusion in Pollen's official store; this is
  your fork.
- Creates `https://huggingface.co/spaces/<you>/<app>` with the code.
- Re-run `publish` to push updates; the dashboard can then "Update" the app.

## 4. Install on the robot
Open the robot dashboard (the Reachy Mini web UI), find your app (by its Space),
and **Install** — it pip-installs the package + deps into the robot's apps venv.
`onnxruntime`, `piper-tts`, and `webrtcvad-wheels` all ship arm64 wheels, so the
Pi install works without compiling.

## 5. Configure on the robot
The app reads its instance `.env`. Set at minimum:
```dotenv
OLLAMA_URL="http://<your-ollama-host>:11434"   # MUST be remote/reachable from the Pi
OLLAMA_MODEL="gemma4:latest"
PIPER_VOICE="en_US-lessac-medium"
REACHY_MINI_CUSTOM_PROFILE="default"
```
Personality and MCP servers are also editable from the app's settings page in
the dashboard. The Piper voice auto-downloads on first run.

## 6. Run
Start the app from the dashboard. Audio/camera come from the robot's own
hardware; STT + chat go to the remote Ollama; Piper speaks locally on the Pi.

## Alternative: local install (no HuggingFace)
You can also copy the repo onto the Pi and `pip install -e .` into the apps venv
(the dashboard also supports a LOCAL app source). Handy for private iteration
without publishing a Space.
