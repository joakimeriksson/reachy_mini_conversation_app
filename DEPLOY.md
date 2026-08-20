# Deploying to a real Reachy Mini Wireless

This is the **local-first** deployment: the **app runs on your Mac/workstation**,
the **robot (onboard Pi) provides the body** (mic, speaker, camera, motors via its
own daemon), and the **heavy models run on a remote Ollama** server.

```
   ┌──────────────┐   WiFi / mDNS    ┌──────────────────────┐
   │ Reachy Mini  │◀───robot.media──▶│  Your Mac            │
   │ Wireless     │  (audio/camera,  │  reachy-local-       │
   │ (onboard Pi  │   motor commands)│  assistant           │
   │  + daemon)   │   :8000          │  (VAD · turn loop)   │
   └──────────────┘                  └────┬────────────┬────┘
                                          │ OLLAMA_URL │ TTS_URL / STT_URL
                                          ▼            ▼
                            ┌─────────────────┐  ┌──────────────────────┐
                            │ Ollama server   │  │ Voice server         │
                            │ gemma4          │  │ Kokoro TTS + Whisper │
                            │  :11434         │  │  :8880               │
                            └─────────────────┘  └──────────────────────┘
```

**Both** backend services must be running. The app probes them at startup and
logs precisely what is unreachable; the settings page shows the same status with
a **Check connection** button.

Nothing about the conversation pipeline changes — only *where* audio/camera come
from (the real robot instead of sim/sounddevice). The Mac-sim headaches
(GStreamer daemon crash, browser WebRTC, cv2/`av`) **do not apply here**: the
daemon runs on the robot's Linux, and the camera comes from `robot.media`.

---

## 1. Network

- Mac and robot on the **same WiFi**.
- Confirm the robot's daemon is reachable (it runs on the Pi, port 8000). Probe
  the **API**, not `/` — since SDK 1.9.0 the web dashboard is gone and `/` only
  serves a "download the desktop app" page:
  ```bash
  curl -s http://reachy-mini.local:8000/api/daemon/status | grep -o '"state":"[a-z]*"'
  ```
  `"state":"running"` is what matters.

  > **Do not gate on `backend_status.ready`.** It stays `false` (with
  > `last_alive: null`) on a perfectly healthy robot: those fields are only
  > written once the daemon's WebSocket publishers are wired, and a client that
  > connects in **network / WebRTC** mode never triggers that. Verified on a
  > wireless unit at SDK 1.9.0 — `ready:false`, `nb_error:0`, control loop at
  > ~50 Hz, and the app held a full conversation (mic, LLM, speech, tool calls)
  > the whole time. Judge readiness by the app's own startup log instead:
  > `Backend OK — …` lines followed by `Ollama conversation handler ready`.

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
TTS_URL="http://<VOICE_HOST>:8880/v1/audio/speech"    # required — see step 4
TTS_VOICE="Stina"
REACHY_MINI_CUSTOM_PROFILE="default"        # your personality
# MCP_SERVER_URLS="https://your-host/mcp api_key=..."   # optional external tools
```

## 4. Voice server

Required — without it Reachy hears and thinks but never speaks. It runs in its
own environment (see [voice-server/README.md](voice-server/README.md)):

```bash
cd voice-server && uv sync
PYTORCH_ENABLE_MPS_FALLBACK=1 SWEDISH_KOKORO_PATH=/abs/path/to/swedish-kokoro \
  uv run python voice_server.py \
      --engine kokoro-svml --voice Stina --port 8880 --whisper base
```

The Swedish engine additionally needs the separate `swedish-kokoro` checkout for
its g2p module; for a stack that needs nothing outside this repo use
`--engine kokoro --voice af_heart`.

Verify, and note `"stt":true` — without `--whisper` the conversation still works
but history keeps raw audio instead of transcripts:
```bash
curl -s http://<VOICE_HOST>:8880/health
```

## 5. Run the app (on the Mac)

```bash
uv run reachy-local-assistant
#   add --robot-name <name> if the daemon was started with one
```

**Do NOT pass** `--mockup-sim`, `--no-media`, or `--local-webcam` — those are
sim/dev-only. On the real robot the app uses the headless console path: audio is
the **robot's mic/speaker**, the camera is the **robot's camera** (so "what do you
see?" uses the real camera via `robot.media`).

You should see:
```
Connection mode selected: ...                        # connected to the robot daemon
Backend OK — Ollama: Up, N model(s), gemma4:latest available.
Backend OK — Voice server (TTS): Up at http://...:8880.
Ollama conversation handler ready (model=gemma4:latest)
```
Any `Backend DOWN` line names the service and what to do about it. Then just talk
to the robot.

## 6. Tuning

- **VAD** (if it captures too long / never stops): set `VAD_AGGRESSIVENESS=3` and/or
  lower `VAD_SILENCE_MS` in `.env`. The handler re-reads these live.
- **Echo / self-listening**: the half-duplex gate holds the mic for the full reply
  + a 0.6 s tail. For talking *over* Reachy, set `BARGE_IN=1` and `AEC=1` (needs
  the `aec` extra); the echo delay auto-calibrates with a chirp at startup.
- **Latency**: three hops (robot↔Mac, Mac↔Ollama, Mac↔voice server). Keep them on
  one LAN. The Ollama call dominates; a faster Ollama host helps most. Measure with
  `python scripts/latency_bench.py`.

---

# Self-hosted voice server

Engines, voices, Whisper, and running it as a service are documented in
**[voice-server/README.md](voice-server/README.md)**. In short:

```bash
cd voice-server && uv sync
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python voice_server.py \
    --engine kokoro-svml --voice Stina --port 8880 --whisper base
```

```dotenv
TTS_URL=http://<voice-host>:8880/v1/audio/speech
TTS_VOICE=Stina
```

All engines return WAV that `RemoteTTS` decodes via `soundfile` (Kokoro @ 24 kHz).
Bind to `0.0.0.0` and open the port so the robot/client can reach it.
`voice-server/voice_server.py` lives outside `src/`, so it never ships in the robot wheel.

---

# Deploy as a standalone on-robot app

Reachy Mini's "app store" is **HuggingFace Spaces**. You publish the app to a
Space under your account, then install it from the **Reachy Mini Control** app,
which pip-installs it onto the Pi and runs it via the `reachy_mini_apps` entry
point.

> [!IMPORTANT]
> **The robot's web dashboard is gone as of SDK 1.9.0.** `http://reachy-mini.local:8000/`
> (and `/settings`, `/logs`) now serve only a "Web Dashboard Deprecated" page
> pointing at the **Reachy Mini Control** desktop app —
> <https://pollen-robotics-reachy-mini.hf.space/download>. Install/start/config
> all happen there now.
>
> The daemon's **REST API is unaffected** (`/api/apps/*`, `/api/daemon/*`,
> `/api/media/*`, …), so the scripted paths below still work, and this app's own
> settings page is a *separate* server that is **not** deprecated (see step 5).

> **Hard constraint:** neither gemma4 nor the Kokoro voice **can run on the Pi**.
> The Pi runs the app + VAD + turn loop only; `OLLAMA_URL` and `TTS_URL` must both
> point at machines reachable from the robot over WiFi (your Mac or a server).
> Keep them always-on and bound to `0.0.0.0`. (OpenCV is not needed on the Pi —
> the camera tool uses Pillow and the robot's own camera.)

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
uv run reachy-mini-app-assistant publish . "Fully local Ollama + Kokoro conversation app" --public
```
- Omit `--official` — that requests inclusion in Pollen's official store; this is
  your fork.
- Creates `https://huggingface.co/spaces/<you>/<app>` with the code.
- Re-run `publish` to push updates; Reachy Mini Control can then "Update" the app.

## 4. Install on the robot
In **Reachy Mini Control** (the desktop app — the web dashboard no longer exists),
find your app by its Space and **Install**. It pip-installs the package + deps
into the robot's apps venv; `webrtcvad-wheels`, `numpy` and `scipy` all ship
arm64 wheels, so the Pi install works without compiling.

Or drive the daemon's REST API directly, which is handy when the app vanishes
after a failed update (the updater uninstalls before it installs):
```bash
curl -X POST http://reachy-mini.local:8000/api/apps/install-private-space \
     -H 'Content-Type: application/json' \
     -d '{"space_id":"<you>/reachy_mini_conversation_app"}'   # uses the stored HF token
curl -X POST http://reachy-mini.local:8000/api/apps/start-app/reachy_local_assistant
```

> Bump `version` in `pyproject.toml` on **every dependency change**: the robot's
> uv caches built metadata by `(name, version)`, so a re-published Space with the
> same version can be installed with stale, previously-failing dependencies.

## 5. Configure on the robot
Easiest is the app's own settings page ("Ollama & voice server" panel), which
persists to the instance `.env` and applies live — then hit **Check connection**.
Remember that on the robot `localhost` is the Pi, so use your Mac's LAN IP:
```dotenv
OLLAMA_URL="http://<your-ollama-host>:11434"   # MUST be reachable from the Pi
OLLAMA_MODEL="gemma4:latest"
TTS_URL="http://<your-voice-host>:8880/v1/audio/speech"
TTS_VOICE="Stina"
REACHY_MINI_CUSTOM_PROFILE="default"
```
Personality, MCP servers and the live conversation transcript are on that same
page. It is served by **this app**, not the daemon (`custom_app_url` in
`main.py`, port 7860), so the dashboard deprecation does not affect it — Reachy
Mini Control opens it for you, and it also works by browsing straight to
`http://reachy-mini.local:7860/` while the app is running.

## 6. Run
Start the app from Reachy Mini Control. Audio/camera come from the robot's own
hardware; speech + chat go to the remote Ollama; the voice comes back from the
remote voice server. The app's log (and the settings page) reports whether both
are reachable — check there first if Reachy listens but never answers.

## Alternative: local install (no HuggingFace)
You can also copy the repo onto the Pi and `pip install -e .` into the apps venv
(a LOCAL app source is still supported). Handy for private iteration without
publishing a Space.
