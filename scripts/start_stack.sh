#!/usr/bin/env bash
# Bring up the whole local stack: Ollama, the voice server, and the app.
#
# The three pieces live in different environments and start at very different
# speeds (Kokoro + Whisper take tens of seconds to load), so starting them by
# hand usually means launching the app against a backend that is not listening
# yet. This waits for each one before moving on.
#
#   ./scripts/start_stack.sh                 # start everything, run the app in the foreground
#   ./scripts/start_stack.sh --no-app        # just the backends (then run the app yourself)
#   ./scripts/start_stack.sh --check         # report what is already up, start nothing
#
# Ctrl-C stops whatever this script started; anything that was already running
# is left alone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Read config from .env when present, so the script targets the same hosts the
# app will. Only the few keys we need, and without executing the file.
env_get() {
    local key="$1" default="${2:-}"
    local line
    if [[ -f .env ]]; then
        line="$(grep -E "^[[:space:]]*${key}=" .env | tail -1 || true)"
        if [[ -n "$line" ]]; then
            line="${line#*=}"
            line="${line#"${line%%[![:space:]]*}"}"   # strip leading whitespace
            # Quoted values end at the closing quote; unquoted ones end at an
            # inline comment. Without this a real .env line like
            #   TTS_VOICE="Stina"   # Swedish pack
            # yielded 'Stina"   # Swedish pack' and was passed on as the voice.
            case "$line" in
                \"*) line="${line#\"}"; line="${line%%\"*}" ;;
                \'*) line="${line#\'}"; line="${line%%\'*}" ;;
                *)   line="${line%%#*}"
                     line="${line%"${line##*[![:space:]]}"}" ;;  # strip trailing whitespace
            esac
            [[ -n "$line" ]] && { printf '%s' "$line"; return; }
        fi
    fi
    printf '%s' "$default"
}

OLLAMA_URL="$(env_get OLLAMA_URL "http://localhost:11434")"
OLLAMA_MODEL="$(env_get OLLAMA_MODEL "gemma4:latest")"
TTS_URL="$(env_get TTS_URL "http://localhost:8880/v1/audio/speech")"
# Default to the engine that needs nothing outside this repo. The Swedish engines
# additionally require the separate swedish-kokoro checkout for their g2p module,
# so only select one automatically when that checkout is actually present.
if [[ -z "${VOICE_ENGINE:-}" ]]; then
    if [[ -n "${SWEDISH_KOKORO_PATH:-}" && -d "${SWEDISH_KOKORO_PATH:-}" ]]; then
        VOICE_ENGINE=kokoro-svml
    else
        VOICE_ENGINE=kokoro
    fi
fi
# Base Kokoro voices look like af_heart / bm_george; Swedish packs are names like
# Stina. Pick a default that matches the engine we are about to start.
if [[ "$VOICE_ENGINE" == "kokoro" ]]; then
    VOICE_VOICE="$(env_get TTS_VOICE "af_heart")"
else
    VOICE_VOICE="$(env_get TTS_VOICE "Stina")"
fi
WHISPER_MODEL="${WHISPER_MODEL:-base}"
# Languages the Swedish engine is allowed to speak; anything else is clamped to the
# first one, so a mis-heard snippet can't send the robot off into another language.
# Only 'sv' uses the fine-tuned voice packs — the rest use base Kokoro's own voices.
VOICE_LANGS="${KOKORO_SV_LANGS:-sv,en,fr,es,it}"

# Derive the voice server's origin and port from TTS_URL so one setting drives both.
VOICE_ORIGIN="$(printf '%s' "$TTS_URL" | sed -E 's#(https?://[^/]+).*#\1#')"
VOICE_PORT="$(printf '%s' "$VOICE_ORIGIN" | sed -E 's#.*:([0-9]+)$#\1#')"
[[ "$VOICE_PORT" == "$VOICE_ORIGIN" ]] && VOICE_PORT=8880

RUN_APP=1
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-app) RUN_APP=0 ;;
        --check)  CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

STARTED_PIDS=()
cleanup() {
    for pid in "${STARTED_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "Stopping pid $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

up() { curl -sf -m 2 -o /dev/null "$1"; }

wait_for() {
    local url="$1" name="$2" tries="${3:-60}"
    printf 'Waiting for %s' "$name"
    for ((i = 0; i < tries; i++)); do
        if up "$url"; then printf ' — up\n'; return 0; fi
        printf '.'
        sleep 2
    done
    printf '\nTimed out waiting for %s at %s\n' "$name" "$url" >&2
    return 1
}

# ---------- status ----------
if [[ $CHECK_ONLY -eq 1 ]]; then
    if up "$OLLAMA_URL/api/tags"; then
        echo "✓ Ollama up at $OLLAMA_URL"
        curl -sf "$OLLAMA_URL/api/tags" | grep -q "$OLLAMA_MODEL" \
            && echo "  ✓ $OLLAMA_MODEL pulled" \
            || echo "  ✗ $OLLAMA_MODEL NOT pulled — run: ollama pull $OLLAMA_MODEL"
    else
        echo "✗ Ollama DOWN at $OLLAMA_URL"
    fi
    if up "$VOICE_ORIGIN/health"; then
        echo "✓ Voice server up at $VOICE_ORIGIN"
        curl -sf "$VOICE_ORIGIN/health"; echo
    else
        echo "✗ Voice server DOWN at $VOICE_ORIGIN"
    fi
    echo "Would start: engine=$VOICE_ENGINE voice=$VOICE_VOICE whisper=$WHISPER_MODEL langs=$VOICE_LANGS"
    exit 0
fi

# ---------- 1. Ollama ----------
if up "$OLLAMA_URL/api/tags"; then
    echo "Ollama already running at $OLLAMA_URL"
elif [[ "$OLLAMA_URL" == *localhost* || "$OLLAMA_URL" == *127.0.0.1* ]]; then
    command -v ollama >/dev/null || { echo "ollama not installed — see https://ollama.com" >&2; exit 1; }
    echo "Starting Ollama..."
    OLLAMA_HOST=0.0.0.0 ollama serve >/tmp/reachy-ollama.log 2>&1 &
    STARTED_PIDS+=($!)
    wait_for "$OLLAMA_URL/api/tags" "Ollama"
else
    echo "Ollama at $OLLAMA_URL is not reachable, and it is remote — start it there." >&2
    exit 1
fi

if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "${OLLAMA_MODEL%%:*}"; then
    echo "Model $OLLAMA_MODEL is not pulled. Run: ollama pull $OLLAMA_MODEL" >&2
    exit 1
fi

# ---------- 2. Voice server ----------
if up "$VOICE_ORIGIN/health"; then
    echo "Voice server already running at $VOICE_ORIGIN"
elif [[ "$VOICE_ORIGIN" == *localhost* || "$VOICE_ORIGIN" == *127.0.0.1* ]]; then
    [[ -f voice-server/voice_server.py ]] || { echo "voice-server submodule missing — run: git submodule update --init" >&2; exit 1; }
    [[ -d voice-server/.venv ]] || { echo "voice-server not set up — run: cd voice-server && uv sync" >&2; exit 1; }
    echo "Starting voice server (engine=$VOICE_ENGINE, whisper=$WHISPER_MODEL, langs=$VOICE_LANGS)..."
    (
        cd voice-server
        # --langs is only read by the Swedish engines; base kokoro ignores it.
        PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python voice_server.py \
            --engine "$VOICE_ENGINE" --voice "$VOICE_VOICE" \
            --port "$VOICE_PORT" --whisper "$WHISPER_MODEL" --langs "$VOICE_LANGS"
    ) >/tmp/reachy-voice-server.log 2>&1 &
    STARTED_PIDS+=($!)
    # Kokoro + Whisper model loading is slow on a cold cache; allow ~4 minutes.
    wait_for "$VOICE_ORIGIN/health" "voice server" 120 || {
        echo "Last 20 lines of /tmp/reachy-voice-server.log:" >&2
        tail -20 /tmp/reachy-voice-server.log >&2
        exit 1
    }
else
    echo "Voice server at $VOICE_ORIGIN is not reachable, and it is remote — start it there." >&2
    exit 1
fi

curl -sf "$VOICE_ORIGIN/health" | grep -q '"stt":true' \
    || echo "Note: transcription is off (started without --whisper); history will keep raw audio."

# ---------- 3. App ----------
if [[ $RUN_APP -eq 0 ]]; then
    echo
    echo "Backends are up. Start the app with:  reachy-local-assistant"
    echo "Logs: /tmp/reachy-ollama.log  /tmp/reachy-voice-server.log"
    echo "Press Ctrl-C to stop the services this script started."
    wait
    exit 0
fi

echo
echo "Starting the app..."
# Deliberately not exec: the EXIT trap must still fire so the services this
# script started are stopped when the app exits.
reachy-local-assistant
