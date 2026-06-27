import os
import sys
import logging
from pathlib import Path
from importlib.resources import files

from dotenv import find_dotenv, load_dotenv


# Locked profile: set to a profile name (e.g., "astronomer") to lock the app
# to that profile and disable all profile switching. Leave as None for normal behavior.
LOCKED_PROFILE: str | None = None
PROJECT_ROOT = Path(__file__).parents[2].resolve()


def _is_source_checkout_root(root: Path) -> bool:
    """Return whether the given root looks like this project's source checkout."""
    return (root / "pyproject.toml").is_file() and (root / "src" / "reachy_local_assistant").is_dir()


def _packaged_profiles_directory() -> Path | None:
    """Return the installed wheel's packaged profiles directory when available."""
    try:
        return Path(str(files("reachy_local_data").joinpath("profiles")))
    except Exception:
        return None


def _resolve_default_profiles_directory() -> Path:
    """Resolve built-in profiles from source checkout or installed package data."""
    source_profiles = PROJECT_ROOT / "profiles"
    if _is_source_checkout_root(PROJECT_ROOT) and source_profiles.is_dir():
        return source_profiles

    packaged_profiles = _packaged_profiles_directory()
    if packaged_profiles is not None and packaged_profiles.is_dir():
        return packaged_profiles

    return source_profiles


DEFAULT_PROFILES_DIRECTORY = _resolve_default_profiles_directory()

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag.

    Accepted truthy values: 1, true, yes, on
    Accepted falsy values: 0, false, no, off
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid boolean value for %s=%r, using default=%s", name, raw, default)
    return default


def _env_int(name: str, default: int) -> int:
    """Parse an integer env var, falling back to *default* on missing/invalid."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r, using default=%s", name, raw, default)
        return default


def _env_opt_float(name: str) -> float | None:
    """Parse an optional float env var; None when unset (use the library default)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r, ignoring", name, raw)
        return None


def _collect_profile_names(profiles_root: Path) -> set[str]:
    """Return profile folder names from a profiles root directory."""
    if not profiles_root.exists() or not profiles_root.is_dir():
        return set()
    return {p.name for p in profiles_root.iterdir() if p.is_dir()}


def _collect_tool_module_names(tools_root: Path) -> set[str]:
    """Return tool module names from a tools directory."""
    if not tools_root.exists() or not tools_root.is_dir():
        return set()
    ignored = {"__init__", "core_tools"}
    return {
        p.stem
        for p in tools_root.glob("*.py")
        if p.is_file() and p.stem not in ignored
    }


def _raise_on_name_collisions(
    *,
    label: str,
    external_root: Path,
    internal_root: Path,
    external_names: set[str],
    internal_names: set[str],
) -> None:
    """Raise with a clear message when external/internal names collide."""
    collisions = sorted(external_names & internal_names)
    if not collisions:
        return

    raise RuntimeError(
        f"Config.__init__(): Ambiguous {label} names found in both external and built-in libraries: {collisions}. "
        f"External {label} root: {external_root}. Built-in {label} root: {internal_root}. "
        f"Please rename the conflicting external {label}(s) to continue."
    )


# Validate LOCKED_PROFILE at startup
if LOCKED_PROFILE is not None:
    _profiles_dir = DEFAULT_PROFILES_DIRECTORY
    _profile_path = _profiles_dir / LOCKED_PROFILE
    _instructions_file = _profile_path / "instructions.txt"
    if not _profile_path.is_dir():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' does not exist in {_profiles_dir}", file=sys.stderr)
        sys.exit(1)
    if not _instructions_file.is_file():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' has no instructions.txt", file=sys.stderr)
        sys.exit(1)

_skip_dotenv = _env_flag("REACHY_MINI_SKIP_DOTENV", default=False)

if _skip_dotenv:
    logger.info("Skipping .env loading because REACHY_MINI_SKIP_DOTENV is set")
else:
    # Locate .env file (search upward from current working directory)
    dotenv_path = find_dotenv(usecwd=True)

    if dotenv_path:
        # Load .env and override environment variables
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"Configuration loaded from {dotenv_path}")
    else:
        logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Optional (HuggingFace cache for robot emotion + dance datasets)
    HF_HOME = os.getenv("HF_HOME", "./cache")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"HF_HOME: {HF_HOME}")

    # Filesystem root containing profile directories, not a Python import path.
    _profiles_directory_env = os.getenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY")
    PROFILES_DIRECTORY = Path(_profiles_directory_env) if _profiles_directory_env else DEFAULT_PROFILES_DIRECTORY
    _tools_directory_env = os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    TOOLS_DIRECTORY = Path(_tools_directory_env) if _tools_directory_env else None
    AUTOLOAD_EXTERNAL_TOOLS = _env_flag("AUTOLOAD_EXTERNAL_TOOLS", default=False)
    REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    MCP_SERVER_URLS = os.getenv("MCP_SERVER_URLS")

    # --- Local backend: Ollama (Gemma audio LLM/STT) + Piper TTS ---
    # Base URL of the Ollama server. The native API lives at this root; the
    # OpenAI-compatible chat API at "<url>/v1".
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    # Conversation model — must support tool calling (and, for direct audio STT,
    # audio input). The tag is whatever the user has pulled locally.
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")
    # Model used for speech-to-text. Defaults to OLLAMA_MODEL (one audio-capable
    # model for both). Override to use a separate STT model if desired.
    OLLAMA_STT_MODEL = os.getenv("OLLAMA_STT_MODEL") or OLLAMA_MODEL
    # Generation tuning (passed to Ollama via `options`).
    OLLAMA_TEMPERATURE = _env_opt_float("OLLAMA_TEMPERATURE")  # None -> model default
    OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 0)  # 0 -> model default
    # How long Ollama keeps the model loaded between calls (e.g. "5m", "-1" = forever).
    OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    # Enable chain-of-thought "thinking" for the chat model (slower, sometimes better).
    OLLAMA_THINK = _env_flag("OLLAMA_THINK", default=False)
    # Feed the user's audio straight into the chat model (one call: speech -> reply)
    # instead of a separate STT pass. ~2x lower LLM latency and the model hears tone,
    # but there's no separate transcript and TTS language is auto-detected from the reply.
    OLLAMA_DIRECT_AUDIO = _env_flag("OLLAMA_DIRECT_AUDIO", default=False)

    # --- Text-to-speech backend ---
    # "piper" = local Piper (default); "remote" = external voice generator via an
    # OpenAI-compatible /v1/audio/speech endpoint (thin-client / on-prem setup).
    TTS_BACKEND = (os.getenv("TTS_BACKEND", "piper") or "piper").strip().lower()
    # Remote TTS (when TTS_BACKEND=remote): full speech endpoint URL.
    TTS_URL = os.getenv("TTS_URL", "")  # e.g. http://voicehost:8880/v1/audio/speech
    TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
    TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
    TTS_FORMAT = os.getenv("TTS_FORMAT", "wav")  # wav/flac/ogg (decoded via soundfile)
    TTS_API_KEY = os.getenv("TTS_API_KEY")  # optional bearer token

    # --- Piper TTS (when TTS_BACKEND=piper) ---
    # Voice (ONNX): a voice name resolvable in PIPER_DATA_DIR or an absolute .onnx
    # path. Profiles may override via their voice.txt.
    PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
    PIPER_DATA_DIR = os.getenv("PIPER_DATA_DIR")  # dir holding downloaded Piper voices
    # Synthesis tuning (None -> the voice's built-in default).
    PIPER_LENGTH_SCALE = _env_opt_float("PIPER_LENGTH_SCALE")  # >1 slower, <1 faster
    PIPER_NOISE_SCALE = _env_opt_float("PIPER_NOISE_SCALE")    # expressiveness
    PIPER_NOISE_W_SCALE = _env_opt_float("PIPER_NOISE_W_SCALE")  # phoneme-length variation
    PIPER_VOLUME = _env_opt_float("PIPER_VOLUME")             # 1.0 = unchanged
    PIPER_SPEAKER_ID = _env_int("PIPER_SPEAKER_ID", -1)      # -1 -> default speaker

    # VAD tuning (see audio/vad.py).
    VAD_AGGRESSIVENESS = _env_int("VAD_AGGRESSIVENESS", 2)
    VAD_SILENCE_MS = _env_int("VAD_SILENCE_MS", 800)
    VAD_MIN_SPEECH_MS = _env_int("VAD_MIN_SPEECH_MS", 200)
    VAD_MAX_UTTERANCE_MS = _env_int("VAD_MAX_UTTERANCE_MS", 15000)

    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    def __init__(self) -> None:
        """Initialize the configuration."""
        if self.REACHY_MINI_CUSTOM_PROFILE and self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            selected_profile_path = self.PROFILES_DIRECTORY / self.REACHY_MINI_CUSTOM_PROFILE
            if not selected_profile_path.is_dir():
                available_profiles = sorted(_collect_profile_names(self.PROFILES_DIRECTORY))
                raise RuntimeError(
                    "Config.__init__(): Selected profile "
                    f"'{self.REACHY_MINI_CUSTOM_PROFILE}' was not found in external profiles root "
                    f"{self.PROFILES_DIRECTORY}. "
                    f"Available external profiles: {available_profiles}. "
                    "Either set 'REACHY_MINI_CUSTOM_PROFILE' to one of the available external profiles "
                    "or unset 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' to use built-in profiles."
                )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            external_profiles = _collect_profile_names(self.PROFILES_DIRECTORY)
            internal_profiles = _collect_profile_names(DEFAULT_PROFILES_DIRECTORY)
            _raise_on_name_collisions(
                label="profile",
                external_root=self.PROFILES_DIRECTORY,
                internal_root=DEFAULT_PROFILES_DIRECTORY,
                external_names=external_profiles,
                internal_names=internal_profiles,
            )

        if self.TOOLS_DIRECTORY is not None:
            builtin_tools_root = Path(__file__).parent / "tools"
            external_tools = _collect_tool_module_names(self.TOOLS_DIRECTORY)
            internal_tools = _collect_tool_module_names(builtin_tools_root)
            _raise_on_name_collisions(
                label="tool",
                external_root=self.TOOLS_DIRECTORY,
                internal_root=builtin_tools_root,
                external_names=external_tools,
                internal_names=internal_tools,
            )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.warning(
                "Environment variable 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' is set. "
                "Profiles (instructions.txt, ...) will be loaded from %s.",
                self.PROFILES_DIRECTORY,
            )
        else:
            logger.info(
                "'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' is not set. "
                "Using built-in profiles from %s.",
                DEFAULT_PROFILES_DIRECTORY,
            )

        if self.TOOLS_DIRECTORY is not None:
            logger.warning(
                "Environment variable 'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY' is set. "
                "External tools will be loaded from %s.",
                self.TOOLS_DIRECTORY,
            )
        else:
            logger.info(
                "'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY' is not set. "
                "Using built-in shared tools only."
            )


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    if LOCKED_PROFILE is not None:
        return
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
