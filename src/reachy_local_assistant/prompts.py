import re
import logging
from pathlib import Path

from reachy_local_assistant.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_local_assistant.memory import format_memory_for_prompt


logger = logging.getLogger(__name__)


PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r'^\[([a-zA-Z0-9/_-]+)\]$')

    lines = content.split('\n')
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return '\n'.join(expanded_lines)


def get_session_instructions(instance_path: str | Path | None = None) -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set.

    Long-term memory facts (if any) are prepended so the model recalls them.
    """
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.info(f"Loading default prompt from {PROMPTS_LIBRARY_DIRECTORY / 'default_prompt.txt'}")
        instructions_file = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    else:
        if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.info(
                "Loading prompt from external profile '%s' (root=%s)",
                profile,
                config.PROFILES_DIRECTORY,
            )
        else:
            logger.info(f"Loading prompt from profile '{profile}'")
        instructions_file = config.PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME

    instructions = _read_instructions_file(instructions_file, profile or "default")
    if instructions is None and profile:
        # Fall back to the default prompt rather than dying: this runs not only at
        # startup but from the settings page's "Apply", where a sys.exit(1) (a
        # BaseException) would sail past the caller's `except Exception` and kill
        # the whole app over one broken profile folder.
        fallback = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
        logger.warning("Profile '%s' has no usable %s; using the default prompt", profile, INSTRUCTIONS_FILENAME)
        instructions = _read_instructions_file(fallback, "default")
    if instructions is None:
        logger.error("No usable instructions found (profile %r); using a minimal built-in prompt", profile)
        instructions = "You are Reachy Mini, a small friendly desk robot. Keep replies short and conversational."

    # Expand [<name>] placeholders with content from prompts library
    expanded_instructions = _expand_prompt_includes(instructions)
    memory_prompt = format_memory_for_prompt(instance_path)
    if memory_prompt:
        return f"{memory_prompt}\n\n{expanded_instructions}"
    return expanded_instructions


def _read_instructions_file(instructions_file: Path, profile_name: str) -> str | None:
    """Read a profile's instructions; None (with a warning) when missing/empty/unreadable."""
    try:
        if not instructions_file.exists():
            logger.warning("Profile '%s' has no %s", profile_name, INSTRUCTIONS_FILENAME)
            return None
        instructions = instructions_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as e:
        logger.warning("Failed to load instructions from profile '%s': %s", profile_name, e)
        return None
    if not instructions:
        logger.warning("Profile '%s' has empty %s", profile_name, INSTRUCTIONS_FILENAME)
        return None
    return instructions


def get_session_voice(default: str | None = None) -> str:
    """Resolve the voice to use for the session, for the active TTS backend.

    If a custom profile is selected and contains a ``voice.txt``, return its
    trimmed content; otherwise fall back to *default* or the configured
    ``TTS_VOICE`` (the voice-server voice, e.g. "Stina").
    """
    if default is None:
        default = config.TTS_VOICE
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return default
    try:
        voice_file = config.PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or default
    except Exception:
        pass
    return default
