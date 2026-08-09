"""Tests for the incomplete-profile fallback (ported from upstream #442)."""

from pathlib import Path

import pytest

from reachy_local_assistant import prompts
from reachy_local_assistant.config import config


@pytest.fixture()
def profiles_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the profile machinery at a temp directory."""
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    return tmp_path


def test_broken_profile_falls_back_instead_of_exiting(
    profiles_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile with no instructions.txt previously called sys.exit(1).

    SystemExit is a BaseException, so it sailed past apply_personality's
    `except Exception` and killed the app when applied from the settings page.
    """
    (profiles_root / "broken").mkdir()
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "broken")

    instructions = prompts.get_session_instructions()  # must not raise SystemExit

    assert instructions.strip(), "fallback must produce a usable prompt"


def test_empty_instructions_also_fall_back(profiles_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty instructions.txt is as broken as a missing one."""
    profile = profiles_root / "empty"
    profile.mkdir()
    (profile / "instructions.txt").write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "empty")

    assert prompts.get_session_instructions().strip()


def test_a_valid_profile_is_still_used(profiles_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback must not shadow profiles that are fine."""
    profile = profiles_root / "pirate"
    profile.mkdir()
    (profile / "instructions.txt").write_text("You are a pirate robot.", encoding="utf-8")
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "pirate")

    assert "pirate" in prompts.get_session_instructions()
