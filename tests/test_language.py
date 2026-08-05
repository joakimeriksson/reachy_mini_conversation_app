"""Tests for the conversation-language hint."""

from reachy_local_assistant.language import LanguageHistory, normalize, evidence_weight


LONG_SV = "Hej Reachy, jag skulle vilja att du berättar något roligt för mig idag."
LONG_FR = "Bonjour, je voudrais que tu me parles en français à partir de maintenant."
LONG_EN = "Hello there, could you please switch over and talk to me in English now?"


def test_no_history_yields_no_hint() -> None:
    """A fresh conversation must not assert a language it has no evidence for."""
    assert LanguageHistory().hint() == ""


def test_dominant_language_wins() -> None:
    """One odd turn must not flip the conversation's language."""
    history = LanguageHistory()
    for text in (LONG_SV, LONG_SV, LONG_SV):
        history.record("sv", text)
    history.record("en", "Okay.")

    assert history.hint() == "sv"


def test_one_long_statement_beats_several_short_ones() -> None:
    """The point of weighting: a full sentence outweighs throwaway replies.

    Counting turns equally made the hint answer "sv" immediately after the user
    said a whole sentence in French — the switch was invisible to it.
    """
    history = LanguageHistory()
    for _ in range(4):
        history.record("sv", "Ja!")
    history.record("fr", LONG_FR)

    assert history.hint() == "fr"


def test_a_short_aside_does_not_derail_a_long_conversation() -> None:
    """The converse: one stray word must not flip a well-established language."""
    history = LanguageHistory()
    for _ in range(4):
        history.record("sv", LONG_SV)
    history.record("fr", "Oui.")

    assert history.hint() == "sv"


def test_recency_breaks_equal_evidence() -> None:
    """Same weight on both sides — what the user speaks now should win."""
    history = LanguageHistory()
    history.record("sv", LONG_SV)
    history.record("en", LONG_EN)

    assert history.hint() == "en"


def test_a_sustained_switch_is_followed() -> None:
    """A real language change must take over within a couple of exchanges."""
    history = LanguageHistory(window=5)
    for text in (LONG_SV, LONG_SV):
        history.record("sv", text)
    for text in (LONG_EN, LONG_EN, LONG_EN):
        history.record("en", text)

    assert history.hint() == "en"


def test_old_turns_fall_out_of_the_window() -> None:
    """The hint tracks the recent conversation, not the whole session."""
    history = LanguageHistory(window=3)
    for _ in range(3):
        history.record("sv", LONG_SV)
    for _ in range(3):
        history.record("fr", LONG_FR)

    assert history.hint() == "fr"


def test_unknown_values_are_ignored() -> None:
    """Whisper returns "" when it cannot tell; that must not pollute the history."""
    history = LanguageHistory()
    history.record("sv", LONG_SV)
    history.record("", LONG_FR)
    history.record(None, LONG_FR)
    history.record("   ", LONG_FR)

    assert history.hint() == "sv"


def test_a_turn_with_no_transcript_still_counts() -> None:
    """Without Whisper there is no text, but the language is still evidence."""
    history = LanguageHistory()
    history.record("sv")

    assert history.hint() == "sv"


def test_clear_forgets_everything() -> None:
    """A personality switch may change language; the old hint must not fight it."""
    history = LanguageHistory()
    history.record("sv", LONG_SV)
    history.clear()

    assert history.hint() == ""


def test_evidence_weight_grows_with_length_and_is_capped() -> None:
    """Longer turns count for more, up to a ceiling."""
    assert evidence_weight("Ja!") < evidence_weight("Jag vet inte riktigt.")
    assert evidence_weight(LONG_SV) == 1.0
    # A monologue must not outweigh everything that follows it, forever.
    assert evidence_weight(LONG_SV + " " + LONG_SV) == 1.0
    assert 0.0 < evidence_weight("") <= 1.0


def test_normalize_maps_labels_and_regions() -> None:
    """STT backends spell languages inconsistently."""
    assert normalize("Swedish") == "sv"
    assert normalize("svenska") == "sv"
    assert normalize("SV") == "sv"
    assert normalize("english") == "en"
    assert normalize("pt-br") == "pt"
    assert normalize("en-gb") == "en-gb"  # a real Kokoro voice, not a region to strip
    assert normalize(None) == ""
