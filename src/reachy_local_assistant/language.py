"""Which language the conversation is being held in.

Detecting a language from a single reply works fine on a full sentence and badly
on the short ones a conversation is mostly made of — "Ja!", "Absolut.", "Sure!"
carry almost no signal, and with several languages enabled the detector lands on
the wrong one often enough to be audible (a Swedish "Absolut." spoken as French).

The conversation itself is the better evidence, and we already have it: Whisper
transcribes each user turn and reports the language it heard. This keeps a short
rolling window of that and offers the dominant language as a *hint* the voice
server may fall back on.

Turns are **not** equal evidence, so they are not counted equally:

* **Length.** A long sentence pins the language; a one-word turn barely moves it.
  Without this, four throwaway "Ja!"s would outvote the full French sentence the
  user just spoke — the hint would keep asserting Swedish after an obvious switch.
* **Recency.** Older turns fade, so the language someone is speaking *now* wins
  over what they were speaking a minute ago.

Deliberately a hint, not an instruction: it never overrides an explicit language
or a confident detection, so a mid-conversation switch is followed regardless.
"""

from __future__ import annotations
from typing import Deque, Tuple, Optional
from collections import deque


# Long enough to outvote a single odd turn, short enough to follow a real switch
# within a couple of exchanges.
DEFAULT_WINDOW = 5

# Word count at which a turn counts as full evidence. Capped so one monologue
# cannot pin the language for the rest of the conversation.
STRONG_WORDS = 8

# Weight a turn keeps per step into the past. 0.7 lets one long, recent statement
# outweigh several short older ones without erasing them entirely.
RECENCY_DECAY = 0.7

# Floor for a turn whose text we never saw — some evidence, but not much.
MIN_WEIGHT = 0.15


def normalize(lang: Optional[str]) -> str:
    """Normalise an STT language label to a short code (``""`` when unknown)."""
    value = (lang or "").strip().lower()
    if not value:
        return ""
    aliases = {"swedish": "sv", "svenska": "sv", "se": "sv", "english": "en", "eng": "en"}
    value = aliases.get(value, value)
    # Whisper may report a region ("en-us"); the voice server keys on the base code.
    return value.split("-")[0] if value not in ("en-gb", "en-us") else value


def evidence_weight(text: Optional[str]) -> float:
    """How much a turn of *text* should count, in ``[MIN_WEIGHT, 1.0]``.

    Grows with length up to ``STRONG_WORDS``: the same property that makes a
    detector confident makes a turn good evidence about the conversation.
    An unknown/empty text falls back to ``MIN_WEIGHT`` rather than 0 — we know
    *something* was said in that language, just not how much.
    """
    words = len((text or "").split())
    if not words:
        return MIN_WEIGHT
    return max(MIN_WEIGHT, min(words, STRONG_WORDS) / STRONG_WORDS)


class LanguageHistory:
    """Rolling, evidence-weighted record of the languages recent turns used."""

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        """Track at most *window* recent turns."""
        self._recent: Deque[Tuple[str, float]] = deque(maxlen=window)

    def record(self, lang: Optional[str], text: Optional[str] = None) -> None:
        """Note a turn's language, weighted by the *text* that was said in it.

        Unknown/empty languages are ignored; pass the transcript when there is one
        so a long statement counts for more than a one-word acknowledgement.
        """
        code = normalize(lang)
        if code:
            self._recent.append((code, evidence_weight(text)))

    def hint(self) -> str:
        """Return the best-supported recent language, or ``""`` when there is none.

        Scores each language by ``weight * RECENCY_DECAY ** turns_ago``, so one
        long recent statement beats a run of short older ones. Ties go to the most
        recent, so a switch is never stalled by an exact draw.
        """
        if not self._recent:
            return ""
        scores: dict[str, float] = {}
        newest_first = list(reversed(self._recent))
        for age, (code, weight) in enumerate(newest_first):
            scores[code] = scores.get(code, 0.0) + weight * (RECENCY_DECAY**age)
        best = max(scores.values())
        for code, _weight in newest_first:  # most recent among the tied
            # Float sums make exact equality unreliable; treat near-ties as ties.
            if scores[code] >= best - 1e-9:
                return code
        return ""

    def clear(self) -> None:
        """Forget the history (a new personality may speak a different language)."""
        self._recent.clear()
