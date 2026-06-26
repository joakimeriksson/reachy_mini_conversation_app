"""Tests for the sentence splitter used by streaming TTS."""

from reachy_local_assistant.audio.text_chunk import split_sentences


def test_empty_and_whitespace_return_empty():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_single_sentence_is_one_chunk():
    assert split_sentences("Hej världen") == ["Hej världen"]


def test_splits_on_sentence_punctuation():
    out = split_sentences("Hej! Jag heter Reachy. Vad kul? Ja.")
    assert out == ["Hej!", "Jag heter Reachy.", "Vad kul?", "Ja."]


def test_long_sentence_is_hard_wrapped_under_max_chars():
    text = "ord " * 200  # 800 chars, no terminal punctuation
    out = split_sentences(text, max_chars=50)
    assert len(out) > 1
    assert all(len(c) <= 50 for c in out)
    # no words lost / mangled
    assert " ".join(out).split() == text.split()


def test_each_chunk_within_kokoro_token_budget():
    long_reply = (
        "Hej allihopa! Jag heter Candytron, och jag är en liten robot som älskar "
        "att dela ut godis. Idag har jag choklad, lakrits och kola i min magvälva. "
        "Vi kan dansa, sjunga och berätta roliga historier tillsammans!"
    )
    out = split_sentences(long_reply)
    assert len(out) >= 3
    assert all(len(c) <= 300 for c in out)
