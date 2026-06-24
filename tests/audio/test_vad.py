"""Tests for the WebRTC-VAD utterance segmenter."""

import numpy as np
import pytest

from reachy_mini_conversation_app.audio.vad import FRAME_SAMPLES, VAD_SAMPLE_RATE, VadSegmenter


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(VAD_SAMPLE_RATE * seconds), dtype=np.int16)


def _voiced(seconds: float) -> np.ndarray:
    """A loud 200 Hz tone — WebRTC VAD reliably flags it as speech."""
    t = np.arange(int(VAD_SAMPLE_RATE * seconds)) / VAD_SAMPLE_RATE
    return (np.sin(2 * np.pi * 200 * t) * 20000).astype(np.int16)


def _feed_all(seg: VadSegmenter, pcm: np.ndarray) -> list[np.ndarray]:
    """Feed audio in 20 ms frames, collecting any completed utterances."""
    out: list[np.ndarray] = []
    for i in range(0, len(pcm), FRAME_SAMPLES):
        out += seg.feed(pcm[i : i + FRAME_SAMPLES])
    return out


def test_pure_silence_yields_no_utterance():
    seg = VadSegmenter(silence_ms=300)
    assert _feed_all(seg, _silence(1.0)) == []
    assert not seg.in_speech


def test_speech_then_silence_emits_one_utterance():
    seg = VadSegmenter(silence_ms=300, min_speech_ms=100)
    pcm = np.concatenate([_voiced(0.8), _silence(0.6)])
    utterances = _feed_all(seg, pcm)
    assert len(utterances) == 1
    assert utterances[0].dtype == np.int16
    # Utterance covers roughly the voiced span (pre-roll + speech), not the full buffer.
    assert len(utterances[0]) >= int(VAD_SAMPLE_RATE * 0.5)
    assert not seg.in_speech  # closed after trailing silence


def test_in_speech_flag_tracks_state():
    seg = VadSegmenter(silence_ms=300, min_speech_ms=100)
    seg.feed(_voiced(0.2))  # start of an utterance, no closing silence yet
    assert seg.in_speech


def test_reset_drops_in_progress_utterance():
    seg = VadSegmenter(silence_ms=300)
    seg.feed(_voiced(0.3))
    assert seg.in_speech
    seg.reset()
    assert not seg.in_speech
    # After reset, trailing silence alone must not emit a stale utterance.
    assert _feed_all(seg, _silence(0.6)) == []


def test_sub_threshold_blip_is_discarded():
    seg = VadSegmenter(silence_ms=200, min_speech_ms=400)
    pcm = np.concatenate([_voiced(0.1), _silence(0.5)])  # 100 ms speech < 400 ms min
    assert _feed_all(seg, pcm) == []


def test_unaligned_chunks_are_buffered():
    """Chunks not aligned to 20 ms must not drop samples."""
    seg = VadSegmenter(silence_ms=300, min_speech_ms=100)
    pcm = np.concatenate([_voiced(0.8), _silence(0.6)])
    out: list[np.ndarray] = []
    for i in range(0, len(pcm), 137):  # deliberately odd chunk size
        out += seg.feed(pcm[i : i + 137])
    assert len(out) == 1


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_non_int16_input_is_coerced(dtype):
    seg = VadSegmenter(silence_ms=300, min_speech_ms=100)
    pcm = np.concatenate([_voiced(0.8), _silence(0.6)]).astype(dtype)
    out = _feed_all(seg, pcm)
    assert len(out) == 1 and out[0].dtype == np.int16
