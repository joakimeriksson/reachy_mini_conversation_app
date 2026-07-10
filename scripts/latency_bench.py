"""Local latency benchmark for the conversation pipeline (no robot needed).

Drives the SAME components the robot's main loop uses — Gemma (direct-audio or
STT+chat) + the remote voice server — and reports per-stage latency:

  * LLM   : audio -> reply text  (direct-audio one call, or STT + chat)
  * TTS   : reply text -> first spoken sentence (time to first audio)
  * total : what the user perceives from "you stopped talking" to "robot speaks"

Run:  .venv/bin/python scripts/latency_bench.py
Needs Ollama + the voice server up (see .env: OLLAMA_URL, TTS_URL).
"""
from __future__ import annotations
import time
import asyncio

import numpy as np
from dotenv import load_dotenv


load_dotenv()

from reachy_local_assistant.config import config  # noqa: E402
from reachy_local_assistant.audio.tts import make_tts  # noqa: E402
from reachy_local_assistant.audio.gemma_stt import GemmaSTT  # noqa: E402
from reachy_local_assistant.llm.ollama_chat import OllamaChat  # noqa: E402
from reachy_local_assistant.prompts import get_session_instructions  # noqa: E402
from reachy_local_assistant.conversation.turn import generate_reply  # noqa: E402
from reachy_local_assistant.conversation.speech import stream_sentences  # noqa: E402
from reachy_local_assistant.audio.dsp import resample_int16  # noqa: E402
from reachy_local_assistant.audio.vad import VAD_SAMPLE_RATE  # noqa: E402


PROMPTS = [
    ("sv", "Hej Reachy, hur mår du idag?"),
    ("en", "What's the weather like on Mars?"),
    ("sv", "Berätta en kort rolig grej."),
    ("en", "Can you help me with something quick?"),
]


def synth_user_audio(tts, text: str) -> np.ndarray:
    """Synthesize the *user* prompt via the voice server -> 16 kHz int16 (what the mic feeds)."""
    chunks = list(tts.synthesize(text, voice="Stina", language=None))
    sr = chunks[0][0]
    pcm = np.concatenate([c[1] for c in chunks])
    return resample_int16(pcm, sr, VAD_SAMPLE_RATE)


async def main() -> None:
    tts = make_tts()
    stt = GemmaSTT(config.OLLAMA_STT_MODEL, config.OLLAMA_URL)
    instr = get_session_instructions()
    if config.OLLAMA_DIRECT_AUDIO:
        instr += (
            "\n\nThe user speaks to you through attached audio. Listen and respond "
            "directly and naturally to what they say — never transcribe or repeat it back."
        )
    # enable_tools=False: measure the core turn latency (a plain reply, no tool round-trip).
    chat = OllamaChat(config.OLLAMA_MODEL, config.OLLAMA_URL, deps=None, system_prompt=instr, enable_tools=False)

    print(f"model={config.OLLAMA_MODEL}  direct_audio={config.OLLAMA_DIRECT_AUDIO}  tts={config.TTS_URL}")
    print(f"{'turn':>4} {'lang':>4} {'LLM s':>7} {'TTS1 s':>7} {'total s':>8}   reply")
    llm_t, tts_t, tot_t = [], [], []
    for i, (lang, text) in enumerate(PROMPTS, 1):
        audio = synth_user_audio(tts, text)
        chat.reset()  # fresh turn, no history carry-over

        t0 = time.perf_counter()
        turn = await generate_reply(
            stt, chat, audio, direct_audio=config.OLLAMA_DIRECT_AUDIO, sample_rate=VAD_SAMPLE_RATE
        )
        t_llm = time.perf_counter() - t0

        t1 = time.perf_counter()
        async for _sr, _pcm in stream_sentences(tts, turn.reply, voice="Stina", language=turn.language, should_stop=lambda: False):
            break  # first spoken sentence = time-to-first-audio
        t_tts = time.perf_counter() - t1

        total = t_llm + t_tts
        llm_t.append(t_llm); tts_t.append(t_tts); tot_t.append(total)
        print(f"{i:>4} {lang:>4} {t_llm:>7.2f} {t_tts:>7.2f} {total:>8.2f}   {turn.reply[:60]!r}")

    n = len(tot_t)
    print(f"{'avg':>4} {'':>4} {sum(llm_t)/n:>7.2f} {sum(tts_t)/n:>7.2f} {sum(tot_t)/n:>8.2f}")
    print("\nLLM = audio->reply; TTS1 = reply->first spoken sentence; total = perceived lag after you stop talking.")


if __name__ == "__main__":
    asyncio.run(main())
