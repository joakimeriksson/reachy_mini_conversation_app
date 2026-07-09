"""Tiny in-tree replacements for the handful of ``fastrtc`` helpers we used.

The conversation app runs headless on the robot: audio flows through the robot's
own media pipeline (``robot.media``) and the loop drives ``receive()``/``emit()``
directly from :class:`console.LocalStream`. It never used fastrtc's browser
streaming engine on that path — only three small utilities and a base class.

fastrtc (via gradio<6) caps ``pydantic<=2.12.3``, which is incompatible with
``reachy-mini>=1.8.4`` (needs ``pydantic>=2.12.5``), so the whole gradio/fastrtc
stack was dropped (matching upstream). These shims keep the headless path working
without that dependency.
"""

from __future__ import annotations
import asyncio
from typing import Any

import numpy as np
from numpy.typing import NDArray


class AdditionalOutputs:
    """Marker wrapping non-audio outputs (chat transcript messages).

    Mirrors ``fastrtc.AdditionalOutputs``: the positional args are stashed on
    ``.args`` and drained by the player loop to log role/content.
    """

    def __init__(self, *args: Any) -> None:
        self.args = args


def audio_to_float32(audio: NDArray[np.int16 | np.float32]) -> NDArray[np.float32]:
    """Convert int16 PCM to float32 in ``[-1.0, 1.0)`` (float32 passes through)."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.float32:
        return audio
    raise TypeError(f"Unsupported audio data type: {audio.dtype}")


async def wait_for_item(queue: "asyncio.Queue[Any]", timeout: float = 0.1) -> Any:
    """Await a queue item, returning ``None`` on timeout so ``emit`` never blocks."""
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        return None


class AsyncStreamHandler:
    """Minimal base for the conversation handler (no fastrtc streaming engine).

    Only the constructor surface the handler relies on is kept; it defines its own
    ``output_queue`` and the ``receive``/``emit``/``start_up``/``shutdown``/``copy``
    methods, which ``console.LocalStream`` calls directly.
    """

    def __init__(
        self,
        expected_layout: str = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int | None = None,
        input_sample_rate: int = 48000,
        fps: int = 30,
    ) -> None:
        self.expected_layout = expected_layout
        self.output_sample_rate = output_sample_rate
        self.output_frame_size = output_frame_size
        self.input_sample_rate = input_sample_rate
        self.fps = fps
