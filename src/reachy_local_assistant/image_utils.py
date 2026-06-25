"""Image helpers with no OpenCV dependency (so the robot wheel stays cv2-free)."""

from __future__ import annotations
import io
from typing import Any

import numpy as np
from numpy.typing import NDArray


def encode_jpeg(frame_bgr: NDArray[Any], quality: int = 80) -> bytes:
    """Encode a BGR (OpenCV-convention) uint8 frame to JPEG bytes via Pillow.

    Camera frames from ``robot.media`` and ``cv2.VideoCapture`` are BGR; Pillow
    expects RGB, so the channels are swapped before encoding.
    """
    from PIL import Image  # Pillow is a core dep (pulled by gradio); lazy-imported anyway

    arr = np.asarray(frame_bgr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # BGR -> RGB
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
