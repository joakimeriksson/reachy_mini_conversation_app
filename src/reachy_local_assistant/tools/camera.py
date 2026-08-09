import base64
import logging
from typing import Any, Dict

from reachy_local_assistant.image_utils import encode_jpeg
from reachy_local_assistant.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    # Wordy on purpose (ported from upstream #454): the model under-used the terse
    # description and asked "what should I look at?" instead of just looking.
    description = (
        "Take a picture with the camera to see what is in front of the robot. "
        "Use this when the user asks you to look at something, see what they are holding, "
        "check their appearance, describe the scene, or comment on how they look. "
        "Also use it when the user asks what you can see or wants your visual opinion. "
        "The camera is live; each call captures the current moment. "
        "If the user asks you to look without saying at what, do not ask for clarification — "
        "call this tool and describe what you see."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", question[:120])

        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        # Encode to JPEG via Pillow (keeps the robot wheel free of OpenCV). The
        # frame is handed to the multimodal LLM (Gemma) — the handler injects it
        # into the chat so the model "sees" it directly.
        b64_encoded = base64.b64encode(encode_jpeg(frame)).decode("utf-8")
        return {"b64_im": b64_encoded}
