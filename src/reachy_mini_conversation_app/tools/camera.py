import math
import base64
import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.config import ALIYUN_BACKEND, get_backend_choice
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg


logger = logging.getLogger(__name__)

_MAX_SEQUENCE_SECONDS = 120


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
            "duration_seconds": {
                "type": "number",
                "description": (
                    "Optional duration for a 1 FPS camera sequence. Use 1 for a single snapshot. "
                    "Values are capped at 120 seconds."
                ),
                "minimum": 1,
                "maximum": _MAX_SEQUENCE_SECONDS,
                "default": 1,
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

        duration_seconds = _coerce_duration_seconds(kwargs.get("duration_seconds"))
        logger.info("Tool call: camera question=%s duration_seconds=%s", question[:120], duration_seconds)

        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        if deps.vision_processor is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_processor.process_image,
                frame,
                question,
            )
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        if duration_seconds <= 1 or get_backend_choice() != ALIYUN_BACKEND:
            jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
            return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}

        jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
        return {
            "camera_sequence_requested": True,
            "b64_im": base64.b64encode(jpeg_bytes).decode("utf-8"),
            "fps": 1,
            "duration_seconds": duration_seconds,
        }


def _coerce_duration_seconds(value: Any) -> int:
    """Return a bounded integer duration for the 1 FPS camera sequence."""
    try:
        seconds = math.ceil(float(value)) if value is not None else 1
    except (TypeError, ValueError):
        seconds = 1
    return max(1, min(_MAX_SEQUENCE_SECONDS, seconds))
