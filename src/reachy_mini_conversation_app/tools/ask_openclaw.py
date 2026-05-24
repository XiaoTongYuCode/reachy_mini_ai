from __future__ import annotations
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.openclaw_bridge import OpenClawBridge
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class AskOpenClaw(Tool):
    """Forward complex requests to an OpenClaw agent."""

    name = "ask_openclaw"
    description = (
        "Ask the user's OpenClaw agent for help with requests that need OpenClaw's external brain: "
        "long-term OpenClaw memory, cross-channel context, external systems, long-running agent tasks, "
        "or tools that are not available locally. Prefer local robot tools for movement, camera, weather, "
        "search, and email when they are sufficient."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's request to forward to OpenClaw.",
            },
            "include_camera_image": {
                "type": "boolean",
                "description": (
                    "Whether the current camera frame is relevant. This v1 bridge is text-only and will "
                    "not send image bytes to OpenClaw; use the local camera tool for actual vision."
                ),
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Forward the query to OpenClaw and return a structured tool result."""
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return {"error": "query must be a non-empty string"}

        gateway_url = (config.OPENCLAW_GATEWAY_URL or "").strip()
        gateway_token = (config.OPENCLAW_TOKEN or "").strip()
        if not gateway_url or not gateway_token:
            return {
                "error": (
                    "OpenClaw gateway is not configured. Set OPENCLAW_GATEWAY_URL and OPENCLAW_TOKEN, "
                    "or disable ask_openclaw."
                ),
                "source": "openclaw",
            }

        include_camera_image = _bool_arg(kwargs.get("include_camera_image"), default=False)
        image_note = ""
        image_available = False
        if include_camera_image:
            image_available = _has_camera_frame(deps)
            image_note = (
                "\n\n[Robot camera requested: a current frame is available, but this v1 OpenClaw bridge "
                "is text-only and did not attach image bytes. Use the local camera tool for visual analysis.]"
                if image_available
                else "\n\n[Robot camera requested, but no current camera frame is available.]"
            )

        bridge = OpenClawBridge()
        try:
            connected = await bridge.connect()
            if not connected:
                return {
                    "error": f"OpenClaw gateway is unreachable at {gateway_url}",
                    "source": "openclaw",
                    "image_requested": include_camera_image,
                    "image_attached": False,
                }

            logger.info("Tool call: ask_openclaw query=%r image_requested=%s", query[:120], include_camera_image)
            response = await bridge.chat(
                query + image_note,
                system_context=(
                    "The user is speaking through a Reachy Mini robot. Answer concisely for voice. "
                    "Do not claim you received an image unless the message explicitly says image bytes were attached."
                ),
            )
            if response.error:
                return {
                    "error": response.error,
                    "source": "openclaw",
                    "image_requested": include_camera_image,
                    "image_attached": False,
                }
            return {
                "response": response.content,
                "source": "openclaw",
                "image_requested": include_camera_image,
                "image_available": image_available,
                "image_attached": False,
            }
        except Exception as e:
            logger.warning("ask_openclaw failed: %s", e)
            return {
                "error": f"OpenClaw request failed: {type(e).__name__}: {e}",
                "source": "openclaw",
                "image_requested": include_camera_image,
                "image_attached": False,
            }
        finally:
            await bridge.disconnect()


def _bool_arg(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _has_camera_frame(deps: ToolDependencies) -> bool:
    camera_worker = getattr(deps, "camera_worker", None)
    if camera_worker is None:
        return False
    get_latest_frame = getattr(camera_worker, "get_latest_frame", None)
    if not callable(get_latest_frame):
        return False
    try:
        return get_latest_frame() is not None
    except Exception:
        return False
