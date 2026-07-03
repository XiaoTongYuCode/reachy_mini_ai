"""Cancel running Aliyun camera sequence tasks."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.tool_constants import ToolState


if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)

ALIYUN_CAMERA_SEQUENCE_TOOL_NAME = "aliyun_camera_sequence"


class CancelAliyunCameraSequence(Tool):
    """Cancel a running Aliyun camera image sequence."""

    name = "cancel_aliyun_camera_sequence"
    description = (
        "Cancel a running Aliyun camera image sequence. "
        "Use this when the user asks to stop watching, stop the camera sequence, or cancel visual streaming."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "tool_id": {
                "type": "string",
                "description": "Optional specific Aliyun camera sequence tool ID. If omitted, cancels the latest running sequence.",
            }
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Cancel the latest or specified Aliyun camera sequence."""
        tool_manager: BackgroundToolManager | None = kwargs.get("tool_manager")
        if tool_manager is None:
            return {"error": "Tool manager is required."}

        requested_tool_id = str(kwargs.get("tool_id") or "").strip()
        if requested_tool_id:
            tool = tool_manager.get_tool(requested_tool_id)
            if tool is None:
                return {"error": f"Tool {requested_tool_id} not found."}
            if tool.tool_name != ALIYUN_CAMERA_SEQUENCE_TOOL_NAME:
                return {
                    "error": (
                        f"Tool {requested_tool_id} is '{tool.tool_name}', not "
                        f"'{ALIYUN_CAMERA_SEQUENCE_TOOL_NAME}'."
                    )
                }
            target_tool_id = requested_tool_id
        else:
            candidates = [
                tool
                for tool in tool_manager.get_running_tools()
                if tool.tool_name == ALIYUN_CAMERA_SEQUENCE_TOOL_NAME and tool.status == ToolState.RUNNING
            ]
            if not candidates:
                return {"status": "idle", "message": "No Aliyun camera sequence is running."}
            target_tool_id = max(candidates, key=lambda tool: tool.started_at).tool_id

        logger.info("Tool call: cancel_aliyun_camera_sequence tool_id=%s", target_tool_id)
        if await tool_manager.cancel_tool(target_tool_id):
            return {
                "status": "cancelled",
                "message": "Aliyun camera sequence cancelled.",
                "tool_id": target_tool_id,
            }
        return {"error": f"Could not cancel Aliyun camera sequence {target_tool_id}."}
