"""Tests for the Aliyun camera sequence cancel tool."""

import asyncio
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager
from reachy_mini_conversation_app.tools.cancel_aliyun_camera_sequence import CancelAliyunCameraSequence


@pytest.mark.asyncio
async def test_cancel_aliyun_camera_sequence_cancels_latest_running_sequence() -> None:
    """The dedicated cancel tool should stop the latest running Aliyun camera sequence."""
    manager = BackgroundToolManager()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    async def never_finishes(_tool_id: str) -> dict[str, object]:
        await asyncio.sleep(60)
        return {"status": "finished"}

    older = await manager.start_coroutine_tool(
        call_id="older",
        tool_name="aliyun_camera_sequence",
        coroutine_factory=never_finishes,
        is_idle_tool_call=False,
    )
    newer = await manager.start_coroutine_tool(
        call_id="newer",
        tool_name="aliyun_camera_sequence",
        coroutine_factory=never_finishes,
        is_idle_tool_call=False,
    )

    result = await CancelAliyunCameraSequence()(deps, tool_manager=manager)

    assert result["status"] == "cancelled"
    assert result["tool_id"] == newer.tool_id
    assert older._task is not None and not older._task.cancelled()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_aliyun_camera_sequence_rejects_other_tool_names() -> None:
    """The dedicated cancel tool should not cancel unrelated background tools."""
    manager = BackgroundToolManager()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    async def never_finishes(_tool_id: str) -> dict[str, object]:
        await asyncio.sleep(60)
        return {"status": "finished"}

    other = await manager.start_coroutine_tool(
        call_id="other",
        tool_name="dance",
        coroutine_factory=never_finishes,
        is_idle_tool_call=False,
    )

    result = await CancelAliyunCameraSequence()(deps, tool_manager=manager, tool_id=other.tool_id)

    assert "not 'aliyun_camera_sequence'" in result["error"]
    assert other._task is not None and not other._task.cancelled()
    await manager.shutdown()
