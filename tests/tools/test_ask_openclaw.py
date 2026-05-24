from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.ask_openclaw as tool_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.openclaw_bridge import OpenClawResponse
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.ask_openclaw import AskOpenClaw


def _deps(*, camera_worker: object | None = None) -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )


@pytest.mark.asyncio
async def test_ask_openclaw_returns_clear_error_when_gateway_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a model-readable error when OpenClaw is disabled."""
    monkeypatch.setattr(config, "OPENCLAW_GATEWAY_URL", "")
    monkeypatch.setattr(config, "OPENCLAW_TOKEN", "test-token")

    result = await AskOpenClaw()(_deps(), query="what do you remember about me?")

    assert result["source"] == "openclaw"
    assert "not configured" in result["error"]


@pytest.mark.asyncio
async def test_ask_openclaw_returns_clear_error_when_token_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a model-readable error when OpenClaw auth is disabled."""
    monkeypatch.setattr(config, "OPENCLAW_GATEWAY_URL", "ws://openclaw.test")
    monkeypatch.setattr(config, "OPENCLAW_TOKEN", "")

    result = await AskOpenClaw()(_deps(), query="what do you remember about me?")

    assert result["source"] == "openclaw"
    assert "OPENCLAW_TOKEN" in result["error"]


@pytest.mark.asyncio
async def test_ask_openclaw_success_returns_response_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the OpenClaw response payload on a successful bridge call."""
    calls: dict[str, object] = {}

    class FakeBridge:
        async def connect(self) -> bool:
            calls["connected"] = True
            return True

        async def chat(self, message: str, *, system_context: str | None = None) -> OpenClawResponse:
            calls["message"] = message
            calls["system_context"] = system_context
            return OpenClawResponse(content="OpenClaw says yes.")

        async def disconnect(self) -> None:
            calls["disconnected"] = True

    monkeypatch.setattr(config, "OPENCLAW_GATEWAY_URL", "ws://openclaw.test")
    monkeypatch.setattr(config, "OPENCLAW_TOKEN", "test-token")
    monkeypatch.setattr(tool_mod, "OpenClawBridge", FakeBridge)

    result = await AskOpenClaw()(_deps(), query="check my OpenClaw memory")

    assert result == {
        "response": "OpenClaw says yes.",
        "source": "openclaw",
        "image_requested": False,
        "image_available": False,
        "image_attached": False,
    }
    assert calls["message"] == "check my OpenClaw memory"
    assert calls["connected"] is True
    assert calls["disconnected"] is True


@pytest.mark.asyncio
async def test_ask_openclaw_connection_failure_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert bridge connection failures into structured tool errors."""
    class FakeBridge:
        async def connect(self) -> bool:
            return False

        async def disconnect(self) -> None:
            pass

    monkeypatch.setattr(config, "OPENCLAW_GATEWAY_URL", "ws://openclaw.test")
    monkeypatch.setattr(config, "OPENCLAW_TOKEN", "test-token")
    monkeypatch.setattr(tool_mod, "OpenClawBridge", FakeBridge)

    result = await AskOpenClaw()(_deps(), query="run an external task")

    assert result["source"] == "openclaw"
    assert result["image_attached"] is False
    assert "unreachable" in result["error"]


@pytest.mark.asyncio
async def test_ask_openclaw_timeout_does_not_escape_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep bridge timeouts from escaping into the realtime loop."""
    class FakeBridge:
        async def connect(self) -> bool:
            return True

        async def chat(self, message: str, *, system_context: str | None = None) -> OpenClawResponse:
            raise TimeoutError("too slow")

        async def disconnect(self) -> None:
            pass

    monkeypatch.setattr(config, "OPENCLAW_GATEWAY_URL", "ws://openclaw.test")
    monkeypatch.setattr(config, "OPENCLAW_TOKEN", "test-token")
    monkeypatch.setattr(tool_mod, "OpenClawBridge", FakeBridge)

    result = await AskOpenClaw()(_deps(), query="do a long OpenClaw task")

    assert result["source"] == "openclaw"
    assert result["image_attached"] is False
    assert "TimeoutError" in result["error"]


@pytest.mark.asyncio
async def test_ask_openclaw_camera_request_without_camera_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explain camera limitations when no frame is available."""
    captured: dict[str, str] = {}

    class FakeBridge:
        async def connect(self) -> bool:
            return True

        async def chat(self, message: str, *, system_context: str | None = None) -> OpenClawResponse:
            captured["message"] = message
            return OpenClawResponse(content="I need the local camera tool for vision.")

        async def disconnect(self) -> None:
            pass

    monkeypatch.setattr(config, "OPENCLAW_GATEWAY_URL", "ws://openclaw.test")
    monkeypatch.setattr(config, "OPENCLAW_TOKEN", "test-token")
    monkeypatch.setattr(tool_mod, "OpenClawBridge", FakeBridge)

    result = await AskOpenClaw()(
        _deps(camera_worker=None),
        query="what do you see?",
        include_camera_image=True,
    )

    assert result["image_requested"] is True
    assert result["image_available"] is False
    assert result["image_attached"] is False
    assert "no current camera frame is available" in captured["message"]


@pytest.mark.asyncio
async def test_ask_openclaw_camera_request_with_frame_marks_unattached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mark camera frames as available but unattached for the text-only bridge."""
    captured: dict[str, str] = {}
    camera_worker = SimpleNamespace(get_latest_frame=lambda: object())

    class FakeBridge:
        async def connect(self) -> bool:
            return True

        async def chat(self, message: str, *, system_context: str | None = None) -> OpenClawResponse:
            captured["message"] = message
            return OpenClawResponse(content="Use the local camera tool.")

        async def disconnect(self) -> None:
            pass

    monkeypatch.setattr(config, "OPENCLAW_GATEWAY_URL", "ws://openclaw.test")
    monkeypatch.setattr(config, "OPENCLAW_TOKEN", "test-token")
    monkeypatch.setattr(tool_mod, "OpenClawBridge", FakeBridge)

    result = await AskOpenClaw()(
        _deps(camera_worker=camera_worker),
        query="what is in front of me?",
        include_camera_image=True,
    )

    assert result["image_available"] is True
    assert result["image_attached"] is False
    assert "text-only" in captured["message"]
