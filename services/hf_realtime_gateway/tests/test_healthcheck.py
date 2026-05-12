import sys
import json
from typing import Any

import pytest

from reachy_mini_hf_realtime_gateway.config import GatewayConfig
from reachy_mini_hf_realtime_gateway.healthcheck import check_realtime_endpoint


class _FakeWebSocket:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = list(messages)
        self.sent: list[str] = []

    async def __aenter__(self) -> "_FakeWebSocket":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> Any:
        if not self.messages:
            raise TimeoutError("no more messages")
        return self.messages.pop(0)


class _FakeWebsocketsModule:
    def __init__(self, messages: list[Any]) -> None:
        self.websocket = _FakeWebSocket(messages)
        self.connected_url: str | None = None

    def connect(self, url: str) -> _FakeWebSocket:
        self.connected_url = url
        return self.websocket


def _valid_config() -> GatewayConfig:
    return GatewayConfig(llm_base_url="https://llm.example.test/v1", llm_model="test-model")


@pytest.mark.asyncio
async def test_healthcheck_waits_for_session_updated(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_websockets = _FakeWebsocketsModule([json.dumps({"type": "session.updated"})])
    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)

    result = await check_realtime_endpoint(_valid_config())

    assert result.ok is True
    assert result.message == "session.updated received"
    assert fake_websockets.connected_url == "ws://127.0.0.1:8765/v1/realtime"
    sent_payload = json.loads(fake_websockets.websocket.sent[0])
    assert sent_payload["type"] == "session.update"


@pytest.mark.asyncio
async def test_healthcheck_reports_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_websockets = _FakeWebsocketsModule([json.dumps({"type": "error", "error": {"message": "invalid session"}})])
    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)

    result = await check_realtime_endpoint(_valid_config())

    assert result.ok is False
    assert "invalid session" in result.message
