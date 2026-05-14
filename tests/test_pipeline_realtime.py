"""Tests for the local pipeline realtime backend."""

from __future__ import annotations
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.pipeline_realtime as pipeline_mod
from reachy_mini_conversation_app.config import (
    HF_DEFAULTS,
    PIPELINE_BACKEND,
    config,
    get_default_voice_for_backend,
    refresh_runtime_config_from_env,
    get_available_voices_for_backend,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.pipeline_realtime import PipelineRealtimeHandler, get_pipeline_realtime_ws_url


def _make_handler() -> PipelineRealtimeHandler:
    """Build a pipeline handler with fake dependencies."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        head_wobbler=MagicMock(),
    )
    return PipelineRealtimeHandler(deps)


def test_gateway_llm_env_aliases_feed_pipeline_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline LLM config should accept the gateway env names used by the local service."""
    original_model = config.PIPELINE_LLM_MODEL
    original_key = config.PIPELINE_LLM_API_KEY
    original_base_url = config.PIPELINE_LLM_BASE_URL
    try:
        monkeypatch.delenv("PIPELINE_LLM_MODEL", raising=False)
        monkeypatch.delenv("PIPELINE_LLM_API_KEY", raising=False)
        monkeypatch.delenv("PIPELINE_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("GATEWAY_LLM_MODEL", "z-ai/glm-4.5-air:free")
        monkeypatch.setenv("GATEWAY_LLM_API_KEY", "test-openrouter-key")
        monkeypatch.setenv("GATEWAY_LLM_BASE_URL", "https://openrouter.ai/api/v1")

        refresh_runtime_config_from_env()

        assert config.PIPELINE_LLM_MODEL == "z-ai/glm-4.5-air:free"
        assert config.PIPELINE_LLM_API_KEY == "test-openrouter-key"
        assert config.PIPELINE_LLM_BASE_URL == "https://openrouter.ai/api/v1"
    finally:
        monkeypatch.delenv("GATEWAY_LLM_MODEL", raising=False)
        monkeypatch.delenv("GATEWAY_LLM_API_KEY", raising=False)
        monkeypatch.delenv("GATEWAY_LLM_BASE_URL", raising=False)
        config.PIPELINE_LLM_MODEL = original_model
        config.PIPELINE_LLM_API_KEY = original_key
        config.PIPELINE_LLM_BASE_URL = original_base_url


def test_pipeline_realtime_url_defaults_to_local_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline backend should default to the local speech-to-speech gateway."""
    monkeypatch.delenv("PIPELINE_REALTIME_WS_URL", raising=False)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)

    assert get_pipeline_realtime_ws_url() == "ws://127.0.0.1:8765/v1/realtime"


def test_pipeline_uses_local_qwen_voice_defaults() -> None:
    """Pipeline voice defaults must be compatible with the local Qwen3 TTS backend."""
    assert get_default_voice_for_backend(PIPELINE_BACKEND) == HF_DEFAULTS.voice
    assert "cedar" not in get_available_voices_for_backend(PIPELINE_BACKEND)
    assert HF_DEFAULTS.voice in get_available_voices_for_backend(PIPELINE_BACKEND)


@pytest.mark.asyncio
async def test_pipeline_uses_existing_gateway_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachable local gateway should be used without starting another process."""
    monkeypatch.setenv("PIPELINE_REALTIME_WS_URL", "ws://127.0.0.1:9876/v1/realtime")
    probe = AsyncMock()
    monkeypatch.setattr(pipeline_mod, "_probe_realtime_websocket", probe)
    start_gateway = AsyncMock()
    handler = _make_handler()
    monkeypatch.setattr(handler, "_ensure_local_gateway_started", start_gateway)
    fake_client = MagicMock()
    monkeypatch.setattr(
        pipeline_mod,
        "_build_openai_compatible_client_from_realtime_url",
        lambda *_args: (fake_client, {"model": "ignored"}),
    )

    client = await handler._build_realtime_client()

    assert client is fake_client
    probe.assert_awaited_once_with("ws://127.0.0.1:9876/v1/realtime")
    start_gateway.assert_not_called()
    assert handler._realtime_connect_query == {"model": "ignored"}


@pytest.mark.asyncio
async def test_pipeline_starts_managed_gateway_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreachable local gateway should trigger the managed speech-to-speech launcher."""
    monkeypatch.setenv("PIPELINE_REALTIME_WS_URL", "ws://127.0.0.1:9876/v1/realtime")

    async def fail_probe(_url: str) -> None:
        raise OSError("not ready")

    monkeypatch.setattr(pipeline_mod, "_probe_realtime_websocket", fail_probe)
    start_gateway = AsyncMock()
    handler = _make_handler()
    monkeypatch.setattr(handler, "_ensure_local_gateway_started", start_gateway)
    fake_client = MagicMock()
    monkeypatch.setattr(
        pipeline_mod,
        "_build_openai_compatible_client_from_realtime_url",
        lambda *_args: (fake_client, {}),
    )

    await handler._build_realtime_client()

    start_gateway.assert_awaited_once_with("ws://127.0.0.1:9876/v1/realtime")


def test_pipeline_managed_gateway_env_uses_requested_host_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Managed gateway launch should pass the local host and port to GATEWAY_* env."""
    captured: dict[str, Any] = {}

    class FakeGatewayProcess:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs

        async def ensure_started(self, _url: str) -> None:
            return

    monkeypatch.setattr(pipeline_mod, "HFRealtimeGatewayProcess", FakeGatewayProcess)
    handler = _make_handler()

    import asyncio

    asyncio.run(handler._ensure_local_gateway_started("ws://0.0.0.0:9888/v1/realtime"))

    env = captured["kwargs"]["env"]
    assert env["GATEWAY_HOST"] == "0.0.0.0"
    assert env["GATEWAY_PORT"] == "9888"
