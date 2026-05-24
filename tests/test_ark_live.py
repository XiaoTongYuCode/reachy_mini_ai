from __future__ import annotations
import sys
import json
import struct
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.ark_live as ark_mod
from reachy_mini_conversation_app.config import (
    ARK_BACKEND,
    config,
    get_default_voice_for_backend,
    refresh_runtime_config_from_env,
)
from reachy_mini_conversation_app.ark_live import (
    ArkLiveHandler,
    ArkRealtimeAuthenticationError,
    _build_audio_payload,
    _parse_realtime_frame,
    _build_full_client_payload,
    _sanitize_tool_result_for_model,
    _openai_tool_specs_to_chat_tools,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _make_handler() -> ArkLiveHandler:
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return ArkLiveHandler(deps)


def _server_frame(
    event: int, session_id: str, payload: bytes, *, serialization: int = 1, message_type: int = 9
) -> bytes:
    session_bytes = session_id.encode("utf-8")
    return b"".join(
        [
            bytes([(1 << 4) | 1, (message_type << 4) | 4, (serialization << 4), 0]),
            struct.pack(">I", event),
            struct.pack(">I", len(session_bytes)),
            session_bytes,
            struct.pack(">I", len(payload)),
            payload,
        ]
    )


def test_ark_backend_uses_volcengine_voice_default() -> None:
    """Ark backend should default to a Volcengine realtime speaker."""
    assert get_default_voice_for_backend(ARK_BACKEND) == "zh_female_vv_jupiter_bigtts"


def test_ark_full_client_frame_round_trips_json_payload() -> None:
    """Server JSON frames should parse event, session, and payload."""
    frame = _server_frame(
        150,
        "session-1",
        json.dumps({"dialog": {"bot_name": "Reachy"}}).encode("utf-8"),
    )

    parsed = _parse_realtime_frame(frame)

    assert parsed["event"] == 150
    assert parsed["session_id"] == "session-1"
    assert parsed["payload"] == {"dialog": {"bot_name": "Reachy"}}


def test_ark_audio_frame_round_trips_raw_payload() -> None:
    """Server audio frames should preserve raw PCM payload bytes."""
    frame = _server_frame(352, "session-1", b"\x00\x01\x02\x03", serialization=0, message_type=11)

    parsed = _parse_realtime_frame(frame)

    assert parsed["event"] == 352
    assert parsed["serialization"] == 0
    assert parsed["session_id"] == "session-1"
    assert parsed["raw_payload"] == b"\x00\x01\x02\x03"


def test_ark_client_frames_use_documented_event_ids() -> None:
    """Client frames should encode official Volcengine event ids."""
    assert _build_full_client_payload(1)[4:8] == struct.pack(">I", 1)
    assert _build_audio_payload("session-1", b"\x00")[4:8] == struct.pack(">I", 200)


def test_ark_tool_specs_convert_to_chat_completions_schema() -> None:
    """Profile tool specs should be converted for OpenAI-compatible Chat Completions."""
    tools = _openai_tool_specs_to_chat_tools(
        [
            {
                "type": "function",
                "name": "move_head",
                "description": "Move the head.",
                "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}},
            },
            {"type": "not_function", "name": "ignored"},
        ]
    )

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "move_head",
                "description": "Move the head.",
                "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}},
            },
        }
    ]


def test_ark_tool_specs_include_ask_openclaw_for_sidecar_router() -> None:
    """Ark sidecar tool conversion should preserve the OpenClaw bridge tool."""
    tools = _openai_tool_specs_to_chat_tools(
        [
            {
                "type": "function",
                "name": "ask_openclaw",
                "description": "Ask OpenClaw.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "ask_openclaw",
                "description": "Ask OpenClaw.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]


def test_ark_tool_result_sanitizer_removes_large_payloads() -> None:
    """Large image payloads should not be sent back through text-only model context."""
    result = _sanitize_tool_result_for_model(
        "camera",
        {
            "description": "桌上有一本书",
            "b64_im": "large-image",
            "nested": {"image_base64": "also-large"},
            "raw": b"bytes",
        },
    )

    assert result == {"description": "桌上有一本书", "nested": {}, "image_attached": True}


def test_ark_pipeline_llm_config_falls_back_to_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ark sidecar config should reuse the local gateway LLM environment variables."""
    original = (
        getattr(config, "PIPELINE_LLM_BASE_URL", None),
        getattr(config, "PIPELINE_LLM_API_KEY", None),
        getattr(config, "PIPELINE_LLM_MODEL", None),
    )
    monkeypatch.delenv("PIPELINE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("PIPELINE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("PIPELINE_LLM_MODEL", raising=False)
    monkeypatch.setenv("GATEWAY_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("GATEWAY_LLM_API_KEY", "gateway-key")
    monkeypatch.setenv("GATEWAY_LLM_MODEL", "gateway-model")

    try:
        refresh_runtime_config_from_env()

        assert config.PIPELINE_LLM_BASE_URL == "http://127.0.0.1:8000/v1"
        assert config.PIPELINE_LLM_API_KEY == "gateway-key"
        assert config.PIPELINE_LLM_MODEL == "gateway-model"
    finally:
        config.PIPELINE_LLM_BASE_URL, config.PIPELINE_LLM_API_KEY, config.PIPELINE_LLM_MODEL = original


def test_ark_headers_use_volcengine_x_api_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Websocket headers should use Volcengine X-Api credential names."""
    monkeypatch.setattr(config, "ARK_REALTIME_APP_ID", "app-id")
    monkeypatch.setattr(config, "ARK_REALTIME_ACCESS_KEY", "access-key")
    monkeypatch.setattr(config, "ARK_REALTIME_APP_KEY", "app-key")
    monkeypatch.setattr(config, "ARK_REALTIME_RESOURCE_ID", "volc.speech.dialog")

    headers = _make_handler()._headers()

    assert headers["X-Api-App-ID"] == "app-id"
    assert headers["X-Api-Access-Key"] == "access-key"
    assert headers["X-Api-App-Key"] == "app-key"
    assert headers["X-Api-Resource-Id"] == "volc.speech.dialog"
    assert headers["X-Api-Connect-Id"]


def test_ark_session_config_uses_profile_instructions_and_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    """StartSession payload should include profile instructions and PCM audio config."""
    monkeypatch.setattr(ark_mod, "get_session_instructions", lambda: "你是 Reachy。")
    monkeypatch.setattr(ark_mod, "get_session_voice", lambda default=None: "zh_male_yunzhou_jupiter_bigtts")
    monkeypatch.setattr(config, "MEMORY_CONTEXT_ENABLED", False)

    session_config = _make_handler()._session_config_payload()

    assert session_config["dialog"]["system_role"] == "你是 Reachy。"
    assert session_config["tts"]["speaker"] == "zh_male_yunzhou_jupiter_bigtts"
    assert session_config["tts"]["audio_config"] == {
        "channel": 1,
        "format": "pcm_s16le",
        "sample_rate": 24000,
    }
    assert session_config["asr"]["audio_info"] == {
        "format": "pcm",
        "sample_rate": 16000,
        "channel": 1,
    }


@pytest.mark.asyncio
async def test_ark_asr_response_emits_final_user_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """Final ASR responses should surface user transcripts."""
    monkeypatch.setattr(config, "MEMORY_CONTEXT_ENABLED", False)
    movement_manager = MagicMock()
    handler = ArkLiveHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager))

    await handler._handle_frame(
        ark_mod.ArkRealtimeFrame(
            event=451,
            message_type=9,
            payload={"results": [{"text": "你好", "is_interim": False}]},
            raw_payload=b"",
        )
    )

    item = await handler.output_queue.get()
    assert item.args[0] == {"role": "user", "content": "你好"}
    movement_manager.set_listening.assert_called_with(False)


@pytest.mark.asyncio
async def test_ark_asr_response_deduplicates_repeated_interim_snapshots() -> None:
    """Repeated ASR interim snapshots should not spam UI outputs."""
    handler = _make_handler()
    frame = ark_mod.ArkRealtimeFrame(
        event=451,
        message_type=9,
        payload={"results": [{"text": "你好", "is_interim": True}]},
        raw_payload=b"",
    )

    await handler._handle_frame(frame)
    await handler._handle_frame(frame)

    assert handler.output_queue.qsize() == 1
    item = await handler.output_queue.get()
    assert item.args[0] == {"role": "user_partial", "content": "你好"}


@pytest.mark.asyncio
async def test_ark_chat_response_flushes_on_chat_ended() -> None:
    """Streaming chat chunks should flush as one assistant transcript."""
    handler = _make_handler()

    await handler._handle_frame(
        ark_mod.ArkRealtimeFrame(event=550, message_type=9, payload={"content": "你好，"}, raw_payload=b"")
    )
    await handler._handle_frame(
        ark_mod.ArkRealtimeFrame(event=550, message_type=9, payload={"content": "我是 Reachy。"}, raw_payload=b"")
    )
    await handler._handle_frame(ark_mod.ArkRealtimeFrame(event=559, message_type=9, payload={}, raw_payload=b""))

    item = await handler.output_queue.get()
    assert item.args[0] == {"role": "assistant", "content": "你好，我是 Reachy。"}


@pytest.mark.asyncio
async def test_ark_suppresses_realtime_chat_and_tts_until_chat_ended() -> None:
    """Sidecar tool calls should prevent duplicate E2E text/audio for the same turn."""
    handler = _make_handler()
    handler._suppress_realtime_response = True

    await handler._handle_frame(
        ark_mod.ArkRealtimeFrame(event=550, message_type=9, payload={"content": "我不能跳舞。"}, raw_payload=b"")
    )
    await handler._handle_frame(
        ark_mod.ArkRealtimeFrame(event=352, message_type=11, serialization=0, payload=b"", raw_payload=b"\x00\x00")
    )
    await handler._handle_frame(ark_mod.ArkRealtimeFrame(event=559, message_type=9, payload={}, raw_payload=b""))

    assert handler.output_queue.empty()
    assert handler._suppress_realtime_response is False
    assert handler._pending_assistant_chunks == []


@pytest.mark.asyncio
async def test_ark_skips_non_raw_tts_response_payload() -> None:
    """JSON TTSResponse payloads should not be decoded as PCM audio."""
    handler = _make_handler()

    await handler._handle_frame(
        ark_mod.ArkRealtimeFrame(
            event=352,
            message_type=9,
            serialization=1,
            payload={"status": "ok"},
            raw_payload=b'{"status":"ok"}',
        )
    )

    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_ark_skips_odd_length_audio_payload() -> None:
    """Malformed audio payloads should not crash the session."""
    handler = _make_handler()

    await handler._handle_audio_payload(b"\x00\x01\x02")

    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_ark_handles_even_length_audio_payload() -> None:
    """Valid PCM payloads should be emitted as int16 audio."""
    handler = _make_handler()

    await handler._handle_audio_payload(b"\x00\x00\x01\x00")

    sample_rate, pcm = await handler.output_queue.get()
    assert sample_rate == handler.output_sample_rate
    assert pcm.tolist() == [[0, 1]]


@pytest.mark.asyncio
async def test_ark_sidecar_uses_gateway_llm_config_and_starts_selected_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ark sidecar should use local gateway LLM settings and active profile tools."""
    created: dict[str, object] = {}

    class FakeCreate:
        async def __call__(self, **kwargs: object) -> object:
            created["kwargs"] = kwargs
            tool_call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="move_head", arguments='{"direction":"left"}'),
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call]))])

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            created["client"] = {"api_key": api_key, "base_url": base_url}
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=FakeCreate()))

    handler = _make_handler()
    started: list[dict[str, object]] = []

    async def fake_start_tool(**kwargs: object) -> object:
        started.append(kwargs)
        return SimpleNamespace(tool_id="move_head-call-1")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    monkeypatch.setattr(
        ark_mod,
        "get_active_tool_specs",
        lambda deps: [
            {
                "type": "function",
                "name": "move_head",
                "description": "Move the head.",
                "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}},
            }
        ],
    )
    monkeypatch.setattr(config, "PIPELINE_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(config, "PIPELINE_LLM_API_KEY", "gateway-key")
    monkeypatch.setattr(config, "PIPELINE_LLM_MODEL", "local-tool-model")
    handler.tool_manager = SimpleNamespace(start_tool=fake_start_tool)

    selected = await handler._maybe_start_tool_calls("向左看")

    assert selected is True
    assert handler._suppress_realtime_response is True
    assert created["client"] == {"api_key": "gateway-key", "base_url": "http://127.0.0.1:8000/v1"}
    kwargs = created["kwargs"]
    assert kwargs["model"] == "local-tool-model"
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"][0]["function"]["name"] == "move_head"
    assert kwargs["messages"][-1] == {"role": "user", "content": "向左看"}
    routine = started[0]["tool_call_routine"]
    assert routine.tool_name == "move_head"
    assert routine.args_json_str == '{"direction":"left"}'


@pytest.mark.asyncio
async def test_ark_sidecar_no_gateway_config_disables_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing sidecar LLM config should leave Ark realtime in normal voice-only mode."""
    handler = _make_handler()
    monkeypatch.setattr(ark_mod, "get_active_tool_specs", lambda deps: [{"type": "function", "name": "move_head"}])
    monkeypatch.setattr(config, "PIPELINE_LLM_BASE_URL", "")
    monkeypatch.setattr(config, "PIPELINE_LLM_MODEL", "")

    selected = await handler._maybe_start_tool_calls("向左看")

    assert selected is False
    assert handler._suppress_realtime_response is False


@pytest.mark.asyncio
async def test_ark_sidecar_no_tool_call_keeps_realtime_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sidecar tool call should not suppress normal Ark chat/audio."""
    class FakeCreate:
        async def __call__(self, **kwargs: object) -> object:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))])

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=FakeCreate()))

    handler = _make_handler()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    monkeypatch.setattr(ark_mod, "get_active_tool_specs", lambda deps: [{"type": "function", "name": "move_head"}])
    monkeypatch.setattr(config, "PIPELINE_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(config, "PIPELINE_LLM_API_KEY", "")
    monkeypatch.setattr(config, "PIPELINE_LLM_MODEL", "local-tool-model")

    selected = await handler._maybe_start_tool_calls("你好")

    assert selected is False
    assert handler._suppress_realtime_response is False


@pytest.mark.asyncio
async def test_ark_sidecar_sends_recent_history_to_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short follow-up utterances should be routed with recent conversation context."""
    created: dict[str, object] = {}

    class FakeCreate:
        async def __call__(self, **kwargs: object) -> object:
            created["kwargs"] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))])

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=FakeCreate()))

    handler = _make_handler()
    handler._append_sidecar_history("user", "给我发一封邮件。")
    handler._append_sidecar_history("assistant", "发邮件呀，你先跟我说下内容吧。")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    monkeypatch.setattr(ark_mod, "get_active_tool_specs", lambda deps: [{"type": "function", "name": "send_email"}])
    monkeypatch.setattr(config, "PIPELINE_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(config, "PIPELINE_LLM_API_KEY", "")
    monkeypatch.setattr(config, "PIPELINE_LLM_MODEL", "local-tool-model")

    selected = await handler._maybe_start_tool_calls("随便测试邮件。")

    assert selected is False
    assert created["kwargs"]["messages"][-3:] == [
        {"role": "user", "content": "给我发一封邮件。"},
        {"role": "assistant", "content": "发邮件呀，你先跟我说下内容吧。"},
        {"role": "user", "content": "随便测试邮件。"},
    ]


@pytest.mark.asyncio
async def test_ark_pending_email_followup_starts_send_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """Email intent followed by content should start send_email when the router first asks for more info."""
    class FakeCreate:
        async def __call__(self, **kwargs: object) -> object:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))])

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=FakeCreate()))

    handler = _make_handler()
    started: list[dict[str, object]] = []

    async def fake_start_tool(**kwargs: object) -> object:
        started.append(kwargs)
        return SimpleNamespace(tool_id="send_email-call-1")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    monkeypatch.setattr(ark_mod, "get_active_tool_specs", lambda deps: [{"type": "function", "name": "send_email"}])
    monkeypatch.setattr(config, "PIPELINE_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(config, "PIPELINE_LLM_API_KEY", "")
    monkeypatch.setattr(config, "PIPELINE_LLM_MODEL", "local-tool-model")
    handler.tool_manager = SimpleNamespace(start_tool=fake_start_tool)

    first_selected = await handler._maybe_start_tool_calls("给我发一封邮件。")
    second_selected = await handler._maybe_start_tool_calls("随便测试邮件。")

    assert first_selected is False
    assert second_selected is True
    assert handler._pending_tool_request is None
    assert handler._suppress_realtime_response is True
    routine = started[0]["tool_call_routine"]
    assert routine.tool_name == "send_email"
    assert json.loads(routine.args_json_str) == {"subject": "测试邮件", "body": "随便测试邮件。"}


@pytest.mark.asyncio
async def test_ark_tool_result_is_sent_as_chat_rag_text() -> None:
    """Completed local tools should be returned to Volcengine using ChatRAGText."""
    sent: list[bytes] = []

    class FakeConnection:
        async def send(self, payload: bytes) -> None:
            sent.append(payload)

    handler = _make_handler()
    handler._connection = FakeConnection()
    handler._session_id = "session-1"

    await handler._send_tool_result_to_realtime("move_head", {"status": "ok"})

    assert sent
    payload = sent[0]
    assert payload[4:8] == struct.pack(">I", 502)
    offset = 8
    session_length = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4 + session_length
    payload_length = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    body = json.loads(payload[offset : offset + payload_length].decode("utf-8"))
    assert "external_rag" in body
    assert "tool_name=move_head" in body["external_rag"]


@pytest.mark.asyncio
async def test_ark_tool_result_waits_for_suppressed_turn_to_end() -> None:
    """Fast tool results should not race the suppressed Ark realtime response."""
    sent: list[bytes] = []

    class FakeConnection:
        async def send(self, payload: bytes) -> None:
            sent.append(payload)

    handler = _make_handler()
    handler._connection = FakeConnection()
    handler._session_id = "session-1"
    handler._suppress_realtime_response = True

    await handler._handle_tool_result(
        SimpleNamespace(error=None, result={"status": "ok"}, id="call-1", tool_name="send_email")
    )

    assert sent == []
    assert handler._queued_tool_rag_results == [("send_email", {"status": "ok"})]

    await handler._handle_frame(ark_mod.ArkRealtimeFrame(event=559, message_type=9, payload={}, raw_payload=b""))

    assert len(sent) == 1
    assert handler._queued_tool_rag_results == []


@pytest.mark.asyncio
async def test_ark_queued_tool_result_drains_without_chat_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued tool results should still be spoken if Ark never emits ChatEnded for the suppressed turn."""
    sent: list[bytes] = []

    class FakeConnection:
        async def send(self, payload: bytes) -> None:
            sent.append(payload)

    handler = _make_handler()
    handler._connection = FakeConnection()
    handler._session_id = "session-1"
    handler._suppress_realtime_response = True
    monkeypatch.setattr(ark_mod, "_ARK_SUPPRESSED_TURN_DRAIN_DELAY_SECONDS", 0.01)

    await handler._handle_tool_result(
        SimpleNamespace(error=None, result={"status": "ok"}, id="call-1", tool_name="dance")
    )
    await asyncio.sleep(0.05)

    assert len(sent) == 1
    assert handler._suppress_realtime_response is False
    assert handler._queued_tool_rag_results == []


def test_ark_does_not_import_openai_sdk() -> None:
    """Native Volcengine transport should not depend on the OpenAI SDK."""
    assert not hasattr(ark_mod, "AsyncOpenAI")


@pytest.mark.asyncio
async def test_ark_authentication_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 401 handshake failures should fail fast with a clear auth error."""
    handler = _make_handler()
    calls = 0

    async def fail_auth() -> None:
        nonlocal calls
        calls += 1
        raise ArkRealtimeAuthenticationError("HTTP 401")

    monkeypatch.setattr(handler, "_run_session", fail_auth)

    with pytest.raises(ArkRealtimeAuthenticationError):
        await handler.start_up()

    assert calls == 1
