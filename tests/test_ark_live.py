from __future__ import annotations
import json
import struct
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.ark_live as ark_mod
from reachy_mini_conversation_app.config import ARK_BACKEND, config, get_default_voice_for_backend
from reachy_mini_conversation_app.ark_live import (
    ArkLiveHandler,
    ArkRealtimeAuthenticationError,
    _build_audio_payload,
    _parse_realtime_frame,
    _build_full_client_payload,
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
