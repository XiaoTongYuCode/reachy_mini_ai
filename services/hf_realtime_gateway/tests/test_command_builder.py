from reachy_mini_hf_realtime_gateway.config import GatewayConfig
from reachy_mini_hf_realtime_gateway.command_builder import build_command, format_command, redact_command


def _config(**overrides: object) -> GatewayConfig:
    values = {
        "llm_base_url": "https://llm.example.test/v1",
        "llm_api_key": "secret-key",
        "llm_model": "test-model",
    }
    values.update(overrides)
    return GatewayConfig(**values)


def test_build_command_uses_chinese_gateway_defaults() -> None:
    command = build_command(_config())

    assert command == [
        "speech-to-speech",
        "--mode",
        "realtime",
        "--ws_host",
        "127.0.0.1",
        "--ws_port",
        "8765",
        "--stt",
        "faster-whisper",
        "--language",
        "zh",
        "--llm_backend",
        "responses-api",
        "--responses_api_base_url",
        "https://llm.example.test/v1",
        "--faster_whisper_stt_model_name",
        "large-v3",
        "--faster_whisper_stt_gen_language",
        "zh",
        "--faster_whisper_stt_device",
        "auto",
        "--faster_whisper_stt_compute_type",
        "auto",
        "--responses_api_api_key",
        "secret-key",
        "--model_name",
        "test-model",
        "--responses_api_stream",
        "--tts",
        "qwen3",
        "--qwen3_tts_model_name",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "--qwen3_tts_speaker",
        "Vivian",
        "--qwen3_tts_language",
        "Chinese",
        "--qwen3_tts_mlx_quantization",
        "6bit",
        "--enable_live_transcription",
    ]


def test_build_command_omits_optional_api_key_and_boolean_flags() -> None:
    command = build_command(
        _config(
            llm_api_key="",
            responses_api_stream=False,
            enable_live_transcription=False,
        )
    )

    assert "--responses_api_api_key" not in command
    assert "--responses_api_stream" not in command
    assert "--enable_live_transcription" not in command


def test_build_command_uses_shared_stt_model_name_for_non_faster_whisper() -> None:
    command = build_command(_config(stt="whisper-mlx"))

    assert "--stt" in command
    assert "whisper-mlx" in command
    assert "--stt_model_name" in command
    assert "--faster_whisper_stt_model_name" not in command


def test_build_command_omits_qwen3_options_for_other_tts_backend() -> None:
    command = build_command(_config(tts="kokoro"))

    assert "--tts" in command
    assert "kokoro" in command
    assert "--qwen3_tts_model_name" not in command
    assert "--qwen3_tts_speaker" not in command


def test_redact_command_hides_api_key_value() -> None:
    command = build_command(_config())

    redacted = redact_command(command)

    assert "secret-key" not in redacted
    assert "<redacted>" in redacted


def test_format_command_hides_api_key_value() -> None:
    command = build_command(_config())

    rendered = format_command(command)

    assert "secret-key" not in rendered
    assert "<redacted>" in rendered


def test_format_command_hides_secret_like_model_name() -> None:
    command = build_command(_config(llm_model="nvapi-secret-like-model-misconfiguration"))

    rendered = format_command(command)

    assert "nvapi-secret-like-model-misconfiguration" not in rendered
    assert "--model_name '<redacted>'" in rendered
