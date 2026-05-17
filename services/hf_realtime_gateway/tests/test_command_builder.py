import sys

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
        sys.executable,
        "-m",
        "reachy_mini_hf_realtime_gateway.speech_to_speech_runner",
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
        "--min_silence_ms",
        "180",
        "--min_speech_ms",
        "180",
        "--speech_pad_ms",
        "120",
        "--llm_backend",
        "responses-api",
        "--responses_api_base_url",
        "https://llm.example.test/v1",
        "--faster_whisper_stt_model_name",
        "small",
        "--faster_whisper_stt_gen_language",
        "zh",
        "--faster_whisper_stt_device",
        "auto",
        "--faster_whisper_stt_compute_type",
        "auto",
        "--faster_whisper_stt_gen_max_new_tokens",
        "64",
        "--faster_whisper_stt_gen_beam_size",
        "1",
        "--responses_api_api_key",
        "secret-key",
        "--model_name",
        "test-model",
        "--responses_api_stream",
        "--stream_batch_sentences",
        "1",
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
        "--no_enable_live_transcription",
    ]


def test_build_command_includes_live_transcription_when_enabled() -> None:
    command = build_command(_config(enable_live_transcription=True))

    assert "--enable_live_transcription" in command
    assert "--no_enable_live_transcription" not in command
    assert command[command.index("--live_transcription_update_interval") + 1] == "0.12"


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
    assert "--no_enable_live_transcription" in command


def test_build_command_can_disable_responses_api_thinking_flag() -> None:
    command = build_command(_config(responses_api_disable_thinking=False))

    assert "--no_responses_api_disable_thinking" in command


def test_build_command_uses_shared_stt_model_name_for_non_faster_whisper() -> None:
    command = build_command(_config(stt="whisper-mlx"))

    assert "--stt" in command
    assert "whisper-mlx" in command
    assert "--stt_model_name" in command
    assert "--faster_whisper_stt_model_name" not in command


def test_build_command_respects_custom_speech_to_speech_binary() -> None:
    command = build_command(_config(speech_to_speech_bin="/opt/bin/speech-to-speech"))

    assert command[0] == "/opt/bin/speech-to-speech"
    assert "-m" not in command[:3]


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
