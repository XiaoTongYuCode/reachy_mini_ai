import pytest

from reachy_mini_hf_realtime_gateway.config import (
    ConfigError,
    GatewayConfig,
    config_from_env,
)


def test_config_defaults_without_validation() -> None:
    config = config_from_env({}, validate=False)

    assert config.host == "127.0.0.1"
    assert config.port == 8765
    assert config.stt == "faster-whisper"
    assert config.stt_model == "small"
    assert config.faster_whisper_stt_device == "auto"
    assert config.faster_whisper_stt_compute_type == "auto"
    assert config.language == "zh"
    assert config.tts == "qwen3"
    assert config.tts_model == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert config.tts_speaker == "Vivian"
    assert config.enable_live_transcription is False
    assert config.live_transcription_update_interval == 0.12
    assert config.vad_min_silence_ms == 180
    assert config.vad_min_speech_ms == 180
    assert config.vad_speech_pad_ms == 120
    assert config.llm_stream_batch_sentences == 1
    assert config.responses_api_disable_thinking is True
    assert config.faster_whisper_stt_gen_max_new_tokens == 64
    assert config.faster_whisper_stt_gen_beam_size == 1
    assert config.realtime_url == "ws://127.0.0.1:8765/v1/realtime"


def test_env_overrides_config_values() -> None:
    config = config_from_env(
        {
            "GATEWAY_HOST": "0.0.0.0",
            "GATEWAY_PORT": "9001",
            "GATEWAY_STT": "faster-whisper",
            "GATEWAY_STT_MODEL": "small",
            "GATEWAY_FASTER_WHISPER_STT_DEVICE": "cpu",
            "GATEWAY_FASTER_WHISPER_STT_COMPUTE_TYPE": "int8",
            "GATEWAY_LANGUAGE": "auto",
            "GATEWAY_LLM_BASE_URL": "https://llm.example.test/v1",
            "GATEWAY_LLM_API_KEY": "secret-key",
            "GATEWAY_LLM_MODEL": "test-model",
            "GATEWAY_RESPONSES_API_STREAM": "false",
            "GATEWAY_RESPONSES_API_DISABLE_THINKING": "false",
            "GATEWAY_LLM_STREAM_BATCH_SENTENCES": "2",
            "GATEWAY_TTS_SPEAKER": "Serena",
            "GATEWAY_ENABLE_LIVE_TRANSCRIPTION": "1",
            "GATEWAY_LIVE_TRANSCRIPTION_UPDATE_INTERVAL": "0.08",
            "GATEWAY_VAD_MIN_SILENCE_MS": "140",
            "GATEWAY_VAD_MIN_SPEECH_MS": "120",
            "GATEWAY_VAD_SPEECH_PAD_MS": "60",
            "GATEWAY_FASTER_WHISPER_STT_GEN_MAX_NEW_TOKENS": "32",
            "GATEWAY_FASTER_WHISPER_STT_GEN_BEAM_SIZE": "1",
        }
    )

    assert config.host == "0.0.0.0"
    assert config.port == 9001
    assert config.stt == "faster-whisper"
    assert config.stt_model == "small"
    assert config.faster_whisper_stt_device == "cpu"
    assert config.faster_whisper_stt_compute_type == "int8"
    assert config.language == "auto"
    assert config.llm_base_url == "https://llm.example.test/v1"
    assert config.llm_api_key == "secret-key"
    assert config.llm_model == "test-model"
    assert config.responses_api_stream is False
    assert config.responses_api_disable_thinking is False
    assert config.llm_stream_batch_sentences == 2
    assert config.tts_speaker == "Serena"
    assert config.enable_live_transcription is True
    assert config.live_transcription_update_interval == 0.08
    assert config.vad_min_silence_ms == 140
    assert config.vad_min_speech_ms == 120
    assert config.vad_speech_pad_ms == 60
    assert config.faster_whisper_stt_gen_max_new_tokens == 32
    assert config.faster_whisper_stt_gen_beam_size == 1


def test_missing_llm_base_url_is_invalid() -> None:
    with pytest.raises(ConfigError, match="GATEWAY_LLM_BASE_URL is required"):
        GatewayConfig(llm_model="test-model").validate()


def test_missing_llm_model_is_invalid() -> None:
    with pytest.raises(ConfigError, match="GATEWAY_LLM_MODEL is required"):
        GatewayConfig(llm_base_url="https://llm.example.test/v1").validate()


def test_invalid_port_reports_clear_error() -> None:
    with pytest.raises(ConfigError, match="GATEWAY_PORT must be an integer"):
        config_from_env({"GATEWAY_PORT": "not-a-port"}, validate=False)


def test_invalid_boolean_reports_clear_error() -> None:
    with pytest.raises(ConfigError, match="GATEWAY_RESPONSES_API_STREAM must be a boolean"):
        config_from_env({"GATEWAY_RESPONSES_API_STREAM": "sometimes"}, validate=False)


def test_invalid_float_reports_clear_error() -> None:
    with pytest.raises(ConfigError, match="GATEWAY_LIVE_TRANSCRIPTION_UPDATE_INTERVAL must be a number"):
        config_from_env({"GATEWAY_LIVE_TRANSCRIPTION_UPDATE_INTERVAL": "fast"}, validate=False)
