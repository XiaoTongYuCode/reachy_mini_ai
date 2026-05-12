from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

from dotenv import find_dotenv, load_dotenv


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_STT = "faster-whisper"
DEFAULT_STT_MODEL = "large-v3"
DEFAULT_FASTER_WHISPER_STT_DEVICE = "auto"
DEFAULT_FASTER_WHISPER_STT_COMPUTE_TYPE = "auto"
DEFAULT_LANGUAGE = "zh"
DEFAULT_LLM_BACKEND = "responses-api"
DEFAULT_TTS = "qwen3"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_TTS_SPEAKER = "Vivian"
DEFAULT_TTS_LANGUAGE = "Chinese"
DEFAULT_QWEN3_TTS_MLX_QUANTIZATION = "6bit"
DEFAULT_SPEECH_TO_SPEECH_BIN = "speech-to-speech"


class ConfigError(ValueError):
    """Raised when gateway configuration is invalid."""


@dataclass(frozen=True)
class GatewayConfig:
    """Runtime configuration for the speech-to-speech gateway wrapper."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    stt: str = DEFAULT_STT
    stt_model: str = DEFAULT_STT_MODEL
    faster_whisper_stt_device: str = DEFAULT_FASTER_WHISPER_STT_DEVICE
    faster_whisper_stt_compute_type: str = DEFAULT_FASTER_WHISPER_STT_COMPUTE_TYPE
    language: str = DEFAULT_LANGUAGE
    llm_backend: str = DEFAULT_LLM_BACKEND
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    responses_api_stream: bool = True
    tts: str = DEFAULT_TTS
    tts_model: str = DEFAULT_TTS_MODEL
    tts_speaker: str = DEFAULT_TTS_SPEAKER
    tts_language: str = DEFAULT_TTS_LANGUAGE
    qwen3_tts_mlx_quantization: str = DEFAULT_QWEN3_TTS_MLX_QUANTIZATION
    enable_live_transcription: bool = True
    speech_to_speech_bin: str = DEFAULT_SPEECH_TO_SPEECH_BIN

    @property
    def realtime_url(self) -> str:
        return f"ws://{self.host}:{self.port}/v1/realtime"

    def validate(self) -> None:
        if not self.host:
            raise ConfigError("GATEWAY_HOST must not be empty.")
        if self.port <= 0 or self.port > 65535:
            raise ConfigError("GATEWAY_PORT must be between 1 and 65535.")
        if not self.llm_base_url:
            raise ConfigError("GATEWAY_LLM_BASE_URL is required.")
        if not self.llm_model:
            raise ConfigError("GATEWAY_LLM_MODEL is required.")
        if not self.speech_to_speech_bin:
            raise ConfigError("GATEWAY_SPEECH_TO_SPEECH_BIN must not be empty.")


def load_dotenv_for_gateway(env_file: str | Path | None = None) -> None:
    """Load a gateway env file, defaulting to the nearest `.env` from cwd."""
    if env_file is not None:
        load_dotenv(dotenv_path=Path(env_file), override=True)
        return

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)


def _get(env: Mapping[str, str], name: str, default: str = "") -> str:
    return (env.get(name, default) or "").strip()


def _get_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _get(env, name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean value, got {raw!r}.")


def config_from_env(
    env: Mapping[str, str] | None = None,
    *,
    validate: bool = True,
) -> GatewayConfig:
    """Build gateway configuration from environment variables."""
    source = os.environ if env is None else env
    config = GatewayConfig(
        host=_get(source, "GATEWAY_HOST", DEFAULT_HOST),
        port=_get_int(source, "GATEWAY_PORT", DEFAULT_PORT),
        stt=_get(source, "GATEWAY_STT", DEFAULT_STT),
        stt_model=_get(source, "GATEWAY_STT_MODEL", DEFAULT_STT_MODEL),
        faster_whisper_stt_device=_get(
            source,
            "GATEWAY_FASTER_WHISPER_STT_DEVICE",
            DEFAULT_FASTER_WHISPER_STT_DEVICE,
        ),
        faster_whisper_stt_compute_type=_get(
            source,
            "GATEWAY_FASTER_WHISPER_STT_COMPUTE_TYPE",
            DEFAULT_FASTER_WHISPER_STT_COMPUTE_TYPE,
        ),
        language=_get(source, "GATEWAY_LANGUAGE", DEFAULT_LANGUAGE),
        llm_backend=_get(source, "GATEWAY_LLM_BACKEND", DEFAULT_LLM_BACKEND),
        llm_base_url=_get(source, "GATEWAY_LLM_BASE_URL"),
        llm_api_key=_get(source, "GATEWAY_LLM_API_KEY"),
        llm_model=_get(source, "GATEWAY_LLM_MODEL"),
        responses_api_stream=_get_bool(source, "GATEWAY_RESPONSES_API_STREAM", True),
        tts=_get(source, "GATEWAY_TTS", DEFAULT_TTS),
        tts_model=_get(source, "GATEWAY_TTS_MODEL", DEFAULT_TTS_MODEL),
        tts_speaker=_get(source, "GATEWAY_TTS_SPEAKER", DEFAULT_TTS_SPEAKER),
        tts_language=_get(source, "GATEWAY_TTS_LANGUAGE", DEFAULT_TTS_LANGUAGE),
        qwen3_tts_mlx_quantization=_get(
            source,
            "GATEWAY_QWEN3_TTS_MLX_QUANTIZATION",
            DEFAULT_QWEN3_TTS_MLX_QUANTIZATION,
        ),
        enable_live_transcription=_get_bool(source, "GATEWAY_ENABLE_LIVE_TRANSCRIPTION", True),
        speech_to_speech_bin=_get(source, "GATEWAY_SPEECH_TO_SPEECH_BIN", DEFAULT_SPEECH_TO_SPEECH_BIN),
    )
    if validate:
        config.validate()
    return config
