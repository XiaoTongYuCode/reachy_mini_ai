from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Sequence

from reachy_mini_hf_realtime_gateway.config import DEFAULT_SPEECH_TO_SPEECH_BIN, GatewayConfig


SECRET_VALUE_FLAGS = frozenset(
    {
        "--responses_api_api_key",
    }
)
SECRET_VALUE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^sk-[A-Za-z0-9_-]{8,}",
        r"^nvapi-[A-Za-z0-9_-]{8,}",
        r"^hf[_-][A-Za-z0-9_-]{8,}",
        r"^AIza[0-9A-Za-z_-]{8,}",
    )
)


def build_command(config: GatewayConfig) -> list[str]:
    """Build the speech-to-speech realtime gateway command."""
    config.validate()
    command = _speech_to_speech_command_prefix(config)
    command.extend(
        [
            "--mode",
            "realtime",
            "--ws_host",
            config.host,
            "--ws_port",
            str(config.port),
            "--stt",
            config.stt,
            "--language",
            config.language,
            "--min_silence_ms",
            str(config.vad_min_silence_ms),
            "--min_speech_ms",
            str(config.vad_min_speech_ms),
            "--speech_pad_ms",
            str(config.vad_speech_pad_ms),
            "--llm_backend",
            config.llm_backend,
            "--responses_api_base_url",
            config.llm_base_url,
        ]
    )

    if config.stt == "faster-whisper":
        command.extend(
            [
                "--faster_whisper_stt_model_name",
                config.stt_model,
                "--faster_whisper_stt_gen_language",
                config.language,
                "--faster_whisper_stt_device",
                config.faster_whisper_stt_device,
                "--faster_whisper_stt_compute_type",
                config.faster_whisper_stt_compute_type,
                "--faster_whisper_stt_gen_max_new_tokens",
                str(config.faster_whisper_stt_gen_max_new_tokens),
                "--faster_whisper_stt_gen_beam_size",
                str(config.faster_whisper_stt_gen_beam_size),
            ]
        )
    else:
        command.extend(["--stt_model_name", config.stt_model])

    if config.llm_api_key:
        command.extend(["--responses_api_api_key", config.llm_api_key])

    command.extend(["--model_name", config.llm_model])

    if config.responses_api_stream:
        command.append("--responses_api_stream")
    if not config.responses_api_disable_thinking:
        command.append("--no_responses_api_disable_thinking")
    command.extend(["--stream_batch_sentences", str(config.llm_stream_batch_sentences)])

    command.extend(["--tts", config.tts])

    if config.tts == "qwen3":
        command.extend(
            [
                "--qwen3_tts_model_name",
                config.tts_model,
                "--qwen3_tts_speaker",
                config.tts_speaker,
                "--qwen3_tts_language",
                config.tts_language,
                "--qwen3_tts_mlx_quantization",
                config.qwen3_tts_mlx_quantization,
            ]
        )

    if config.enable_live_transcription:
        command.extend(
            [
                "--enable_live_transcription",
                "--live_transcription_update_interval",
                str(config.live_transcription_update_interval),
            ]
        )
    else:
        command.append("--no_enable_live_transcription")

    return command


def _speech_to_speech_command_prefix(config: GatewayConfig) -> list[str]:
    """Return the command prefix, applying Reachy runtime patches for the default binary."""
    if config.speech_to_speech_bin != DEFAULT_SPEECH_TO_SPEECH_BIN:
        return [config.speech_to_speech_bin]
    return [sys.executable, "-m", "reachy_mini_hf_realtime_gateway.speech_to_speech_runner"]


def redact_command(command: Sequence[str]) -> list[str]:
    """Return command args with secret flag values redacted."""
    redacted: list[str] = []
    redact_next = False
    for arg in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append("<redacted>" if _looks_like_secret(arg) else arg)
        if arg in SECRET_VALUE_FLAGS:
            redact_next = True
    return redacted


def _looks_like_secret(value: str) -> bool:
    return any(pattern.match(value) for pattern in SECRET_VALUE_PATTERNS)


def format_command(command: Sequence[str], *, redact: bool = True) -> str:
    """Return a shell-safe command string for logs and dry runs."""
    display_args = redact_command(command) if redact else list(command)
    return " ".join(shlex.quote(arg) for arg in display_args)
