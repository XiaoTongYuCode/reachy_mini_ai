import sys
from types import ModuleType
from typing import Any
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

from reachy_mini_conversation_app.config import config


def _load_probe_module() -> ModuleType:
    """Load the repo-root probe script as a module for testing."""
    probe_path = Path(__file__).resolve().parents[1] / "probe_s2s_realtime_audio.py"
    spec = spec_from_file_location("probe_s2s_realtime_audio", probe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load probe module from {probe_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_session_update_payload_omits_voice_when_unset(monkeypatch: Any) -> None:
    """Probe payloads should omit voice when no profile voice is configured."""
    probe = _load_probe_module()
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "speech-to-speech")
    monkeypatch.setattr(probe, "get_tool_specs", lambda: [])

    payload = probe.build_session_update_payload(
        probe.ProbeArguments(send_rate=16000, recv_rate=16000),
        instructions="test instructions",
        voice=None,
    )

    output = payload["session"]["audio"]["output"]
    assert "voice" not in output


def test_build_session_update_payload_includes_explicit_voice(monkeypatch: Any) -> None:
    """Probe payloads should preserve an explicitly selected voice."""
    probe = _load_probe_module()
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "speech-to-speech")
    monkeypatch.setattr(probe, "get_tool_specs", lambda: [])

    payload = probe.build_session_update_payload(
        probe.ProbeArguments(send_rate=16000, recv_rate=16000),
        instructions="test instructions",
        voice="Aiden",
    )

    output = payload["session"]["audio"]["output"]
    assert output["voice"] == "Aiden"
