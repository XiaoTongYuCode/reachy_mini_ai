from __future__ import annotations

from collections.abc import Sequence

from reachy_mini_hf_realtime_gateway.transcription_dedupe_patch import apply_realtime_gateway_patches


def main(argv: Sequence[str] | None = None) -> None:
    """Run speech-to-speech with Reachy-specific runtime compatibility patches."""
    import sys
    from speech_to_speech.s2s_pipeline import main as speech_to_speech_main

    apply_realtime_gateway_patches()
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    speech_to_speech_main()


if __name__ == "__main__":
    main()
