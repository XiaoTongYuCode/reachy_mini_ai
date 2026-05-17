from __future__ import annotations

import logging
import re
import time
import unicodedata
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

DUPLICATE_TRANSCRIPT_WINDOW_SECONDS = 5.0
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_transcript_for_dedupe(transcript: str) -> str:
    """Normalize ASR text enough to catch duplicate completed-transcript events."""
    normalized = unicodedata.normalize("NFKC", transcript).strip()
    return _WHITESPACE_RE.sub(" ", normalized)


def apply_realtime_gateway_patches() -> None:
    """Patch speech-to-speech realtime behavior for Reachy Mini."""
    apply_progressive_faster_whisper_patch()
    apply_transcription_dedupe_patch()


def apply_progressive_faster_whisper_patch() -> None:
    """Make faster-whisper progressive chunks emit partial, not final, transcripts."""
    from speech_to_speech.STT import faster_whisper_handler
    from speech_to_speech.pipeline.messages import PartialTranscription, Transcription

    handler_cls = faster_whisper_handler.FasterWhisperSTTHandler
    if getattr(handler_cls, "_reachy_progressive_patch_applied", False):
        return

    console = faster_whisper_handler.console

    def process(self: Any, vad_audio: Any) -> Any:
        logger.debug("inferring faster whisper...")

        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        if audio.size == 0:
            logger.debug("empty audio buffer. skipping...")
            return
        if not np.isfinite(audio).all():
            logger.warning("Sanitizing non-finite faster-whisper audio buffer.")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        gen_kwargs = dict(self.gen_kwargs)
        gen_kwargs.setdefault("condition_on_previous_text", False)

        segments, _info = self.model.transcribe(audio, **gen_kwargs)
        output_text = []

        for segment in segments:
            logger.debug("[%.2fs -> %.2fs] %s", segment.start, segment.end, segment.text)
            output_text.append(segment.text)

        pred_text = " ".join(output_text).strip()
        logger.debug("finished whisper inference")
        mode = getattr(vad_audio, "mode", None)
        if not pred_text and mode != "progressive":
            pred_text = getattr(self, "_reachy_last_partial_text", "")
            if pred_text:
                logger.debug("Using last partial transcription for empty final chunk: %s", pred_text)

        if not pred_text:
            logger.debug("no text detected. skipping...")
            return

        if mode == "progressive":
            self._reachy_last_partial_text = pred_text
            yield PartialTranscription(text=pred_text)
            return

        self._reachy_last_partial_text = ""
        console.print(f"[yellow]USER: {pred_text}")
        yield Transcription(text=pred_text)

    handler_cls.process = process
    handler_cls._reachy_progressive_patch_applied = True


def apply_transcription_dedupe_patch() -> None:
    """Patch speech-to-speech to suppress duplicate final STT events before LLM enqueue.

    The upstream realtime bridge enqueues an LLM generation directly inside
    ``RealtimeService._on_transcription_completed``. App-side event dedupe runs
    after that point, so duplicate final STT events must be suppressed here.

    The same upstream implicit-response path only marks ``in_response`` once
    audio begins. Mark it active immediately after queueing an LLM request, so
    barge-in speech during LLM/TTS latency can cancel the pending response.
    """
    from speech_to_speech.api.openai_realtime.service import RealtimeService

    if getattr(RealtimeService, "_reachy_transcription_dedupe_patched", False):
        return

    original_on_transcription_completed = RealtimeService._on_transcription_completed
    original_unregister = RealtimeService.unregister

    def _on_transcription_completed(self: Any, conn_id: str, event: Any) -> list[Any]:
        transcript = getattr(event, "transcript", "") or ""
        normalized = normalize_transcript_for_dedupe(transcript)
        if normalized:
            now = time.monotonic()
            recent_by_conn: dict[str, tuple[str, float]] = getattr(
                self,
                "_reachy_recent_completed_transcripts",
                {},
            )
            setattr(self, "_reachy_recent_completed_transcripts", recent_by_conn)

            last = recent_by_conn.get(conn_id)
            if last is not None:
                last_transcript, last_seen_at = last
                if (
                    normalized == last_transcript
                    and now - last_seen_at <= DUPLICATE_TRANSCRIPT_WINDOW_SECONDS
                ):
                    logger.info(
                        "Ignoring duplicate completed STT transcript before LLM trigger: %s",
                        transcript,
                    )
                    return []

            recent_by_conn[conn_id] = (normalized, now)

        events = original_on_transcription_completed(self, conn_id, event)
        if transcript and getattr(self, "text_prompt_queue", None) is not None:
            st = self._state(conn_id)
            st.in_response = True
            should_listen = getattr(self, "should_listen", None)
            if should_listen is not None:
                should_listen.set()
        return events

    def unregister(self: Any, conn_id: str) -> None:
        recent_by_conn = getattr(self, "_reachy_recent_completed_transcripts", None)
        if isinstance(recent_by_conn, dict):
            recent_by_conn.pop(conn_id, None)
        original_unregister(self, conn_id)

    RealtimeService._on_transcription_completed = _on_transcription_completed
    RealtimeService.unregister = unregister
    RealtimeService._reachy_transcription_dedupe_patched = True
