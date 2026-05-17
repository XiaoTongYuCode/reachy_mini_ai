from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import SpeechStartedEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import VADAudio

from reachy_mini_hf_realtime_gateway import transcription_dedupe_patch as patch_mod
from reachy_mini_hf_realtime_gateway.transcription_dedupe_patch import (
    apply_progressive_faster_whisper_patch,
    apply_realtime_gateway_patches,
    apply_transcription_dedupe_patch,
    normalize_transcript_for_dedupe,
)


def test_normalize_transcript_for_dedupe_collapses_spacing_and_width() -> None:
    assert normalize_transcript_for_dedupe("  hello　world  ") == "hello world"


def test_transcription_dedupe_patch_suppresses_gateway_llm_enqueue(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(patch_mod.time, "monotonic", lambda: now)
    apply_transcription_dedupe_patch()

    text_prompt_queue = Queue()
    service = RealtimeService(text_prompt_queue=text_prompt_queue)
    conn_id = service.register()

    event = TranscriptionCompletedEvent(transcript="洗了下香皂", language_code="zh")
    first_events = service._on_transcription_completed(conn_id, event)
    duplicate_events = service._on_transcription_completed(conn_id, event)

    assert first_events
    assert duplicate_events == []
    assert text_prompt_queue.qsize() == 1


def test_transcription_dedupe_patch_allows_repeat_after_window(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(patch_mod.time, "monotonic", lambda: now)
    apply_transcription_dedupe_patch()

    text_prompt_queue = Queue()
    service = RealtimeService(text_prompt_queue=text_prompt_queue)
    conn_id = service.register()
    event = TranscriptionCompletedEvent(transcript="洗了下香皂", language_code="zh")

    service._on_transcription_completed(conn_id, event)
    now = 106.0
    service._on_transcription_completed(conn_id, event)

    assert text_prompt_queue.qsize() == 2


def test_transcription_completion_marks_implicit_response_active_for_barge_in(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(patch_mod.time, "monotonic", lambda: now)
    apply_transcription_dedupe_patch()

    should_listen = Event()
    text_prompt_queue = Queue()
    service = RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen)
    conn_id = service.register()
    should_listen.clear()

    service._on_transcription_completed(
        conn_id,
        TranscriptionCompletedEvent(transcript="響鐘", language_code="zh"),
    )

    assert service._state(conn_id).in_response is True
    assert should_listen.is_set()

    events = service.audio.on_speech_started(conn_id, SpeechStartedEvent(audio_start_ms=123))

    assert any(event.type == "response.done" for event in events)
    assert service._state(conn_id).in_response is False


def test_progressive_faster_whisper_outputs_partial_transcription() -> None:
    apply_progressive_faster_whisper_patch()

    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler
    from speech_to_speech.pipeline.messages import PartialTranscription, Transcription

    handler = object.__new__(FasterWhisperSTTHandler)
    handler.gen_kwargs = {}
    handler.model = SimpleNamespace(
        transcribe=lambda _audio, **_kwargs: (
            [SimpleNamespace(start=0.0, end=0.5, text="你好")],
            SimpleNamespace(),
        )
    )

    audio = np.zeros(16000, dtype=np.float32)
    progressive = list(handler.process(VADAudio(audio=audio, mode="progressive")))
    final = list(handler.process(VADAudio(audio=audio, mode="final")))

    assert progressive == [PartialTranscription(text="你好")]
    assert final == [Transcription(text="你好")]


def test_progressive_faster_whisper_sanitizes_non_finite_audio() -> None:
    apply_progressive_faster_whisper_patch()

    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler
    from speech_to_speech.pipeline.messages import Transcription

    seen = {}

    def transcribe(audio, **_kwargs):
        seen["all_finite"] = bool(np.isfinite(audio).all())
        seen["min"] = float(np.min(audio))
        seen["max"] = float(np.max(audio))
        return [SimpleNamespace(start=0.0, end=0.5, text="你好")], SimpleNamespace()

    handler = object.__new__(FasterWhisperSTTHandler)
    handler.gen_kwargs = {}
    handler.model = SimpleNamespace(transcribe=transcribe)

    audio = np.array([0.0, np.nan, np.inf, -np.inf, 2.0, -2.0], dtype=np.float32)
    final = list(handler.process(VADAudio(audio=audio, mode="final")))

    assert final == [Transcription(text="你好")]
    assert seen == {"all_finite": True, "min": -1.0, "max": 1.0}


def test_progressive_faster_whisper_uses_last_partial_when_final_is_empty() -> None:
    apply_progressive_faster_whisper_patch()

    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler
    from speech_to_speech.pipeline.messages import PartialTranscription, Transcription

    calls = {"count": 0}

    def transcribe(_audio, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return [SimpleNamespace(start=0.0, end=0.5, text="你好")], SimpleNamespace()
        return [], SimpleNamespace()

    handler = object.__new__(FasterWhisperSTTHandler)
    handler.gen_kwargs = {}
    handler.model = SimpleNamespace(transcribe=transcribe)

    audio = np.zeros(16000, dtype=np.float32)
    progressive = list(handler.process(VADAudio(audio=audio, mode="progressive")))
    final = list(handler.process(VADAudio(audio=audio, mode="final")))

    assert progressive == [PartialTranscription(text="你好")]
    assert final == [Transcription(text="你好")]


def test_apply_realtime_gateway_patches_is_idempotent() -> None:
    apply_realtime_gateway_patches()
    apply_realtime_gateway_patches()
