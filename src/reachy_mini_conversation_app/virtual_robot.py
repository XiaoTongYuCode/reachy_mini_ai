"""Virtual Reachy Mini test double for local development and tests."""

# ruff: noqa: D102, D107

from __future__ import annotations
import time
import logging
import threading
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_mini.utils import create_head_pose


logger = logging.getLogger(__name__)


class VirtualAudio:
    """No-op audio controls used by the virtual media stack."""

    def __init__(self) -> None:
        self._playback_next_pts_ns = 0

    def _get_playback_running_time_ns(self) -> int:
        return 0

    def clear_player(self) -> None:
        """Drop queued virtual playback samples."""
        self._playback_next_pts_ns = 0

    def clear_output_buffer(self) -> None:
        """Drop queued virtual output samples."""
        self.clear_player()

    def apply_audio_config(self, *_args: Any, **_kwargs: Any) -> bool:
        """Pretend the startup audio configuration was applied."""
        return True


class VirtualMedia:
    """In-memory media backend with silent audio and synthetic camera frames."""

    def __init__(self, *, sample_rate: int = 24_000, frame_shape: tuple[int, int, int] = (480, 640, 3)) -> None:
        self.backend = "virtual"
        self.audio = VirtualAudio()
        self._sample_rate = sample_rate
        self._frame_shape = frame_shape
        self._recording = False
        self._playing = False
        self._closed = False
        self.pushed_audio_samples: list[NDArray[np.float32]] = []

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def start_playing(self) -> None:
        self._playing = True

    def stop_playing(self) -> None:
        self._playing = False

    def get_input_audio_samplerate(self) -> int:
        return self._sample_rate

    def get_output_audio_samplerate(self) -> int:
        return self._sample_rate

    def get_audio_sample(self) -> None:
        """Return no microphone data so tests can drive handlers explicitly."""
        return None

    def push_audio_sample(self, audio_frame: NDArray[np.float32]) -> None:
        self.pushed_audio_samples.append(audio_frame)
        self.audio._playback_next_pts_ns += int(len(audio_frame) / self._sample_rate * 1e9)

    def get_frame(self) -> NDArray[np.uint8]:
        return np.zeros(self._frame_shape, dtype=np.uint8)

    def close(self) -> None:
        self._closed = True
        self.stop_recording()
        self.stop_playing()


class VirtualClient:
    """Client facade exposing SDK-like status and disconnect methods."""

    def __init__(self) -> None:
        self.disconnected = False

    def get_status(self) -> Dict[str, bool | str]:
        return {
            "simulation_enabled": True,
            "mockup_sim_enabled": True,
            "virtual_test_environment": True,
            "robot_name": "virtual-reachy-mini",
        }

    def disconnect(self) -> None:
        self.disconnected = True


class VirtualReachyMini:
    """Small Reachy Mini-compatible object for hardware-free app testing."""

    def __init__(self, *, robot_name: str = "virtual-reachy-mini") -> None:
        self.robot_name = robot_name
        self.media = VirtualMedia()
        self.client = VirtualClient()
        self._lock = threading.Lock()
        self._head: NDArray[np.float32] = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True).astype(np.float32)
        self._antennas: Tuple[float, float] = (-0.1745, 0.1745)
        self._body_yaw = 0.0
        self.target_history: list[tuple[NDArray[np.float32], Tuple[float, float], float, float]] = []

    def set_target(
        self,
        *,
        head: NDArray[np.float32],
        antennas: tuple[float, float] | list[float],
        body_yaw: float = 0.0,
    ) -> None:
        with self._lock:
            self._head = np.asarray(head, dtype=np.float32).copy()
            self._antennas = (float(antennas[0]), float(antennas[1]))
            self._body_yaw = float(body_yaw)
            self.target_history.append((self._head.copy(), self._antennas, self._body_yaw, time.monotonic()))

    def goto_target(
        self,
        *,
        head: NDArray[np.float32],
        antennas: tuple[float, float] | list[float],
        duration: float = 0.0,
        body_yaw: float = 0.0,
    ) -> None:
        self.set_target(head=head, antennas=antennas, body_yaw=body_yaw)

    def get_current_joint_positions(self) -> tuple[float, Tuple[float, float]]:
        with self._lock:
            return self._body_yaw, self._antennas

    def get_current_head_pose(self) -> NDArray[np.float32]:
        with self._lock:
            return self._head.copy()

    def look_at_image(
        self,
        x: float,
        y: float,
        *,
        duration: float = 0.0,
        perform_movement: bool = False,
    ) -> NDArray[np.float32]:
        del duration, perform_movement
        # Map image coordinates into a tiny deterministic offset for head-tracking tests.
        width = max(1.0, float(self.media._frame_shape[1]))
        height = max(1.0, float(self.media._frame_shape[0]))
        x_offset = (float(x) / width - 0.5) * 0.04
        y_offset = (float(y) / height - 0.5) * 0.04
        return create_head_pose(x=x_offset, y=y_offset, z=0, roll=0, pitch=0, yaw=0, degrees=False, mm=False).astype(
            np.float32
        )


def create_virtual_reachy_mini(*, robot_name: str | None = None) -> VirtualReachyMini:
    """Create a hardware-free Reachy Mini-compatible robot."""
    name = robot_name or "virtual-reachy-mini"
    logger.info("Using virtual Reachy Mini test environment: %s", name)
    return VirtualReachyMini(robot_name=name)
