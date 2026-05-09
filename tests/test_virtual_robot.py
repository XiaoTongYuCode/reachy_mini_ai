"""Tests for the virtual Reachy Mini test environment."""

import sys

import numpy as np

from reachy_mini_conversation_app.utils import parse_args
from reachy_mini_conversation_app.virtual_robot import create_virtual_reachy_mini


def test_parse_args_accepts_virtual_flag(monkeypatch) -> None:
    """The CLI should expose a hardware-free virtual test environment flag."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app", "--virtual", "--no-camera"])

    args, _ = parse_args()

    assert args.virtual is True
    assert args.no_camera is True


def test_virtual_reachy_mini_reports_simulation_status() -> None:
    """The virtual robot status should trigger simulation-oriented app behavior."""
    robot = create_virtual_reachy_mini(robot_name="ci-robot")

    status = robot.client.get_status()

    assert status["simulation_enabled"] is True
    assert status["mockup_sim_enabled"] is True
    assert status["virtual_test_environment"] is True
    assert robot.robot_name == "ci-robot"


def test_virtual_reachy_mini_records_motion_and_media_calls() -> None:
    """The virtual robot should accept the SDK methods used by movement and media loops."""
    robot = create_virtual_reachy_mini()
    head = np.eye(4, dtype=np.float32)
    antennas = (-0.1, 0.2)

    robot.set_target(head=head, antennas=antennas, body_yaw=0.3)
    body_yaw, current_antennas = robot.get_current_joint_positions()
    current_head = robot.get_current_head_pose()
    frame = robot.media.get_frame()

    assert body_yaw == 0.3
    assert current_antennas == antennas
    assert np.array_equal(current_head, head)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert len(robot.target_history) == 1
