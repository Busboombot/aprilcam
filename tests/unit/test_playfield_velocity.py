"""Tests for velocity computation in Playfield.add_tag() and AprilTagFlow.set_velocity()."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aprilcam.models import AprilTag, AprilTagFlow
from aprilcam.playfield import Playfield


def _make_tag(tag_id: int, cx: float, cy: float, ts: float) -> AprilTag:
    """Create a minimal AprilTag at the given center with a dummy 10x10 square."""
    half = 5.0
    corners = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ], dtype=np.float32)
    return AprilTag.from_corners(tag_id, corners, timestamp=ts)


class TestPlayfieldVelocityFirstTag:
    """First observation has zero velocity (no previous position to diff)."""

    def test_playfield_velocity_first_tag(self) -> None:
        pf = Playfield()
        tag = _make_tag(1, 100.0, 200.0, ts=1.0)
        pf.add_tag(tag)

        flow = pf.get_flows()[1]
        assert flow.vel_px == (0.0, 0.0)
        assert flow.speed_px == 0.0


class TestPlayfieldVelocityMovingTag:
    """Two observations with large position change produces non-zero velocity."""

    def test_playfield_velocity_moving_tag(self) -> None:
        pf = Playfield(ema_alpha=0.3, deadband_threshold=50.0)

        # First observation
        tag1 = _make_tag(1, 100.0, 100.0, ts=1.0)
        pf.add_tag(tag1)

        # Second observation: moved 200px in x over 1 second => 200 px/s
        tag2 = _make_tag(1, 300.0, 100.0, ts=2.0)
        pf.add_tag(tag2)

        flow = pf.get_flows()[1]
        # inst_speed = 200.0, first EMA observation => smoothed = 200.0
        # 200 > 50 deadband => velocity should be reported
        assert flow.speed_px == pytest.approx(200.0)
        assert flow.vel_px[0] == pytest.approx(200.0)  # dx/dt
        assert flow.vel_px[1] == pytest.approx(0.0)    # dy/dt


class TestPlayfieldVelocityDeadband:
    """Small movement below threshold produces zero velocity."""

    def test_playfield_velocity_deadband(self) -> None:
        pf = Playfield(ema_alpha=0.3, deadband_threshold=50.0)

        tag1 = _make_tag(1, 100.0, 100.0, ts=1.0)
        pf.add_tag(tag1)

        # Move only 10px in 1 second => 10 px/s, well below 50 deadband
        tag2 = _make_tag(1, 110.0, 100.0, ts=2.0)
        pf.add_tag(tag2)

        flow = pf.get_flows()[1]
        assert flow.vel_px == (0.0, 0.0)
        assert flow.speed_px == 0.0


class TestPlayfieldVelocityEmaSmoothing:
    """Verify EMA smoothing with known inputs over multiple observations."""

    def test_playfield_velocity_ema_smoothing(self) -> None:
        alpha = 0.3
        pf = Playfield(ema_alpha=alpha, deadband_threshold=0.0)  # no deadband

        # Observation 1: baseline
        pf.add_tag(_make_tag(1, 0.0, 0.0, ts=0.0))

        # Observation 2: move 100px in 1s => inst_speed = 100
        # First EMA value: smoothed = 100.0 (no previous EMA)
        pf.add_tag(_make_tag(1, 100.0, 0.0, ts=1.0))

        flow = pf.get_flows()[1]
        assert flow.speed_px == pytest.approx(100.0)

        # Observation 3: move another 200px in 1s => inst_speed = 200
        # EMA: 0.3 * 200 + 0.7 * 100 = 60 + 70 = 130
        pf.add_tag(_make_tag(1, 300.0, 0.0, ts=2.0))
        assert flow.speed_px == pytest.approx(130.0)

        # Observation 4: move 0px in 1s => inst_speed = 0
        # EMA: 0.3 * 0 + 0.7 * 130 = 91
        pf.add_tag(_make_tag(1, 300.0, 0.0, ts=3.0))
        assert flow.speed_px == pytest.approx(91.0)


class TestAprilTagFlowSetVelocity:
    """Verify set_velocity method works on AprilTagFlow."""

    def test_apriltagflow_set_velocity(self) -> None:
        flow = AprilTagFlow(maxlen=5)

        # Default values
        assert flow.vel_px == (0.0, 0.0)
        assert flow.speed_px == 0.0

        # Set velocity
        flow.set_velocity((150.0, -75.0), 167.7)
        assert flow.vel_px == (150.0, -75.0)
        assert flow.speed_px == pytest.approx(167.7)

        # Overwrite velocity
        flow.set_velocity((0.0, 0.0), 0.0)
        assert flow.vel_px == (0.0, 0.0)
        assert flow.speed_px == 0.0


class TestAprilCamNoVelocityState:
    """Verify AprilCam no longer has _vel_ema or _last_seen."""

    def test_aprilcam_no_velocity_state(self) -> None:
        from aprilcam.aprilcam import AprilCam

        cam = AprilCam(
            index=0,
            backend=None,
            speed_alpha=0.3,
            family="36h11",
            proc_width=960,
            headless=True,
        )
        assert not hasattr(cam, "_vel_ema")
        assert not hasattr(cam, "_last_seen")

        # Also verify reset_state doesn't re-introduce them
        cam.reset_state()
        assert not hasattr(cam, "_vel_ema")
        assert not hasattr(cam, "_last_seen")
