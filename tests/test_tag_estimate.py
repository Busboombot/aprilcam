"""Tests for TagRecord.estimate() and world-space velocity computation."""

import math
import time

import numpy as np
import pytest

from aprilcam.detection import TagRecord
from aprilcam.models import AprilTag, AprilTagFlow
from aprilcam.playfield import Playfield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tag_record(
    *,
    center_px=(100.0, 200.0),
    vel_px=(10.0, 0.0),
    vel_world=None,
    world_xy=None,
    timestamp=1000.0,
    age=0.0,
) -> TagRecord:
    corners = [
        [center_px[0] - 5, center_px[1] - 5],
        [center_px[0] + 5, center_px[1] - 5],
        [center_px[0] + 5, center_px[1] + 5],
        [center_px[0] - 5, center_px[1] + 5],
    ]
    return TagRecord(
        id=1,
        center_px=center_px,
        corners_px=corners,
        orientation_yaw=0.0,
        world_xy=world_xy,
        in_playfield=True,
        vel_px=vel_px,
        speed_px=math.hypot(*vel_px) if vel_px else None,
        vel_world=vel_world,
        speed_world=math.hypot(*vel_world) if vel_world else None,
        heading_rad=math.atan2(vel_world[1], vel_world[0]) if vel_world else None,
        timestamp=timestamp,
        frame_index=42,
        age=age,
    )


# ---------------------------------------------------------------------------
# TagRecord.estimate() tests
# ---------------------------------------------------------------------------

class TestEstimateExplicitTime:
    """Tests for estimate(t) with an explicit target time."""

    def test_shifts_center(self):
        tr = _make_tag_record(center_px=(100.0, 200.0), vel_px=(5.0, -3.0), timestamp=1000.0)
        est = tr.estimate(1000.2)
        assert est.center_px[0] == pytest.approx(101.0, abs=0.01)
        assert est.center_px[1] == pytest.approx(199.4, abs=0.01)

    def test_shifts_all_corners(self):
        tr = _make_tag_record(vel_px=(10.0, 20.0), timestamp=1000.0)
        est = tr.estimate(1000.1)
        dx, dy = 10.0 * 0.1, 20.0 * 0.1
        for orig, shifted in zip(tr.corners_px, est.corners_px):
            assert shifted[0] == pytest.approx(orig[0] + dx, abs=0.01)
            assert shifted[1] == pytest.approx(orig[1] + dy, abs=0.01)

    def test_shifts_world_xy(self):
        tr = _make_tag_record(
            world_xy=(100.0, 200.0),
            vel_world=(50.0, 0.0),
            timestamp=1000.0,
        )
        est = tr.estimate(1000.1)
        assert est.world_xy[0] == pytest.approx(105.0, abs=0.01)
        assert est.world_xy[1] == pytest.approx(200.0, abs=0.01)

    def test_timestamp_set(self):
        tr = _make_tag_record(timestamp=1000.0)
        est = tr.estimate(1000.5)
        assert est.timestamp == 1000.5

    def test_age_updated(self):
        tr = _make_tag_record(timestamp=1000.0, age=0.0)
        est = tr.estimate(1000.3)
        assert est.age == pytest.approx(0.3, abs=0.001)

    def test_preserves_other_fields(self):
        tr = _make_tag_record(vel_px=(10.0, 5.0), timestamp=1000.0)
        est = tr.estimate(1000.1)
        assert est.id == tr.id
        assert est.orientation_yaw == tr.orientation_yaw
        assert est.vel_px == tr.vel_px
        assert est.speed_px == tr.speed_px
        assert est.frame_index == tr.frame_index
        assert est.in_playfield == tr.in_playfield


class TestEstimateDefaultTime:
    """Tests for estimate() with no argument (defaults to now)."""

    def test_shifts_center_to_now(self):
        now = time.monotonic()
        tr = _make_tag_record(center_px=(100.0, 200.0), vel_px=(10.0, 0.0), timestamp=now - 0.1)
        est = tr.estimate()
        # Should have shifted by approximately 10 * 0.1 = 1.0 pixel
        assert est.center_px[0] == pytest.approx(101.0, abs=0.5)


class TestEstimateEdgeCases:
    """Edge cases: no velocity, no world data."""

    def test_no_vel_px_unchanged(self):
        tr = _make_tag_record(vel_px=None, world_xy=(10.0, 20.0))
        est = tr.estimate(tr.timestamp + 0.5)
        assert est.center_px == tr.center_px
        assert est.corners_px == tr.corners_px

    def test_no_vel_world_world_xy_unchanged(self):
        tr = _make_tag_record(vel_world=None, world_xy=(10.0, 20.0))
        est = tr.estimate(tr.timestamp + 0.5)
        assert est.world_xy == tr.world_xy

    def test_no_world_xy_stays_none(self):
        tr = _make_tag_record(world_xy=None, vel_world=None)
        est = tr.estimate(tr.timestamp + 0.1)
        assert est.world_xy is None


# ---------------------------------------------------------------------------
# World-velocity computation tests (Playfield.add_tag with homography)
# ---------------------------------------------------------------------------

def _make_apriltag(tag_id, cx, cy, timestamp):
    """Create an AprilTag with center at (cx, cy)."""
    corners = np.array([
        [cx - 5, cy - 5],
        [cx + 5, cy - 5],
        [cx + 5, cy + 5],
        [cx - 5, cy + 5],
    ], dtype=np.float32)
    return AprilTag.from_corners(tag_id, corners, timestamp=timestamp)


class TestWorldVelocity:
    """Test that Playfield.add_tag() computes world velocity via homography."""

    def test_scale_homography(self):
        """With a 2x scaling homography, world speed should be ~2x pixel speed."""
        pf = Playfield(proc_width=960, deadband_threshold=0.0)
        H = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]], dtype=float)

        t0, t1, t2 = 1000.0, 1000.1, 1000.2
        tag0 = _make_apriltag(1, 100.0, 200.0, t0)
        tag1 = _make_apriltag(1, 110.0, 200.0, t1)
        tag2 = _make_apriltag(1, 120.0, 200.0, t2)

        pf.add_tag(tag0, homography=H)
        pf.add_tag(tag1, homography=H)
        pf.add_tag(tag2, homography=H)

        flow = pf.get_flows()[1]
        assert flow.vel_world is not None
        assert flow.speed_world is not None
        assert flow.speed_world == pytest.approx(flow.speed_px * 2.0, rel=0.1)

    def test_world_velocity_direction(self):
        """World velocity direction should match pixel velocity direction."""
        pf = Playfield(proc_width=960, deadband_threshold=0.0)
        H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=float)

        tag0 = _make_apriltag(1, 100.0, 200.0, 1000.0)
        tag1 = _make_apriltag(1, 100.0, 210.0, 1000.1)

        pf.add_tag(tag0, homography=H)
        pf.add_tag(tag1, homography=H)

        flow = pf.get_flows()[1]
        assert flow.vel_world is not None
        # Movement is purely in y direction
        assert abs(flow.vel_world[0]) < 1.0  # negligible x
        assert flow.vel_world[1] > 50.0  # significant positive y

    def test_no_homography_world_velocity_none(self):
        """Without homography, world velocity fields stay None."""
        pf = Playfield(proc_width=960, deadband_threshold=0.0)

        tag0 = _make_apriltag(1, 100.0, 200.0, 1000.0)
        tag1 = _make_apriltag(1, 110.0, 200.0, 1000.1)

        pf.add_tag(tag0)
        pf.add_tag(tag1)

        flow = pf.get_flows()[1]
        assert flow.vel_world is None
        assert flow.speed_world is None
        assert flow.heading_rad is None

    def test_heading_rad_correct(self):
        """Heading should point in direction of motion."""
        pf = Playfield(proc_width=960, deadband_threshold=0.0)
        H = np.eye(3, dtype=float)

        tag0 = _make_apriltag(1, 100.0, 100.0, 1000.0)
        tag1 = _make_apriltag(1, 110.0, 110.0, 1000.1)  # 45 degrees

        pf.add_tag(tag0, homography=H)
        pf.add_tag(tag1, homography=H)

        flow = pf.get_flows()[1]
        assert flow.heading_rad == pytest.approx(math.pi / 4, abs=0.1)
