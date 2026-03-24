"""Tests for Composite, CompositeManager, and cross-camera homography utilities."""

import math

import numpy as np
import pytest

from aprilcam.composite import (
    Composite,
    CompositeManager,
    compute_cross_camera_homography,
    map_tags_to_primary,
)


# ---------------------------------------------------------------------------
# Composite dataclass
# ---------------------------------------------------------------------------


class TestComposite:
    def test_fields(self):
        H = np.eye(3)
        c = Composite(
            composite_id="abc",
            primary_camera_id="cam1",
            secondary_camera_id="cam2",
            homography=H,
            reprojection_error=0.5,
            playfield_id="pf_1",
        )
        assert c.composite_id == "abc"
        assert c.primary_camera_id == "cam1"
        assert c.secondary_camera_id == "cam2"
        assert c.reprojection_error == 0.5
        assert c.playfield_id == "pf_1"
        np.testing.assert_array_equal(c.homography, H)

    def test_playfield_id_defaults_none(self):
        c = Composite(
            composite_id="x",
            primary_camera_id="a",
            secondary_camera_id="b",
            homography=np.eye(3),
            reprojection_error=0.0,
        )
        assert c.playfield_id is None


# ---------------------------------------------------------------------------
# CompositeManager
# ---------------------------------------------------------------------------


class TestCompositeManager:
    def test_create_returns_composite_with_uuid(self):
        mgr = CompositeManager()
        comp = mgr.create("cam1", "cam2", np.eye(3), 0.1)
        assert len(comp.composite_id.split("-")) == 5  # UUID4 format
        assert comp.primary_camera_id == "cam1"
        assert comp.secondary_camera_id == "cam2"

    def test_get_returns_registered(self):
        mgr = CompositeManager()
        comp = mgr.create("cam1", "cam2", np.eye(3), 0.0)
        assert mgr.get(comp.composite_id) is comp

    def test_get_raises_keyerror(self):
        mgr = CompositeManager()
        with pytest.raises(KeyError):
            mgr.get("nonexistent")

    def test_destroy_removes(self):
        mgr = CompositeManager()
        comp = mgr.create("cam1", "cam2", np.eye(3), 0.0)
        mgr.destroy(comp.composite_id)
        with pytest.raises(KeyError):
            mgr.get(comp.composite_id)

    def test_destroy_raises_keyerror(self):
        mgr = CompositeManager()
        with pytest.raises(KeyError):
            mgr.destroy("nonexistent")

    def test_list_returns_ids(self):
        mgr = CompositeManager()
        c1 = mgr.create("a", "b", np.eye(3), 0.0)
        c2 = mgr.create("c", "d", np.eye(3), 0.0)
        ids = mgr.list()
        assert set(ids) == {c1.composite_id, c2.composite_id}

    def test_list_empty(self):
        mgr = CompositeManager()
        assert mgr.list() == []

    def test_create_with_playfield_id(self):
        mgr = CompositeManager()
        comp = mgr.create("a", "b", np.eye(3), 0.0, playfield_id="pf_x")
        assert comp.playfield_id == "pf_x"


# ---------------------------------------------------------------------------
# compute_cross_camera_homography
# ---------------------------------------------------------------------------


class TestComputeCrossCameraHomography:
    def _make_identity_points(self):
        """4 points that are the same in both cameras -> identity homography."""
        pts = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float64)
        return pts, pts.copy()

    def test_identity_case(self):
        pri, sec = self._make_identity_points()
        H, err = compute_cross_camera_homography(pri, sec)
        assert H.shape == (3, 3)
        # Identity-ish homography, near-zero error
        assert err < 1.0
        # H should be close to identity (up to scale)
        H_norm = H / H[2, 2]
        np.testing.assert_allclose(H_norm, np.eye(3), atol=1e-6)

    def test_translation(self):
        """Secondary camera sees everything shifted by (50, 30)."""
        pri = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float64)
        sec = pri + np.array([50, 30])
        H, err = compute_cross_camera_homography(pri, sec)
        assert err < 1.0

        # Transform a secondary point and check it maps to primary
        test_pt = np.array([[[150.0, 130.0]]], dtype=np.float64)
        import cv2
        mapped = cv2.perspectiveTransform(test_pt, H)
        np.testing.assert_allclose(mapped[0, 0], [100.0, 100.0], atol=1.0)

    def test_fewer_than_4_points_raises(self):
        pts3 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="At least 4"):
            compute_cross_camera_homography(pts3, pts3)

    def test_mismatched_lengths_raises(self):
        p4 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        p5 = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]], dtype=np.float64)
        with pytest.raises(ValueError, match="same length"):
            compute_cross_camera_homography(p4, p5)

    def test_degenerate_raises(self):
        """All points identical -> degenerate."""
        pts = np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=np.float64)
        with pytest.raises(ValueError):
            compute_cross_camera_homography(pts, pts)

    def test_more_than_4_points(self):
        """Works with more than 4 correspondences."""
        pri = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100],
            [50, 50], [75, 25],
        ], dtype=np.float64)
        sec = pri * 2  # scale factor
        H, err = compute_cross_camera_homography(pri, sec)
        assert H.shape == (3, 3)
        assert err < 1.0


# ---------------------------------------------------------------------------
# map_tags_to_primary
# ---------------------------------------------------------------------------


class TestMapTagsToPrimary:
    def test_identity_homography(self):
        """With identity H, mapped coords should equal originals."""
        H = np.eye(3)
        corners = np.array([
            [10, 10], [50, 10], [50, 50], [10, 50]
        ], dtype=np.float32)
        detections = [(corners, np.zeros((10, 10), dtype=np.uint8), 42)]

        result = map_tags_to_primary(detections, H)
        assert len(result) == 1
        r = result[0]
        assert r["id"] == 42
        np.testing.assert_allclose(r["center_px"], [30.0, 30.0], atol=1e-4)
        assert len(r["corners_px"]) == 4
        assert "orientation_yaw" in r

    def test_translation_homography(self):
        """Shift all coords by (100, 200)."""
        H = np.array([
            [1, 0, 100],
            [0, 1, 200],
            [0, 0, 1],
        ], dtype=np.float64)
        corners = np.array([
            [0, 0], [40, 0], [40, 40], [0, 40]
        ], dtype=np.float32)
        detections = [(corners, None, 7)]

        result = map_tags_to_primary(detections, H)
        r = result[0]
        np.testing.assert_allclose(r["center_px"], [120.0, 220.0], atol=1e-4)

    def test_empty_detections(self):
        result = map_tags_to_primary([], np.eye(3))
        assert result == []

    def test_multiple_tags(self):
        H = np.eye(3)
        c1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        c2 = np.array([[100, 100], [110, 100], [110, 110], [100, 110]], dtype=np.float32)
        detections = [
            (c1, None, 1),
            (c2, None, 2),
        ]
        result = map_tags_to_primary(detections, H)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_orientation_yaw(self):
        """Tag with top edge pointing right -> yaw ~0."""
        H = np.eye(3)
        # Square tag, top edge from (0,0) to (40,0), center at (20,20)
        # top_mid = (20, 0), d = (20,0) - (20,20) = (0, -20), yaw = atan2(-20, 0) = -pi/2
        corners = np.array([
            [0, 0], [40, 0], [40, 40], [0, 40]
        ], dtype=np.float32)
        detections = [(corners, None, 5)]
        result = map_tags_to_primary(detections, H)
        yaw = result[0]["orientation_yaw"]
        # Top midpoint is (20,0), center is (20,20), direction is (0,-20) -> atan2(-20,0) = -pi/2
        assert abs(yaw - (-math.pi / 2)) < 0.01
