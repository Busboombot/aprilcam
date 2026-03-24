"""Tests for composite MCP tools: create_composite, get_composite_frame, get_composite_tags."""

import asyncio
import json

import cv2
import numpy as np
import pytest

from aprilcam.mcp_server import (
    composite_manager,
    create_composite,
    registry,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_aruco_image(marker_ids, positions, img_size=(640, 480), marker_size=80):
    """Create a synthetic BGR image with ArUco 4x4 markers at given positions.

    Args:
        marker_ids: list of int marker IDs.
        positions: list of (x, y) top-left corner positions for each marker.
        img_size: (width, height) of the output image.
        marker_size: pixel size of each marker (square).

    Returns:
        BGR image (numpy array) with markers drawn.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200  # light gray

    for mid, (x, y) in zip(marker_ids, positions):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_size)
        # Convert to BGR
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        # Place on canvas (clip to bounds)
        y2 = min(y + marker_size, img_size[1])
        x2 = min(x + marker_size, img_size[0])
        mh = y2 - y
        mw = x2 - x
        canvas[y:y2, x:x2] = marker_bgr[:mh, :mw]

    return canvas


class FakeCapture:
    """Mock camera that returns a pre-built image."""

    def __init__(self, image: np.ndarray):
        self._img = image

    def read(self):
        return True, self._img.copy()

    def release(self):
        pass


@pytest.fixture(autouse=True)
def clean_registries():
    """Reset composite manager and camera registry."""
    composite_manager._composites.clear()
    yield
    composite_manager._composites.clear()
    # Clean up any cameras we registered
    registry._cameras.clear()


# ---------------------------------------------------------------------------
# create_composite: manual correspondence_points
# ---------------------------------------------------------------------------


class TestCreateCompositeManual:
    def test_manual_correspondence_points(self):
        # 4 point pairs: primary and secondary at same positions -> identity
        points = [
            [100, 100, 100, 100],
            [400, 100, 400, 100],
            [400, 400, 400, 400],
            [100, 400, 100, 400],
        ]
        result = _run(create_composite(
            primary_camera_id="cam1",
            secondary_camera_id="cam2",
            correspondence_points=json.dumps(points),
        ))
        data = json.loads(result[0].text)
        assert "composite_id" in data
        assert data["num_correspondences"] == 4
        assert data["reprojection_error"] < 1.0

    def test_manual_with_translation(self):
        points = [
            [100, 100, 150, 130],
            [400, 100, 450, 130],
            [400, 400, 450, 430],
            [100, 400, 150, 430],
        ]
        result = _run(create_composite(
            primary_camera_id="cam1",
            secondary_camera_id="cam2",
            correspondence_points=json.dumps(points),
        ))
        data = json.loads(result[0].text)
        assert "composite_id" in data
        assert data["reprojection_error"] < 1.0

    def test_manual_too_few_points(self):
        points = [[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]
        result = _run(create_composite(
            primary_camera_id="cam1",
            secondary_camera_id="cam2",
            correspondence_points=json.dumps(points),
        ))
        data = json.loads(result[0].text)
        assert "error" in data

    def test_manual_invalid_json(self):
        result = _run(create_composite(
            primary_camera_id="cam1",
            secondary_camera_id="cam2",
            correspondence_points="not-json",
        ))
        data = json.loads(result[0].text)
        assert "error" in data

    def test_manual_with_playfield_id(self):
        points = [
            [100, 100, 100, 100],
            [400, 100, 400, 100],
            [400, 400, 400, 400],
            [100, 400, 100, 400],
        ]
        result = _run(create_composite(
            primary_camera_id="cam1",
            secondary_camera_id="cam2",
            playfield_id="pf_test",
            correspondence_points=json.dumps(points),
        ))
        data = json.loads(result[0].text)
        assert "composite_id" in data
        comp = composite_manager.get(data["composite_id"])
        assert comp.playfield_id == "pf_test"


# ---------------------------------------------------------------------------
# create_composite: auto-detect mode with synthetic ArUco images
# ---------------------------------------------------------------------------


class TestCreateCompositeAutoDetect:
    def test_auto_detect_shared_markers(self):
        """Both cameras see the same 4 ArUco markers at the same positions."""
        marker_ids = [0, 1, 2, 3]
        positions = [(50, 50), (450, 50), (50, 350), (450, 350)]
        img = _make_aruco_image(marker_ids, positions)

        cap1 = FakeCapture(img)
        cap2 = FakeCapture(img.copy())
        cam1_id = registry.open(cap1)
        cam2_id = registry.open(cap2)

        result = _run(create_composite(
            primary_camera_id=cam1_id,
            secondary_camera_id=cam2_id,
        ))
        data = json.loads(result[0].text)
        assert "composite_id" in data
        assert data["num_correspondences"] >= 4
        assert data["reprojection_error"] < 2.0

    def test_auto_detect_not_enough_shared(self):
        """Primary sees markers 0,1 and secondary sees 2,3 -> no shared."""
        img1 = _make_aruco_image([0, 1], [(50, 50), (450, 50)])
        img2 = _make_aruco_image([2, 3], [(50, 350), (450, 350)])

        cap1 = FakeCapture(img1)
        cap2 = FakeCapture(img2)
        cam1_id = registry.open(cap1)
        cam2_id = registry.open(cap2)

        result = _run(create_composite(
            primary_camera_id=cam1_id,
            secondary_camera_id=cam2_id,
        ))
        data = json.loads(result[0].text)
        assert "error" in data
        assert "shared" in data["error"].lower() or "Not enough" in data["error"]

    def test_auto_detect_unknown_primary(self):
        result = _run(create_composite(
            primary_camera_id="nonexistent",
            secondary_camera_id="also_nonexistent",
        ))
        data = json.loads(result[0].text)
        assert "error" in data
        assert "primary" in data["error"].lower()

    def test_auto_detect_unknown_secondary(self):
        img = _make_aruco_image([0], [(50, 50)])
        cap = FakeCapture(img)
        cam_id = registry.open(cap)

        result = _run(create_composite(
            primary_camera_id=cam_id,
            secondary_camera_id="nonexistent",
        ))
        data = json.loads(result[0].text)
        assert "error" in data
        assert "secondary" in data["error"].lower()

    def test_auto_detect_with_offset(self):
        """Secondary camera sees markers shifted by (30, 20)."""
        marker_ids = [0, 1, 2, 3]
        pos1 = [(50, 50), (450, 50), (50, 350), (450, 350)]
        pos2 = [(80, 70), (480, 70), (80, 370), (480, 370)]  # shifted

        img1 = _make_aruco_image(marker_ids, pos1)
        img2 = _make_aruco_image(marker_ids, pos2)

        cap1 = FakeCapture(img1)
        cap2 = FakeCapture(img2)
        cam1_id = registry.open(cap1)
        cam2_id = registry.open(cap2)

        result = _run(create_composite(
            primary_camera_id=cam1_id,
            secondary_camera_id=cam2_id,
        ))
        data = json.loads(result[0].text)
        assert "composite_id" in data
        assert data["reprojection_error"] < 5.0
