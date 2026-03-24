"""Integration tests for multi-camera compositing.

These tests exercise the full round-trip through the MCP tool layer:
auto-detect shared markers, query composite tags, retrieve composite
frames, and verify that individual cameras remain usable after
composite creation.
"""

import asyncio
import json

import cv2
import numpy as np
import pytest

from aprilcam.mcp_server import (
    capture_frame,
    composite_manager,
    create_composite,
    get_composite_frame,
    get_composite_tags,
    registry,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_aruco_image(marker_positions, size=(640, 480)):
    """Create a synthetic BGR image with ArUco 4x4 markers at given center positions.

    Args:
        marker_positions: dict mapping marker_id -> (cx, cy) center positions.
        size: (width, height) of the output image.

    Returns:
        BGR image (numpy array) with markers drawn.
    """
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8) + 128
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 40
    for mid, (cx, cy) in marker_positions.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        x1 = cx - marker_size // 2
        y1 = cy - marker_size // 2
        x2, y2 = x1 + marker_size, y1 + marker_size
        if x1 >= 0 and y1 >= 0 and x2 <= size[0] and y2 <= size[1]:
            img[y1:y2, x1:x2] = marker_bgr
    return img


def _add_apriltag_markers(img, tag_positions, tag_size=60):
    """Draw AprilTag 36h11 markers onto an existing image.

    Args:
        img: BGR image to draw on (modified in place).
        tag_positions: dict mapping tag_id -> (cx, cy) center positions.
        tag_size: pixel size of each marker.

    Returns:
        The modified image.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tid, (cx, cy) in tag_positions.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, tid, tag_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        x1 = cx - tag_size // 2
        y1 = cy - tag_size // 2
        x2, y2 = x1 + tag_size, y1 + tag_size
        h, w = img.shape[:2]
        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
            img[y1:y2, x1:x2] = marker_bgr
    return img


class FakeCapture:
    """Mock camera that returns a pre-built image."""

    def __init__(self, image: np.ndarray):
        self._img = image

    def read(self):
        return True, self._img.copy()

    def release(self):
        pass


@pytest.fixture(autouse=True)
def _clean_registries():
    """Reset composite manager and camera registry between tests."""
    composite_manager._composites.clear()
    yield
    composite_manager._composites.clear()
    registry._cameras.clear()


# ---------------------------------------------------------------------------
# Full auto-detect round-trip
# ---------------------------------------------------------------------------


class TestAutoDetectRoundTrip:
    """Two FakeCaptures with shared ArUco markers at different positions.

    Verifies the complete flow: create_composite (auto) -> get_composite_tags
    -> get_composite_frame.
    """

    def _register_cameras(self):
        """Create two cameras with shared ArUco 4x4 markers at different positions.

        The secondary camera also has AprilTag 36h11 markers so that
        get_composite_tags (which detects 36h11) can find them.

        Returns (cam1_id, cam2_id, apriltag_ids).
        """
        # ArUco 4x4 markers for auto-detect correspondence (corners)
        aruco_primary = {
            0: (100, 100),
            1: (500, 100),
            2: (500, 350),
            3: (100, 350),
        }
        # Same ArUco IDs shifted by (40, 30) on secondary
        aruco_secondary = {
            0: (140, 130),
            1: (540, 130),
            2: (540, 380),
            3: (140, 380),
        }
        # AprilTag 36h11 markers on secondary only (for tag detection)
        apriltag_positions = {
            10: (300, 240),
        }

        img1 = _make_aruco_image(aruco_primary)
        img2 = _make_aruco_image(aruco_secondary)
        # Add AprilTag 36h11 markers to the secondary image
        _add_apriltag_markers(img2, apriltag_positions)

        cam1_id = registry.open(FakeCapture(img1))
        cam2_id = registry.open(FakeCapture(img2))
        return cam1_id, cam2_id, set(apriltag_positions.keys())

    def test_auto_detect_then_get_tags(self):
        """Auto-detect composite, then query tags and verify mapped positions."""
        cam1_id, cam2_id, apriltag_ids = self._register_cameras()

        # Step 1: create composite via auto-detect
        result = _run(create_composite(
            primary_camera_id=cam1_id,
            secondary_camera_id=cam2_id,
        ))
        data = json.loads(result[0].text)
        assert "composite_id" in data, f"Expected composite_id, got: {data}"
        assert data["num_correspondences"] >= 4
        assert data["reprojection_error"] < 5.0
        comp_id = data["composite_id"]

        # Step 2: get composite tags
        result = _run(get_composite_tags(composite_id=comp_id))
        data = json.loads(result[0].text)
        assert data["composite_id"] == comp_id
        assert isinstance(data["tags"], list)

        # Tags should be detected (the secondary camera has AprilTag 36h11 markers)
        detected_ids = {t["id"] for t in data["tags"]}
        # At least one AprilTag 36h11 marker should be detected
        assert len(detected_ids) > 0, "Expected at least one tag detection"
        assert detected_ids.issubset(apriltag_ids), (
            f"Detected IDs {detected_ids} should be a subset of {apriltag_ids}"
        )

        # Each tag should have the expected fields
        for tag in data["tags"]:
            assert "id" in tag
            assert "center_px" in tag
            assert len(tag["center_px"]) == 2
            assert "corners_px" in tag
            assert len(tag["corners_px"]) == 4
            assert "orientation_yaw" in tag

    def test_auto_detect_then_get_frame(self):
        """Auto-detect composite, then retrieve a composite frame image."""
        cam1_id, cam2_id, _ = self._register_cameras()

        # Create composite
        result = _run(create_composite(
            primary_camera_id=cam1_id,
            secondary_camera_id=cam2_id,
        ))
        data = json.loads(result[0].text)
        comp_id = data["composite_id"]

        # Get frame as base64
        result = _run(get_composite_frame(composite_id=comp_id))
        assert len(result) == 1
        assert result[0].type == "image"
        assert result[0].mimeType == "image/jpeg"
        assert len(result[0].data) > 100  # non-trivial image data

    def test_auto_detect_then_get_frame_as_file(self):
        """Auto-detect composite, then retrieve frame as a file path."""
        cam1_id, cam2_id, _ = self._register_cameras()

        result = _run(create_composite(
            primary_camera_id=cam1_id,
            secondary_camera_id=cam2_id,
        ))
        comp_id = json.loads(result[0].text)["composite_id"]

        result = _run(get_composite_frame(composite_id=comp_id, format="file"))
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "path" in data
        assert data["path"].endswith(".jpg")


# ---------------------------------------------------------------------------
# Individual cameras still work after composite creation
# ---------------------------------------------------------------------------


class TestCamerasWorkAfterComposite:
    """Creating a composite must not break individual camera operations."""

    def test_capture_frame_on_both_cameras_after_composite(self):
        """After creating a composite, capture_frame on each camera individually."""
        # Set up two cameras with shared markers
        pos1 = {0: (100, 100), 1: (500, 100), 2: (500, 350), 3: (100, 350)}
        pos2 = {0: (130, 120), 1: (530, 120), 2: (530, 370), 3: (130, 370)}
        img1 = _make_aruco_image(pos1)
        img2 = _make_aruco_image(pos2)

        cam1_id = registry.open(FakeCapture(img1))
        cam2_id = registry.open(FakeCapture(img2))

        # Create composite
        result = _run(create_composite(
            primary_camera_id=cam1_id,
            secondary_camera_id=cam2_id,
        ))
        data = json.loads(result[0].text)
        assert "composite_id" in data

        # Capture from primary camera
        result = _run(capture_frame(camera_id=cam1_id))
        assert len(result) == 1
        assert result[0].type == "image"

        # Capture from secondary camera
        result = _run(capture_frame(camera_id=cam2_id))
        assert len(result) == 1
        assert result[0].type == "image"


# ---------------------------------------------------------------------------
# Error cases with invalid composite_id
# ---------------------------------------------------------------------------


class TestCompositeErrorCases:
    """Error handling for invalid composite IDs."""

    def test_get_composite_tags_invalid_id(self):
        result = _run(get_composite_tags(composite_id="invalid-id-12345"))
        data = json.loads(result[0].text)
        assert "error" in data

    def test_get_composite_frame_invalid_id(self):
        result = _run(get_composite_frame(composite_id="invalid-id-12345"))
        data = json.loads(result[0].text)
        assert "error" in data
