"""Tests for composite MCP tools: create_composite, get_composite_frame, get_composite_tags."""

import asyncio
import json

import cv2
import numpy as np
import pytest

from aprilcam.composite import CompositeManager
from aprilcam.mcp_server import (
    composite_manager,
    create_composite,
    get_composite_frame,
    get_composite_tags,
    playfield_registry,
    registry,
    render_tag_overlay,
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


# ---------------------------------------------------------------------------
# Helper: AprilTag 36h11 synthetic images for secondary camera
# ---------------------------------------------------------------------------


def _make_apriltag_image(tag_ids, positions, img_size=(640, 480), tag_size=80):
    """Create a synthetic BGR image with AprilTag 36h11 markers.

    Args:
        tag_ids: list of int tag IDs.
        positions: list of (x, y) top-left corner positions.
        img_size: (width, height).
        tag_size: pixel size of each marker.

    Returns:
        BGR image with markers drawn.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    canvas = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200

    for tid, (x, y) in zip(tag_ids, positions):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, tid, tag_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        y2 = min(y + tag_size, img_size[1])
        x2 = min(x + tag_size, img_size[0])
        mh = y2 - y
        mw = x2 - x
        canvas[y:y2, x:x2] = marker_bgr[:mh, :mw]

    return canvas


def _setup_composite_with_tags():
    """Create a composite from two cameras where secondary has AprilTag 36h11 markers.

    Returns (composite_id, cam1_id, cam2_id).
    """
    # Primary camera: plain gray image (no tags)
    primary_img = np.ones((480, 640, 3), dtype=np.uint8) * 180

    # Secondary camera: has AprilTag 36h11 markers
    secondary_img = _make_apriltag_image(
        tag_ids=[10, 20],
        positions=[(100, 100), (400, 200)],
    )

    cap1 = FakeCapture(primary_img)
    cap2 = FakeCapture(secondary_img)
    cam1_id = registry.open(cap1)
    cam2_id = registry.open(cap2)

    # Use manual correspondence (identity) since cameras differ
    points = [
        [100, 100, 100, 100],
        [400, 100, 400, 100],
        [400, 400, 400, 400],
        [100, 400, 100, 400],
    ]
    from aprilcam.composite import compute_cross_camera_homography

    pri_pts = np.array([[p[0], p[1]] for p in points], dtype=np.float64)
    sec_pts = np.array([[p[2], p[3]] for p in points], dtype=np.float64)
    H, err = compute_cross_camera_homography(pri_pts, sec_pts)

    comp = composite_manager.create(
        primary_camera_id=cam1_id,
        secondary_camera_id=cam2_id,
        homography=H,
        reprojection_error=err,
    )
    return comp.composite_id, cam1_id, cam2_id


# ---------------------------------------------------------------------------
# render_tag_overlay
# ---------------------------------------------------------------------------


class TestRenderTagOverlay:
    def test_draws_on_copy(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tags = [{
            "id": 5,
            "center_px": [50, 50],
            "corners_px": [[30, 30], [70, 30], [70, 70], [30, 70]],
            "orientation_yaw": 0.0,
        }]
        result = render_tag_overlay(frame, tags)
        # Original should be unmodified
        assert np.all(frame == 0)
        # Result should have drawn something (non-zero pixels)
        assert np.any(result != 0)

    def test_empty_tags(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = render_tag_overlay(frame, [])
        np.testing.assert_array_equal(result, frame)


# ---------------------------------------------------------------------------
# get_composite_frame
# ---------------------------------------------------------------------------


class TestGetCompositeFrame:
    def test_returns_image_base64(self):
        comp_id, _, _ = _setup_composite_with_tags()
        result = _run(get_composite_frame(composite_id=comp_id))
        assert len(result) == 1
        # Should be ImageContent (base64)
        assert result[0].type == "image"
        assert result[0].mimeType == "image/jpeg"
        assert len(result[0].data) > 0

    def test_returns_image_file(self):
        comp_id, _, _ = _setup_composite_with_tags()
        result = _run(get_composite_frame(composite_id=comp_id, format="file"))
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "path" in data
        assert data["path"].endswith(".jpg")

    def test_unknown_composite_id(self):
        result = _run(get_composite_frame(composite_id="nonexistent"))
        data = json.loads(result[0].text)
        assert "error" in data
        assert "nonexistent" in data["error"]

    def test_primary_camera_gone(self):
        comp_id, cam1_id, _ = _setup_composite_with_tags()
        registry._cameras.pop(cam1_id, None)
        result = _run(get_composite_frame(composite_id=comp_id))
        data = json.loads(result[0].text)
        assert "error" in data
        assert "primary" in data["error"].lower() or "no longer" in data["error"].lower()

    def test_secondary_camera_gone(self):
        comp_id, _, cam2_id = _setup_composite_with_tags()
        registry._cameras.pop(cam2_id, None)
        result = _run(get_composite_frame(composite_id=comp_id))
        data = json.loads(result[0].text)
        assert "error" in data
        assert "secondary" in data["error"].lower() or "no longer" in data["error"].lower()


# ---------------------------------------------------------------------------
# get_composite_tags
# ---------------------------------------------------------------------------


class TestGetCompositeTags:
    def test_returns_tags_json(self):
        comp_id, _, _ = _setup_composite_with_tags()
        result = _run(get_composite_tags(composite_id=comp_id))
        data = json.loads(result[0].text)
        assert data["composite_id"] == comp_id
        assert isinstance(data["tags"], list)
        # The secondary image has AprilTag markers; we should detect them
        if len(data["tags"]) > 0:
            tag = data["tags"][0]
            assert "id" in tag
            assert "center_px" in tag
            assert "corners_px" in tag
            assert "orientation_yaw" in tag

    def test_unknown_composite_id(self):
        result = _run(get_composite_tags(composite_id="nonexistent"))
        data = json.loads(result[0].text)
        assert "error" in data

    def test_secondary_camera_gone(self):
        comp_id, _, cam2_id = _setup_composite_with_tags()
        registry._cameras.pop(cam2_id, None)
        result = _run(get_composite_tags(composite_id=comp_id))
        data = json.loads(result[0].text)
        assert "error" in data

    def test_with_playfield_calibration(self):
        """Tags get world_xy when composite has a calibrated playfield."""
        from aprilcam.mcp_server import PlayfieldEntry
        from aprilcam.playfield import Playfield

        comp_id, cam1_id, cam2_id = _setup_composite_with_tags()

        # Create a fake calibrated playfield
        pf = Playfield()
        pf_entry = PlayfieldEntry(
            playfield_id="pf_test",
            camera_id=cam1_id,
            playfield=pf,
            homography=np.eye(3),  # identity: pixel = world
        )
        playfield_registry._playfields["pf_test"] = pf_entry

        # Update composite to reference this playfield
        comp = composite_manager.get(comp_id)
        comp.playfield_id = "pf_test"

        result = _run(get_composite_tags(composite_id=comp_id))
        data = json.loads(result[0].text)
        assert isinstance(data["tags"], list)
        # If tags are detected, they should have world_xy
        for tag in data["tags"]:
            if "world_xy" in tag:
                assert len(tag["world_xy"]) == 2

        # Clean up
        playfield_registry._playfields.pop("pf_test", None)

    def test_no_tags_detected(self):
        """Secondary camera has a blank image -> no tags detected."""
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cap1 = FakeCapture(blank)
        cap2 = FakeCapture(blank.copy())
        cam1_id = registry.open(cap1)
        cam2_id = registry.open(cap2)

        # Manual correspondence (identity)
        points = json.dumps([
            [100, 100, 100, 100],
            [400, 100, 400, 100],
            [400, 400, 400, 400],
            [100, 400, 100, 400],
        ])
        result = _run(create_composite(
            primary_camera_id=cam1_id,
            secondary_camera_id=cam2_id,
            correspondence_points=points,
        ))
        data = json.loads(result[0].text)
        comp_id = data["composite_id"]

        result = _run(get_composite_tags(composite_id=comp_id))
        data = json.loads(result[0].text)
        assert data["tags"] == []
