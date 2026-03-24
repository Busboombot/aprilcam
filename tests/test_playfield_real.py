"""Tests for the core CV pipeline using real captured playfield images.

These tests exercise ArUco corner detection, Playfield polygon locking,
deskew, AprilTag detection in deskewed output, and homography accuracy
against a reference homography.json.
"""

import json

import cv2 as cv
import numpy as np
import pytest
from pathlib import Path

from aprilcam.playfield import Playfield
from aprilcam.homography import (
    detect_aruco_4x4,
    choose_corner_point,
    calibrate_from_corners,
    FieldSpec,
    CORNER_ID_MAP,
)

TEST_DATA = Path(__file__).parent / "data"
REF_HOMOGRAPHY = Path(__file__).parent.parent / "data" / "homography.json"


@pytest.fixture
def playfield_img():
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    return cv.imread(str(path))


@pytest.fixture
def ref_homography():
    if not REF_HOMOGRAPHY.exists():
        pytest.skip("Reference homography not available")
    with open(REF_HOMOGRAPHY) as f:
        return json.load(f)


def test_aruco_corners_detected(playfield_img):
    """All 4 ArUco corner markers (IDs 0-3) are detected."""
    gray = cv.cvtColor(playfield_img, cv.COLOR_BGR2GRAY)
    dets = detect_aruco_4x4(gray)
    found_ids = {tid for _, tid in dets}
    assert {0, 1, 2, 3}.issubset(found_ids), f"Missing corners: {set(range(4)) - found_ids}"


def test_playfield_polygon_from_real_image(playfield_img):
    """Playfield.update() locks polygon from real image."""
    pf = Playfield(detect_inverted=True)
    pf.update(playfield_img)
    poly = pf.get_polygon()
    assert poly is not None
    assert poly.shape == (4, 2)


def test_deskew_from_real_image(playfield_img):
    """Deskew produces a rectangular top-down image."""
    pf = Playfield(detect_inverted=True)
    pf.update(playfield_img)
    deskewed = pf.deskew(playfield_img)
    # Deskewed should have different dimensions than original
    assert deskewed.shape[0] != playfield_img.shape[0] or deskewed.shape[1] != playfield_img.shape[1]
    # Should be a reasonable size
    assert deskewed.shape[0] > 100
    assert deskewed.shape[1] > 100


def test_apriltags_detected_after_deskew(playfield_img):
    """AprilTags are detectable in the deskewed image."""
    pf = Playfield(detect_inverted=True)
    pf.update(playfield_img)
    deskewed = pf.deskew(playfield_img)

    gray = cv.cvtColor(deskewed, cv.COLOR_BGR2GRAY)
    april_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(april_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    assert ids is not None and len(ids) > 0, "No AprilTags detected in deskewed image"
    found_ids = set(ids.flatten().tolist())
    # Should find at least some of the expected tags
    assert len(found_ids) >= 3, f"Only found {len(found_ids)} tags: {found_ids}"


def test_homography_matches_reference(playfield_img, ref_homography):
    """Computed homography is close to the reference homography.json."""
    # Detect corners
    gray = cv.cvtColor(playfield_img, cv.COLOR_BGR2GRAY)
    dets = detect_aruco_4x4(gray)

    corner_centers = {}
    for pts, tid in dets:
        if tid in CORNER_ID_MAP:
            c = choose_corner_point(pts)
            corner_centers[CORNER_ID_MAP[tid]] = (float(c[0]), float(c[1]))

    assert len(corner_centers) == 4, f"Only found {len(corner_centers)} corners"

    # Compute homography with reference dimensions
    field = FieldSpec(
        width_in=ref_homography["width_cm"],
        height_in=ref_homography["height_cm"],
        units="cm",
    )
    H, _, _ = calibrate_from_corners(corner_centers, field)

    # The homography should map corners correctly
    # Test that each corner maps to its expected world coordinate
    ref_world = np.array(ref_homography["world_points_cm"], dtype=np.float32)
    for i, name in enumerate(["upper_left", "upper_right", "lower_left", "lower_right"]):
        px = np.array([corner_centers[name][0], corner_centers[name][1], 1.0])
        mapped = H @ px
        world = mapped[:2] / mapped[2]
        np.testing.assert_allclose(world, ref_world[i], atol=1.0,
            err_msg=f"Corner {name} mapped incorrectly")


def test_both_images_detect_corners():
    """Both test images have detectable ArUco corners."""
    for img_name in ["playfield_cam3.jpg", "playfield_cam3_moved.jpg"]:
        path = TEST_DATA / img_name
        if not path.exists():
            pytest.skip(f"{img_name} not available")
        img = cv.imread(str(path))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dets = detect_aruco_4x4(gray)
        found_ids = {tid for _, tid in dets if tid in (0, 1, 2, 3)}
        # At least 3 corners should be found in each image
        assert len(found_ids) >= 3, f"{img_name}: only found corners {found_ids}"
