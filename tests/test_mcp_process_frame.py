"""Integration tests for the batch operation pipeline and process_frame tool."""

import json
import os

import pytest

from aprilcam.mcp_server import (
    create_frame_from_image,
    frame_registry,
    playfield_registry,
    process_frame,
    run_operations,
)

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "playfield_cam3.jpg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Ensure registries are empty before and after each test."""
    frame_registry.clear()
    yield
    frame_registry.clear()
    for pid in list(playfield_registry._playfields):
        playfield_registry.remove(pid)


# ---------------------------------------------------------------------------
# Helper to create a frame and return its entry
# ---------------------------------------------------------------------------


async def _make_frame():
    """Create a frame from the test image and return (frame_id, entry)."""
    result = await create_frame_from_image(TEST_IMAGE)
    data = json.loads(result[0].text)
    frame_id = data["frame_id"]
    entry = frame_registry.get(frame_id)
    return frame_id, entry


# ---------------------------------------------------------------------------
# process_frame — individual operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_frame_detect_lines():
    frame_id, _ = await _make_frame()
    result = await process_frame(frame_id, ["detect_lines"])
    data = json.loads(result[0].text)

    assert data["frame_id"] == frame_id
    assert "detect_lines" in data["results"]
    assert isinstance(data["results"]["detect_lines"], list)


@pytest.mark.asyncio
async def test_process_frame_detect_circles():
    frame_id, _ = await _make_frame()
    result = await process_frame(frame_id, ["detect_circles"])
    data = json.loads(result[0].text)

    assert data["frame_id"] == frame_id
    assert "detect_circles" in data["results"]
    assert isinstance(data["results"]["detect_circles"], list)


@pytest.mark.asyncio
async def test_process_frame_detect_contours():
    frame_id, _ = await _make_frame()
    result = await process_frame(frame_id, ["detect_contours"])
    data = json.loads(result[0].text)

    assert data["frame_id"] == frame_id
    assert "detect_contours" in data["results"]
    assert isinstance(data["results"]["detect_contours"], list)


@pytest.mark.asyncio
async def test_process_frame_detect_qr():
    frame_id, _ = await _make_frame()
    result = await process_frame(frame_id, ["detect_qr"])
    data = json.loads(result[0].text)

    assert data["frame_id"] == frame_id
    assert "detect_qr" in data["results"]
    # Test image has no QR codes; expect empty list
    assert data["results"]["detect_qr"] == []


@pytest.mark.asyncio
async def test_process_frame_detect_tags():
    frame_id, _ = await _make_frame()
    result = await process_frame(frame_id, ["detect_tags"])
    data = json.loads(result[0].text)

    assert data["frame_id"] == frame_id
    assert "detect_tags" in data["results"]
    tags = data["results"]["detect_tags"]
    assert isinstance(tags, list)
    # The test image should contain at least some AprilTags
    # (even if zero, the structure must be correct)
    for tag in tags:
        assert "id" in tag
        assert "family" in tag
        assert "center_px" in tag
        assert "corners_px" in tag
        assert "orientation_yaw" in tag


@pytest.mark.asyncio
async def test_process_frame_detect_aruco():
    frame_id, _ = await _make_frame()
    result = await process_frame(frame_id, ["detect_aruco"])
    data = json.loads(result[0].text)

    assert data["frame_id"] == frame_id
    assert "detect_aruco" in data["results"]
    markers = data["results"]["detect_aruco"]
    assert isinstance(markers, list)
    # The test image contains ArUco 4x4 markers (IDs 0-3)
    assert len(markers) >= 1
    found_ids = {m["id"] for m in markers}
    # At least some of the corner markers should be found
    assert len(found_ids & {0, 1, 2, 3}) >= 1
    for marker in markers:
        assert "id" in marker
        assert "center_px" in marker
        assert "corners_px" in marker


# ---------------------------------------------------------------------------
# process_frame — batch and error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_frame_batch():
    frame_id, _ = await _make_frame()
    result = await process_frame(
        frame_id, ["detect_aruco", "detect_lines", "detect_contours"]
    )
    data = json.loads(result[0].text)

    assert data["frame_id"] == frame_id
    assert "detect_aruco" in data["results"]
    assert "detect_lines" in data["results"]
    assert "detect_contours" in data["results"]


@pytest.mark.asyncio
async def test_process_frame_unknown_op():
    frame_id, _ = await _make_frame()
    result = await process_frame(frame_id, ["detect_lines", "bogus_op"])
    data = json.loads(result[0].text)

    assert "error" in data
    assert "bogus_op" in data["error"]


@pytest.mark.asyncio
async def test_process_frame_missing_frame():
    result = await process_frame("frm_999", ["detect_lines"])
    data = json.loads(result[0].text)
    assert "error" in data
    assert "frm_999" in data["error"]


# ---------------------------------------------------------------------------
# create_frame_from_image with operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_frame_from_image_with_operations():
    result = await create_frame_from_image(
        TEST_IMAGE, operations=["detect_aruco", "detect_lines"]
    )
    data = json.loads(result[0].text)

    assert "frame_id" in data
    assert "source" in data
    assert "results" in data
    assert "detect_aruco" in data["results"]
    assert "detect_lines" in data["results"]


@pytest.mark.asyncio
async def test_create_frame_from_image_with_unknown_op():
    result = await create_frame_from_image(
        TEST_IMAGE, operations=["detect_lines", "no_such_op"]
    )
    data = json.loads(result[0].text)

    # Frame should still be created, but error reported
    assert "frame_id" in data
    assert "error" in data
    assert "no_such_op" in data["error"]


# ---------------------------------------------------------------------------
# operations_applied tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_operations_applied_tracking():
    frame_id, entry = await _make_frame()
    assert entry.operations_applied == []

    await process_frame(frame_id, ["detect_lines"])
    assert entry.operations_applied == ["detect_lines"]

    await process_frame(frame_id, ["detect_circles", "detect_contours"])
    assert entry.operations_applied == [
        "detect_lines",
        "detect_circles",
        "detect_contours",
    ]


# ---------------------------------------------------------------------------
# results stored on entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_results_stored_on_entry():
    frame_id, entry = await _make_frame()
    await process_frame(frame_id, ["detect_aruco", "detect_qr"])

    assert "detect_aruco" in entry.results
    assert "detect_qr" in entry.results
    # aruco_corners dict should be populated
    assert entry.aruco_corners is not None
    assert isinstance(entry.aruco_corners, dict)


# ---------------------------------------------------------------------------
# run_operations unit test (direct call, no MCP layer)
# ---------------------------------------------------------------------------


def test_run_operations_unknown_raises():
    import cv2

    img = cv2.imread(TEST_IMAGE)
    entry = frame_registry.add(raw=img, source="test")
    with pytest.raises(ValueError, match="Unknown operation"):
        run_operations(entry, ["not_real"])


def test_run_operations_direct():
    import cv2

    img = cv2.imread(TEST_IMAGE)
    entry = frame_registry.add(raw=img, source="test")
    results = run_operations(entry, ["detect_lines", "detect_circles"])

    assert "detect_lines" in results
    assert "detect_circles" in results
    assert entry.operations_applied == ["detect_lines", "detect_circles"]
