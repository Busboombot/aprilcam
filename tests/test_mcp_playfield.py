"""Tests for PlayfieldRegistry and create_playfield MCP tool."""

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aprilcam.mcp_server import (
    PlayfieldEntry,
    PlayfieldRegistry,
    calibrate_playfield,
    create_playfield,
    playfield_registry,
    registry,
)
from aprilcam.playfield import Playfield

TEST_DATA = Path(__file__).parent / "data"


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


class FakeCapture:
    """Mock camera that returns a pre-loaded image."""

    def __init__(self, image_path: str):
        self._img = cv2.imread(str(image_path))
        assert self._img is not None, f"Failed to load {image_path}"

    def read(self):
        return True, self._img.copy()

    def release(self):
        pass


@pytest.fixture(autouse=True)
def clean_playfield_registry():
    """Reset playfield registry before/after each test."""
    playfield_registry._playfields.clear()
    yield
    playfield_registry._playfields.clear()


# ---------------------------------------------------------------------------
# PlayfieldRegistry unit tests
# ---------------------------------------------------------------------------


class TestPlayfieldRegistry:
    def test_register_and_get(self):
        pr = PlayfieldRegistry()
        pf = Playfield()
        entry = PlayfieldEntry(
            playfield_id="pf_test", camera_id="test_cam", playfield=pf
        )
        pr.register(entry)
        assert pr.get("pf_test") is entry

    def test_list(self):
        pr = PlayfieldRegistry()
        pf = Playfield()
        entry = PlayfieldEntry(
            playfield_id="pf_test", camera_id="test_cam", playfield=pf
        )
        pr.register(entry)
        assert "pf_test" in pr.list()

    def test_find_by_camera(self):
        pr = PlayfieldRegistry()
        pf = Playfield()
        entry = PlayfieldEntry(
            playfield_id="pf_test", camera_id="test_cam", playfield=pf
        )
        pr.register(entry)
        assert pr.find_by_camera("test_cam") == "pf_test"
        assert pr.find_by_camera("nonexistent") is None

    def test_remove(self):
        pr = PlayfieldRegistry()
        pf = Playfield()
        entry = PlayfieldEntry(
            playfield_id="pf_test", camera_id="test_cam", playfield=pf
        )
        pr.register(entry)
        pr.remove("pf_test")
        assert len(pr.list()) == 0
        with pytest.raises(KeyError):
            pr.get("pf_test")

    def test_remove_nonexistent_raises_keyerror(self):
        pr = PlayfieldRegistry()
        with pytest.raises(KeyError):
            pr.remove("nonexistent")


# ---------------------------------------------------------------------------
# create_playfield tool tests
# ---------------------------------------------------------------------------


class TestCreatePlayfield:
    def test_unknown_camera(self):
        result = _run(create_playfield(camera_id="nonexistent"))
        data = json.loads(result[0].text)
        assert "error" in data
        assert "nonexistent" in data["error"]

    def test_missing_markers_with_blank_image(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        class BlankCap:
            def read(self):
                return True, blank.copy()

            def release(self):
                pass

        cam_id = registry.open(BlankCap())
        try:
            result = _run(create_playfield(camera_id=cam_id, max_frames=3))
            data = json.loads(result[0].text)
            assert "error" in data
            assert "missing_corner_ids" in data
        finally:
            try:
                registry.close(cam_id)
            except KeyError:
                pass

    def test_success_with_real_image(self):
        img_path = TEST_DATA / "playfield_cam3_moved.jpg"
        if not img_path.exists():
            pytest.skip("Test image not available")

        fake_cap = FakeCapture(str(img_path))
        cam_id = registry.open(fake_cap)

        try:
            result = _run(create_playfield(camera_id=cam_id))
            data = json.loads(result[0].text)

            assert "playfield_id" in data
            assert data["playfield_id"] == f"pf_{cam_id}"
            assert "corners" in data
            assert len(data["corners"]) == 4  # UL, UR, LR, LL
            assert data["calibrated"] is False
        finally:
            try:
                registry.close(cam_id)
            except KeyError:
                pass

    def test_replaces_existing_for_same_camera(self):
        img_path = TEST_DATA / "playfield_cam3_moved.jpg"
        if not img_path.exists():
            pytest.skip("Test image not available")

        fake_cap = FakeCapture(str(img_path))
        cam_id = registry.open(fake_cap)

        try:
            _run(create_playfield(camera_id=cam_id))
            _run(create_playfield(camera_id=cam_id))
            # Should only have one playfield entry
            assert len(playfield_registry.list()) == 1
        finally:
            try:
                registry.close(cam_id)
            except KeyError:
                pass


# ---------------------------------------------------------------------------
# calibrate_playfield tool tests
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_registries():
    """Reset both registries before/after each test."""
    playfield_registry._playfields.clear()
    registry._cameras.clear()
    yield
    playfield_registry._playfields.clear()
    registry._cameras.clear()


@pytest.mark.asyncio
async def test_calibrate_playfield_success(clean_registries):
    """Calibrate a playfield with real measurements."""
    img_path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not img_path.exists():
        pytest.skip("Test image not available")

    fake_cap = FakeCapture(str(img_path))
    cam_id = registry.open(fake_cap)

    try:
        # First create the playfield
        await create_playfield(camera_id=cam_id)
        pf_id = f"pf_{cam_id}"

        # Now calibrate
        result = await calibrate_playfield(
            playfield_id=pf_id, width=40.0, height=35.0, units="inch"
        )
        data = json.loads(result[0].text)

        assert data["calibrated"] is True
        assert abs(data["width_cm"] - 101.6) < 0.1
        assert abs(data["height_cm"] - 88.9) < 0.1
    finally:
        try:
            registry.close(cam_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_calibrate_playfield_unknown_id(clean_registries):
    result = await calibrate_playfield(
        playfield_id="nonexistent", width=40.0, height=35.0
    )
    data = json.loads(result[0].text)
    assert "error" in data


@pytest.mark.asyncio
async def test_calibrate_playfield_overwrites(clean_registries):
    """Re-calibration overwrites previous calibration."""
    img_path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not img_path.exists():
        pytest.skip("Test image not available")

    fake_cap = FakeCapture(str(img_path))
    cam_id = registry.open(fake_cap)

    try:
        await create_playfield(camera_id=cam_id)
        pf_id = f"pf_{cam_id}"

        # Calibrate with inches
        await calibrate_playfield(
            playfield_id=pf_id, width=40.0, height=35.0, units="inch"
        )

        # Re-calibrate with cm
        result = await calibrate_playfield(
            playfield_id=pf_id, width=100.0, height=80.0, units="cm"
        )
        data = json.loads(result[0].text)
        assert data["width_cm"] == 100.0
        assert data["height_cm"] == 80.0
    finally:
        try:
            registry.close(cam_id)
        except Exception:
            pass
