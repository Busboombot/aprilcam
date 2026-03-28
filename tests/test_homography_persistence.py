"""Tests for per-camera homography file persistence."""

from __future__ import annotations

import json
from pathlib import Path

from aprilcam.camutil import camera_slug
from aprilcam.homography import discover_homography, homography_path


# -- camera_slug tests -------------------------------------------------------


def test_camera_slug_basic():
    assert camera_slug("Brio 501", 1920, 1080) == "brio-501-1920x1080"
    assert camera_slug("HD USB CAMERA", 640, 480) == "hd-usb-camera-640x480"
    assert camera_slug("FaceTime HD Camera", 1280, 720) == "facetime-hd-camera-1280x720"


def test_camera_slug_special_chars():
    assert camera_slug("My--Camera!!v2", 800, 600) == "my-camera-v2-800x600"
    assert camera_slug("  spaces  ", 320, 240) == "spaces-320x240"
    assert camera_slug("cam/dev:0", 1920, 1080) == "cam-dev-0-1920x1080"
    assert camera_slug("---leading-trailing---", 100, 100) == "leading-trailing-100x100"


# -- homography_path tests ---------------------------------------------------


def test_homography_path():
    p = homography_path("brio-501-1920x1080")
    assert p == Path("data/homography-brio-501-1920x1080.json")

    p2 = homography_path("cam-0-640x480", data_dir="/tmp/mydata")
    assert p2 == Path("/tmp/mydata/homography-cam-0-640x480.json")


# -- discover_homography tests -----------------------------------------------


def test_discover_homography_per_camera(tmp_path: Path):
    slug = camera_slug("Brio 501", 1920, 1080)
    per_cam = tmp_path / f"homography-{slug}.json"
    per_cam.write_text(json.dumps({"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))

    result = discover_homography("Brio 501", 1920, 1080, data_dir=tmp_path)
    assert result == per_cam


def test_discover_homography_fallback(tmp_path: Path):
    global_file = tmp_path / "homography.json"
    global_file.write_text(json.dumps({"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))

    result = discover_homography("Unknown Cam", 640, 480, data_dir=tmp_path)
    assert result == global_file


def test_discover_homography_none(tmp_path: Path):
    result = discover_homography("Nonexistent", 320, 240, data_dir=tmp_path)
    assert result is None
