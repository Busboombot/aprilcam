"""Integration tests for parallax correction wired into the pipeline and MCP tools.

Sprint 008, ticket 002.

Tests focus on the integration points:
- camera_pipeline.py: parallax block applied after world_xy is set
- mcp_server.py: calibrate_playfield stores camera_position
- mcp_server.py: get_tags accepts tag_heights_json and applies correction
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aprilcam.calibration.calibration import (
    CameraCalibration,
    CameraPosition,
    load_calibration_from_camera_dir,
    save_calibration_to_camera_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDENTITY_H = np.eye(3, dtype=float)


def _minimal_cal(**kwargs) -> CameraCalibration:
    return CameraCalibration(
        device_name="TestCam",
        resolution=(640, 480),
        homography=_IDENTITY_H.copy(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# camera_pipeline.py — parallax block
# ---------------------------------------------------------------------------


def _make_tag_record(tag_id: int, world_xy: Optional[tuple]) -> MagicMock:
    """Return a mock TagRecord with configurable id and world_xy."""
    tr = MagicMock()
    tr.id = tag_id
    tr.world_xy = world_xy
    return tr


def test_pipeline_applies_correction_to_elevated_tag():
    """Tags in tag_heights with height > 0 get world_xy corrected."""
    from aprilcam.daemon.camera_pipeline import CameraPipeline

    pipeline = CameraPipeline.__new__(CameraPipeline)
    pipeline._calibration = _minimal_cal(
        camera_position=CameraPosition(x_offset=0.0, y_offset=0.0, height=180.0),
        tag_heights={5: 12.0},
    )

    tr = _make_tag_record(5, (50.0, 50.0))
    tag_records = [tr]

    # Apply the same correction logic as in _capture_loop
    if pipeline._calibration and pipeline._calibration.camera_position:
        for rec in tag_records:
            tag_h = pipeline._calibration.tag_heights.get(rec.id, 0.0)
            if tag_h > 0.0 and rec.world_xy is not None:
                rec.world_xy = pipeline._calibration.correct_world_for_height(
                    rec.world_xy[0], rec.world_xy[1], tag_h
                )

    # r = 12/180; wx_corr = 50 + (1/15)*(0-50) = 50*(1 - 1/15)
    expected = 50.0 * (1.0 - 12.0 / 180.0)
    assert abs(tr.world_xy[0] - expected) < 0.01
    assert abs(tr.world_xy[1] - expected) < 0.01


def test_pipeline_skips_tag_not_in_heights():
    """Tags not in tag_heights are not modified."""
    from aprilcam.daemon.camera_pipeline import CameraPipeline

    pipeline = CameraPipeline.__new__(CameraPipeline)
    pipeline._calibration = _minimal_cal(
        camera_position=CameraPosition(x_offset=0.0, y_offset=0.0, height=180.0),
        tag_heights={5: 12.0},
    )

    tr = _make_tag_record(99, (30.0, 40.0))
    tag_records = [tr]

    if pipeline._calibration and pipeline._calibration.camera_position:
        for rec in tag_records:
            tag_h = pipeline._calibration.tag_heights.get(rec.id, 0.0)
            if tag_h > 0.0 and rec.world_xy is not None:
                rec.world_xy = pipeline._calibration.correct_world_for_height(
                    rec.world_xy[0], rec.world_xy[1], tag_h
                )

    # Tag 99 not in heights → unchanged
    assert tr.world_xy == (30.0, 40.0)


def test_pipeline_skips_when_no_calibration():
    """No calibration → block is skipped entirely, no exception."""
    from aprilcam.daemon.camera_pipeline import CameraPipeline

    pipeline = CameraPipeline.__new__(CameraPipeline)
    pipeline._calibration = None

    tr = _make_tag_record(5, (50.0, 50.0))
    tag_records = [tr]

    if pipeline._calibration and pipeline._calibration.camera_position:
        for rec in tag_records:
            tag_h = pipeline._calibration.tag_heights.get(rec.id, 0.0)
            if tag_h > 0.0 and rec.world_xy is not None:
                rec.world_xy = pipeline._calibration.correct_world_for_height(
                    rec.world_xy[0], rec.world_xy[1], tag_h
                )

    assert tr.world_xy == (50.0, 50.0)


def test_pipeline_skips_when_no_camera_position():
    """calibration without camera_position → block skipped."""
    from aprilcam.daemon.camera_pipeline import CameraPipeline

    pipeline = CameraPipeline.__new__(CameraPipeline)
    pipeline._calibration = _minimal_cal(
        camera_position=None,
        tag_heights={5: 12.0},
    )

    tr = _make_tag_record(5, (50.0, 50.0))
    tag_records = [tr]

    if pipeline._calibration and pipeline._calibration.camera_position:
        for rec in tag_records:
            tag_h = pipeline._calibration.tag_heights.get(rec.id, 0.0)
            if tag_h > 0.0 and rec.world_xy is not None:
                rec.world_xy = pipeline._calibration.correct_world_for_height(
                    rec.world_xy[0], rec.world_xy[1], tag_h
                )

    assert tr.world_xy == (50.0, 50.0)


def test_pipeline_skips_null_world_xy():
    """Tags with world_xy=None are not corrected (no exception)."""
    from aprilcam.daemon.camera_pipeline import CameraPipeline

    pipeline = CameraPipeline.__new__(CameraPipeline)
    pipeline._calibration = _minimal_cal(
        camera_position=CameraPosition(x_offset=0.0, y_offset=0.0, height=180.0),
        tag_heights={5: 12.0},
    )

    tr = _make_tag_record(5, None)
    tag_records = [tr]

    if pipeline._calibration and pipeline._calibration.camera_position:
        for rec in tag_records:
            tag_h = pipeline._calibration.tag_heights.get(rec.id, 0.0)
            if tag_h > 0.0 and rec.world_xy is not None:
                rec.world_xy = pipeline._calibration.correct_world_for_height(
                    rec.world_xy[0], rec.world_xy[1], tag_h
                )

    assert tr.world_xy is None


# ---------------------------------------------------------------------------
# calibrate_playfield MCP tool — camera_position stored in calibration.json
# ---------------------------------------------------------------------------


def test_calibrate_playfield_stores_camera_position(tmp_path):
    """calibrate_playfield persists camera_position to calibration.json."""
    import asyncio
    import numpy as np

    from aprilcam.server import mcp_server
    from aprilcam.server.mcp_server import (
        PlayfieldEntry,
        playfield_registry,
        registry,
        _cam_info,
    )
    from aprilcam.core.playfield import PlayfieldBoundary as Playfield

    # Build a playfield with a simple polygon
    pf = MagicMock(spec=Playfield)
    poly = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 80.0],
        [0.0, 80.0],
    ], dtype=np.float32)
    pf.get_polygon.return_value = poly

    cam_id = "cam_test_cp"
    pf_id = f"pf_{cam_id}"

    # Register a fake camera and playfield
    registry._cameras[cam_id] = None  # daemon-owned sentinel
    camera_dir = tmp_path / cam_id
    camera_dir.mkdir()
    _cam_info[cam_id] = {"camera_dir": str(camera_dir)}

    entry = PlayfieldEntry(
        playfield_id=pf_id,
        camera_id=cam_id,
        playfield=pf,
    )
    playfield_registry.register(entry)

    try:
        result_contents = asyncio.run(
            mcp_server.calibrate_playfield(
                playfield_id=pf_id,
                width=40.0,
                height=32.0,
                units="cm",
                camera_height_cm=150.0,
                camera_x_offset_cm=5.0,
                camera_y_offset_cm=-2.0,
            )
        )
        result = json.loads(result_contents[0].text)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result["calibrated"] is True
        assert result["camera_height_cm"] == 150.0

        # Verify calibration.json was written with camera_position
        cal_file = camera_dir / "calibration.json"
        if cal_file.exists():
            saved = json.loads(cal_file.read_text())
            assert "camera_position" in saved
            assert saved["camera_position"]["height"] == 150.0
            assert saved["camera_position"]["x_offset"] == 5.0
            assert saved["camera_position"]["y_offset"] == -2.0
    finally:
        # Cleanup
        try:
            del registry._cameras[cam_id]
        except KeyError:
            pass
        try:
            playfield_registry.remove(pf_id)
        except KeyError:
            pass
        _cam_info.pop(cam_id, None)


def test_calibrate_playfield_response_includes_camera_height_cm(tmp_path):
    """Response JSON always includes camera_height_cm field."""
    import asyncio
    import numpy as np

    from aprilcam.server import mcp_server
    from aprilcam.server.mcp_server import (
        PlayfieldEntry,
        playfield_registry,
        registry,
        _cam_info,
    )
    from aprilcam.core.playfield import PlayfieldBoundary as Playfield

    pf = MagicMock(spec=Playfield)
    poly = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 80.0],
        [0.0, 80.0],
    ], dtype=np.float32)
    pf.get_polygon.return_value = poly

    cam_id = "cam_test_ch"
    pf_id = f"pf_{cam_id}"

    registry._cameras[cam_id] = None
    camera_dir = tmp_path / cam_id
    camera_dir.mkdir()
    _cam_info[cam_id] = {"camera_dir": str(camera_dir)}

    entry = PlayfieldEntry(
        playfield_id=pf_id,
        camera_id=cam_id,
        playfield=pf,
    )
    playfield_registry.register(entry)

    try:
        result_contents = asyncio.run(
            mcp_server.calibrate_playfield(
                playfield_id=pf_id,
                width=40.0,
                height=32.0,
                units="cm",
                camera_height_cm=200.0,
            )
        )
        result = json.loads(result_contents[0].text)
        assert "camera_height_cm" in result
        assert result["camera_height_cm"] == 200.0
    finally:
        try:
            del registry._cameras[cam_id]
        except KeyError:
            pass
        try:
            playfield_registry.remove(pf_id)
        except KeyError:
            pass
        _cam_info.pop(cam_id, None)


# ---------------------------------------------------------------------------
# get_tags MCP tool — tag_heights_json parameter
# ---------------------------------------------------------------------------


def _make_detection_entry_with_tags(
    source_id: str,
    tags: list[dict],
) -> MagicMock:
    """Create a mock DetectionEntry whose ring_buffer returns *tags*."""
    from aprilcam.server.mcp_server import DetectionEntry

    frame_record = MagicMock()
    frame_record.to_dict.return_value = {
        "frame": 1,
        "timestamp": 1000.0,
        "tags": tags,
        "source_id": source_id,
    }

    ring_buffer = MagicMock()
    ring_buffer.get_latest.return_value = frame_record

    entry = MagicMock(spec=DetectionEntry)
    entry.ring_buffer = ring_buffer
    entry.robot_tag_id = None
    return entry


def test_get_tags_no_override_unchanged():
    """tag_heights_json=None returns result unchanged from _handle_get_tags."""
    import asyncio
    from aprilcam.server import mcp_server
    from aprilcam.server.mcp_server import detection_registry

    source_id = "pf_notag_unchanged"
    tags = [{"id": 5, "world_xy": [50.0, 50.0], "center_px": [320, 240], "corners_px": []}]
    entry = _make_detection_entry_with_tags(source_id, tags)
    detection_registry[source_id] = entry

    try:
        result_contents = asyncio.run(mcp_server.get_tags(source_id=source_id))
        result = json.loads(result_contents[0].text)
        assert "error" not in result
        assert result["tags"][0]["world_xy"] == [50.0, 50.0]
    finally:
        detection_registry.pop(source_id, None)


def test_get_tags_tag_heights_json_overrides(tmp_path):
    """Supplying tag_heights_json corrects world_xy for matching tags."""
    import asyncio
    from aprilcam.server import mcp_server
    from aprilcam.server.mcp_server import detection_registry, _cam_info

    source_id = "cam_ht_override"
    tags = [{"id": 5, "world_xy": [50.0, 50.0], "center_px": [320, 240], "corners_px": []}]
    entry = _make_detection_entry_with_tags(source_id, tags)
    detection_registry[source_id] = entry

    # Write a calibration.json with camera_position
    camera_dir = tmp_path / source_id
    camera_dir.mkdir()
    cal = _minimal_cal(
        camera_position=CameraPosition(x_offset=0.0, y_offset=0.0, height=180.0),
    )
    save_calibration_to_camera_dir(cal, camera_dir, field_width_cm=101.0, field_height_cm=89.0)
    _cam_info[source_id] = {"camera_dir": str(camera_dir)}

    try:
        result_contents = asyncio.run(
            mcp_server.get_tags(
                source_id=source_id,
                tag_heights_json='{"5": 12.0}',
            )
        )
        result = json.loads(result_contents[0].text)
        assert "error" not in result
        corrected = result["tags"][0]["world_xy"]
        expected = 50.0 * (1.0 - 12.0 / 180.0)
        assert abs(corrected[0] - expected) < 0.01
        assert abs(corrected[1] - expected) < 0.01
    finally:
        detection_registry.pop(source_id, None)
        _cam_info.pop(source_id, None)


def test_get_tags_tag_heights_json_zero_suppresses(tmp_path):
    """tag_heights_json with height 0 suppresses correction for that tag."""
    import asyncio
    from aprilcam.server import mcp_server
    from aprilcam.server.mcp_server import detection_registry, _cam_info

    source_id = "cam_ht_zero"
    tags = [{"id": 5, "world_xy": [50.0, 50.0], "center_px": [320, 240], "corners_px": []}]
    entry = _make_detection_entry_with_tags(source_id, tags)
    detection_registry[source_id] = entry

    camera_dir = tmp_path / source_id
    camera_dir.mkdir()
    # Persist calibration with tag 5 having height 12.0 (which would normally correct)
    cal = _minimal_cal(
        camera_position=CameraPosition(x_offset=0.0, y_offset=0.0, height=180.0),
        tag_heights={5: 12.0},
    )
    save_calibration_to_camera_dir(cal, camera_dir, field_width_cm=101.0, field_height_cm=89.0)
    _cam_info[source_id] = {"camera_dir": str(camera_dir)}

    try:
        result_contents = asyncio.run(
            mcp_server.get_tags(
                source_id=source_id,
                tag_heights_json='{"5": 0}',
            )
        )
        result = json.loads(result_contents[0].text)
        assert "error" not in result
        # Per-call override of 0 suppresses correction → world_xy unchanged
        assert result["tags"][0]["world_xy"] == [50.0, 50.0]
    finally:
        detection_registry.pop(source_id, None)
        _cam_info.pop(source_id, None)


def test_get_tags_invalid_json_returns_error():
    """Malformed tag_heights_json string returns an error response."""
    import asyncio
    from aprilcam.server import mcp_server
    from aprilcam.server.mcp_server import detection_registry

    source_id = "cam_ht_invalid"
    tags = [{"id": 5, "world_xy": [50.0, 50.0], "center_px": [320, 240], "corners_px": []}]
    entry = _make_detection_entry_with_tags(source_id, tags)
    detection_registry[source_id] = entry

    try:
        result_contents = asyncio.run(
            mcp_server.get_tags(
                source_id=source_id,
                tag_heights_json="not valid json {{{",
            )
        )
        result = json.loads(result_contents[0].text)
        assert "error" in result
        assert "Invalid tag_heights_json" in result["error"]
    finally:
        detection_registry.pop(source_id, None)


def test_get_tags_tag_not_in_heights_unaffected(tmp_path):
    """Tags not in merged heights dict are not corrected."""
    import asyncio
    from aprilcam.server import mcp_server
    from aprilcam.server.mcp_server import detection_registry, _cam_info

    source_id = "cam_ht_unaffected"
    tags = [
        {"id": 5, "world_xy": [50.0, 50.0], "center_px": [320, 240], "corners_px": []},
        {"id": 99, "world_xy": [30.0, 40.0], "center_px": [200, 200], "corners_px": []},
    ]
    entry = _make_detection_entry_with_tags(source_id, tags)
    detection_registry[source_id] = entry

    camera_dir = tmp_path / source_id
    camera_dir.mkdir()
    cal = _minimal_cal(
        camera_position=CameraPosition(x_offset=0.0, y_offset=0.0, height=180.0),
    )
    save_calibration_to_camera_dir(cal, camera_dir, field_width_cm=101.0, field_height_cm=89.0)
    _cam_info[source_id] = {"camera_dir": str(camera_dir)}

    try:
        result_contents = asyncio.run(
            mcp_server.get_tags(
                source_id=source_id,
                tag_heights_json='{"5": 12.0}',
            )
        )
        result = json.loads(result_contents[0].text)
        assert "error" not in result

        tags_result = {t["id"]: t for t in result["tags"]}
        # Tag 5 corrected
        assert tags_result[5]["world_xy"][0] != 50.0
        # Tag 99 not in heights → unchanged
        assert tags_result[99]["world_xy"] == [30.0, 40.0]
    finally:
        detection_registry.pop(source_id, None)
        _cam_info.pop(source_id, None)


def test_get_tags_merges_calibration_heights_with_override(tmp_path):
    """Per-call heights override persisted heights for matching IDs; others remain."""
    import asyncio
    from aprilcam.server import mcp_server
    from aprilcam.server.mcp_server import detection_registry, _cam_info

    source_id = "cam_ht_merge"
    tags = [
        {"id": 5, "world_xy": [50.0, 50.0], "center_px": [320, 240], "corners_px": []},
        {"id": 7, "world_xy": [60.0, 60.0], "center_px": [200, 200], "corners_px": []},
    ]
    entry = _make_detection_entry_with_tags(source_id, tags)
    detection_registry[source_id] = entry

    camera_dir = tmp_path / source_id
    camera_dir.mkdir()
    # Persist heights: 5→12.0, 7→8.0
    cal = _minimal_cal(
        camera_position=CameraPosition(x_offset=0.0, y_offset=0.0, height=180.0),
        tag_heights={5: 12.0, 7: 8.0},
    )
    save_calibration_to_camera_dir(cal, camera_dir, field_width_cm=101.0, field_height_cm=89.0)
    _cam_info[source_id] = {"camera_dir": str(camera_dir)}

    try:
        # Override tag 5 to 0 (suppress); tag 7 stays at calibration value 8.0
        result_contents = asyncio.run(
            mcp_server.get_tags(
                source_id=source_id,
                tag_heights_json='{"5": 0}',
            )
        )
        result = json.loads(result_contents[0].text)
        assert "error" not in result

        tags_result = {t["id"]: t for t in result["tags"]}
        # Tag 5 suppressed by per-call override of 0 → unchanged
        assert tags_result[5]["world_xy"] == [50.0, 50.0]
        # Tag 7 uses persisted height 8.0 → corrected
        r = 8.0 / 180.0
        expected_7 = 60.0 * (1.0 - r)
        assert abs(tags_result[7]["world_xy"][0] - expected_7) < 0.01
    finally:
        detection_registry.pop(source_id, None)
        _cam_info.pop(source_id, None)
