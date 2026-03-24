"""Tests for TagRecord and FrameRecord dataclasses."""

import json
import math

import numpy as np

from aprilcam.detection import FrameRecord, TagRecord
from aprilcam.models import AprilTag


def _make_tag_record(**overrides):
    """Helper to create a TagRecord with sensible defaults."""
    defaults = dict(
        id=7,
        center_px=(100.0, 200.0),
        corners_px=[[90.0, 190.0], [110.0, 190.0], [110.0, 210.0], [90.0, 210.0]],
        orientation_yaw=0.5,
        world_xy=(10.0, 20.0),
        in_playfield=True,
        vel_px=(1.0, 2.0),
        speed_px=2.236,
        vel_world=(0.1, 0.2),
        speed_world=0.2236,
        heading_rad=1.107,
        timestamp=1000.0,
        frame_index=42,
    )
    defaults.update(overrides)
    return TagRecord(**defaults)


def test_tagrecord_construction():
    tr = _make_tag_record()
    assert tr.id == 7
    assert tr.center_px == (100.0, 200.0)
    assert len(tr.corners_px) == 4
    assert tr.orientation_yaw == 0.5
    assert tr.world_xy == (10.0, 20.0)
    assert tr.in_playfield is True
    assert tr.vel_px == (1.0, 2.0)
    assert tr.speed_px == 2.236
    assert tr.vel_world == (0.1, 0.2)
    assert tr.speed_world == 0.2236
    assert tr.heading_rad == 1.107
    assert tr.timestamp == 1000.0
    assert tr.frame_index == 42


def test_tagrecord_optional_none():
    tr = _make_tag_record(
        world_xy=None,
        vel_px=None,
        speed_px=None,
        vel_world=None,
        speed_world=None,
        heading_rad=None,
    )
    assert tr.world_xy is None
    assert tr.vel_px is None
    assert tr.speed_px is None
    assert tr.vel_world is None
    assert tr.speed_world is None
    assert tr.heading_rad is None


def test_tagrecord_to_dict():
    tr = _make_tag_record()
    d = tr.to_dict()
    assert isinstance(d, dict)
    assert d["id"] == 7
    assert d["center_px"] == [100.0, 200.0]
    assert d["corners_px"] == [
        [90.0, 190.0],
        [110.0, 190.0],
        [110.0, 210.0],
        [90.0, 210.0],
    ]
    assert d["world_xy"] == [10.0, 20.0]
    assert d["in_playfield"] is True
    assert d["vel_px"] == [1.0, 2.0]
    assert d["timestamp"] == 1000.0
    assert d["frame_index"] == 42


def test_tagrecord_to_dict_json_roundtrip():
    tr = _make_tag_record()
    d = tr.to_dict()
    serialized = json.dumps(d)
    restored = json.loads(serialized)
    assert restored == d


def test_tagrecord_to_dict_none_fields():
    tr = _make_tag_record(
        world_xy=None, vel_px=None, speed_px=None,
        vel_world=None, speed_world=None, heading_rad=None,
    )
    d = tr.to_dict()
    serialized = json.dumps(d)
    restored = json.loads(serialized)
    assert restored["world_xy"] is None
    assert restored["vel_px"] is None


def test_framerecord_construction():
    t1 = _make_tag_record(id=1, frame_index=10)
    t2 = _make_tag_record(id=2, frame_index=10)
    fr = FrameRecord(timestamp=500.0, frame_index=10, tags=[t1, t2])
    assert fr.timestamp == 500.0
    assert fr.frame_index == 10
    assert len(fr.tags) == 2
    assert fr.tags[0].id == 1
    assert fr.tags[1].id == 2


def test_framerecord_to_dict():
    t1 = _make_tag_record(id=1)
    t2 = _make_tag_record(id=2)
    fr = FrameRecord(timestamp=500.0, frame_index=10, tags=[t1, t2])
    d = fr.to_dict()
    assert d["timestamp"] == 500.0
    assert d["frame_index"] == 10
    assert len(d["tags"]) == 2
    assert d["tags"][0]["id"] == 1
    assert d["tags"][1]["id"] == 2
    # Verify full roundtrip
    assert json.loads(json.dumps(d)) == d


def test_tagrecord_from_apriltag():
    corners = np.array(
        [[90.0, 190.0], [110.0, 190.0], [110.0, 210.0], [90.0, 210.0]],
        dtype=np.float32,
    )
    tag = AprilTag.from_corners(tag_id=7, corners_px=corners, timestamp=999.0, frame=42)
    tag.in_playfield = True
    tag.world_xy = (10.0, 20.0)

    tr = TagRecord.from_apriltag(
        tag,
        vel_px=(1.0, 2.0),
        speed_px=2.236,
        vel_world=(0.1, 0.2),
        speed_world=0.2236,
        heading_rad=1.107,
        timestamp=999.0,
        frame_index=42,
    )

    assert tr.id == 7
    assert tr.center_px == tag.center_px
    assert len(tr.corners_px) == 4
    # Verify corners converted from numpy to plain lists
    for row in tr.corners_px:
        assert isinstance(row, list)
        for val in row:
            assert isinstance(val, float)
    assert tr.orientation_yaw == tag.orientation_yaw
    assert tr.world_xy == (10.0, 20.0)
    assert tr.in_playfield is True
    assert tr.vel_px == (1.0, 2.0)
    assert tr.speed_px == 2.236
    assert tr.timestamp == 999.0
    assert tr.frame_index == 42
    # Verify JSON-serializable (no numpy)
    d = tr.to_dict()
    assert json.loads(json.dumps(d)) == d
