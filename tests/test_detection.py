"""Tests for TagRecord, FrameRecord, RingBuffer, and DetectionLoop."""

import json
import math
import threading

import pytest

import numpy as np

from aprilcam.detection import FrameRecord, RingBuffer, TagRecord, DetectionLoop
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


# --- RingBuffer tests ---


def _make_frame(i: int) -> FrameRecord:
    """Create a minimal FrameRecord for testing."""
    return FrameRecord(timestamp=float(i), frame_index=i, tags=[])


def test_ringbuffer_empty():
    rb = RingBuffer()
    assert len(rb) == 0
    assert rb.get_latest() is None
    assert rb.get_last_n(5) == []


def test_ringbuffer_append_and_get_latest():
    rb = RingBuffer()
    fr = _make_frame(1)
    rb.append(fr)
    assert rb.get_latest() is fr


def test_ringbuffer_get_last_n():
    rb = RingBuffer()
    frames = [_make_frame(i) for i in range(5)]
    for f in frames:
        rb.append(f)
    result = rb.get_last_n(3)
    assert len(result) == 3
    assert result == frames[2:]


def test_ringbuffer_overflow():
    rb = RingBuffer(maxlen=300)
    for i in range(310):
        rb.append(_make_frame(i))
    assert len(rb) == 300


def test_ringbuffer_get_last_n_exceeds_size():
    rb = RingBuffer()
    frames = [_make_frame(i) for i in range(5)]
    for f in frames:
        rb.append(f)
    result = rb.get_last_n(100)
    assert len(result) == 5
    assert result == frames


def test_ringbuffer_get_last_n_zero():
    rb = RingBuffer()
    rb.append(_make_frame(0))
    assert rb.get_last_n(0) == []


def test_ringbuffer_clear():
    rb = RingBuffer()
    for i in range(5):
        rb.append(_make_frame(i))
    assert len(rb) == 5
    rb.clear()
    assert len(rb) == 0
    assert rb.get_latest() is None


def test_ringbuffer_thread_safety():
    rb = RingBuffer(maxlen=300)
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(500):
                rb.append(_make_frame(i))
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(500):
                rb.get_latest()
                rb.get_last_n(10)
                len(rb)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"Thread safety errors: {errors}"


# --- process_frame tests ---

import time
from pathlib import Path

import cv2 as cv

from aprilcam.aprilcam import AprilCam

TEST_DATA = Path(__file__).parent / "data"


def _make_aprilcam(img: np.ndarray) -> AprilCam:
    """Create a headless AprilCam suitable for process_frame testing."""
    return AprilCam(
        index=0,
        backend=None,
        speed_alpha=0.5,
        family="36h11",
        proc_width=960,
        headless=True,
        cap=cv.VideoCapture(0),
    )


def test_process_frame_returns_tagrecords():
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    img = cv.imread(str(path))
    assert img is not None, "Failed to read test image"
    cam = _make_aprilcam(img)
    records = cam.process_frame(img, time.monotonic())
    assert isinstance(records, list)
    for r in records:
        assert isinstance(r, TagRecord)


def test_process_frame_increments_frame_idx():
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    img = cv.imread(str(path))
    assert img is not None
    cam = _make_aprilcam(img)
    assert cam._frame_idx == 0
    cam.process_frame(img, time.monotonic())
    assert cam._frame_idx == 1
    cam.process_frame(img, time.monotonic())
    assert cam._frame_idx == 2


def test_reset_state():
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    img = cv.imread(str(path))
    assert img is not None
    cam = _make_aprilcam(img)
    # Process a frame to populate state
    cam.process_frame(img, time.monotonic())
    assert cam._frame_idx == 1
    assert cam._prev_gray is not None

    # Reset and verify clean state
    cam.reset_state()
    assert cam._frame_idx == 0
    assert cam._prev_gray is None
    assert cam._tracks == {}
    assert cam._tag_models == {}


# --- DetectionLoop tests ---


class FakeCapture:
    """A fake video capture that returns the same image repeatedly."""

    def __init__(self, image_path, max_frames=1000):
        self._img = cv.imread(str(image_path))
        assert self._img is not None
        self._count = 0
        self._max = max_frames

    def read(self):
        if self._count >= self._max:
            return False, None
        self._count += 1
        return True, self._img.copy()

    def release(self):
        pass


class FailingCapture:
    """A fake capture that always raises on read()."""

    def read(self):
        raise RuntimeError("simulated capture failure")

    def release(self):
        pass


def _make_loop_aprilcam():
    """Create a headless AprilCam for DetectionLoop tests."""
    dummy_cap = cv.VideoCapture()
    return AprilCam(
        index=0,
        backend=None,
        speed_alpha=0.5,
        family="36h11",
        proc_width=960,
        headless=True,
        cap=dummy_cap,
    )


def test_detectionloop_start_stop():
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    fake_cap = FakeCapture(path)
    cam = _make_loop_aprilcam()
    buf = RingBuffer()
    loop = DetectionLoop(source=fake_cap, aprilcam=cam, ring_buffer=buf)
    loop.start()
    time.sleep(0.5)
    loop.stop()
    assert loop.frame_count > 0
    assert not loop.is_running


def test_detectionloop_writes_to_ringbuffer():
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    fake_cap = FakeCapture(path)
    cam = _make_loop_aprilcam()
    buf = RingBuffer()
    loop = DetectionLoop(source=fake_cap, aprilcam=cam, ring_buffer=buf)
    loop.start()
    time.sleep(0.3)
    loop.stop()
    assert buf.get_latest() is not None


def test_detectionloop_double_start_raises():
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    fake_cap = FakeCapture(path, max_frames=100000)
    cam = _make_loop_aprilcam()
    buf = RingBuffer()
    loop = DetectionLoop(source=fake_cap, aprilcam=cam, ring_buffer=buf)
    loop.start()
    try:
        with pytest.raises(RuntimeError):
            loop.start()
    finally:
        loop.stop()


def test_detectionloop_stop_idempotent():
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    fake_cap = FakeCapture(path)
    cam = _make_loop_aprilcam()
    buf = RingBuffer()
    loop = DetectionLoop(source=fake_cap, aprilcam=cam, ring_buffer=buf)
    loop.start()
    time.sleep(0.1)
    loop.stop()
    loop.stop()  # second stop should not raise


def test_detectionloop_handles_frame_errors():
    cam = _make_loop_aprilcam()
    buf = RingBuffer()
    loop = DetectionLoop(source=FailingCapture(), aprilcam=cam, ring_buffer=buf)
    loop.start()
    time.sleep(0.5)
    loop.stop()
    assert loop.error is not None


def test_detectionloop_consecutive_failure_stops():
    cam = _make_loop_aprilcam()
    buf = RingBuffer()
    loop = DetectionLoop(source=FailingCapture(), aprilcam=cam, ring_buffer=buf)
    loop.start()
    time.sleep(1.0)
    assert not loop.is_running  # auto-stopped due to consecutive failures
    loop.stop()


def test_process_frame_static_velocity_near_zero():
    """Feed the same image 15 times — static tags should have speed_px == 0.0."""
    path = TEST_DATA / "playfield_cam3_moved.jpg"
    if not path.exists():
        pytest.skip("Test image not available")
    img = cv.imread(str(path))
    assert img is not None, "Failed to read test image"
    cam = _make_aprilcam(img)

    # Feed the same image repeatedly with incrementing timestamps
    all_records: list[list[TagRecord]] = []
    base_ts = 1000.0
    for i in range(15):
        records = cam.process_frame(img, base_ts + i * 0.033)  # ~30 fps
        all_records.append(records)

    # Check the last 5 frames: all tags should have speed_px == 0.0
    # because the EMA-smoothed speed of a static tag falls below the
    # 10.0 px/s dead-band threshold.
    for frame_records in all_records[-5:]:
        assert len(frame_records) > 0, "Expected at least one tag detection"
        for tr in frame_records:
            assert tr.speed_px is not None, f"Tag {tr.id}: speed_px should not be None"
            assert tr.speed_px == 0.0, (
                f"Tag {tr.id}: speed_px={tr.speed_px} should be 0.0 (dead-band)"
            )
