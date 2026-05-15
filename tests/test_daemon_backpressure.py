"""Unit tests for CameraPipeline subscriber/backpressure logic (T009).

Tests are hardware-free: they exercise the queue fan-out and subscriber
management logic directly without opening a real VideoCapture.
"""

from __future__ import annotations

import queue

import pytest

from aprilcam.daemon.camera_pipeline import CameraPipeline
from aprilcam.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path) -> CameraPipeline:
    """Return an un-started CameraPipeline backed by a tmp_path config."""
    config = Config(
        data_dir=tmp_path / "data",
        socket_dir=tmp_path / "sockets",
        calibration_source=tmp_path / "calibration.json",
        calibration_save_path=tmp_path / "calibration.json",
        log_level="INFO",
        daemon_pidfile=tmp_path / "aprilcamd.pid",
    )
    return CameraPipeline("cam_test", index=0, config=config)


# ---------------------------------------------------------------------------
# Queue drop logic (pure queue test — no CameraPipeline involved)
# ---------------------------------------------------------------------------


def test_queue_drop_on_full():
    """A bounded queue silently drops items when full via put_nowait."""
    q: queue.Queue[int] = queue.Queue(maxsize=2)

    items = [1, 2, 3]
    for item in items:
        try:
            q.put_nowait(item)
        except queue.Full:
            pass  # silent drop — matches CameraPipeline fan-out behaviour

    assert q.qsize() == 2
    assert q.get_nowait() == 1
    assert q.get_nowait() == 2
    assert q.empty()


def test_queue_drop_does_not_raise():
    """Silently swallowing queue.Full must not propagate any exception."""
    q: queue.Queue[bytes] = queue.Queue(maxsize=1)
    q.put_nowait(b"first")

    # Second put should be silently dropped, not raise
    try:
        q.put_nowait(b"second")
    except queue.Full:
        pass  # expected — swallow

    assert q.qsize() == 1


# ---------------------------------------------------------------------------
# CameraPipeline subscriber management (no camera required)
# ---------------------------------------------------------------------------


def test_add_subscriber_registers_queue(tmp_path):
    """add_subscriber stores the queue; it will receive fan-out data."""
    pipeline = _make_pipeline(tmp_path)
    q: queue.Queue[bytes] = queue.Queue()
    pipeline.add_subscriber(q)

    with pipeline._sub_lock:
        assert q in pipeline._subscribers


def test_remove_subscriber_unregisters_queue(tmp_path):
    """remove_subscriber removes the queue from the subscriber list."""
    pipeline = _make_pipeline(tmp_path)
    q: queue.Queue[bytes] = queue.Queue()
    pipeline.add_subscriber(q)
    pipeline.remove_subscriber(q)

    with pipeline._sub_lock:
        assert q not in pipeline._subscribers


def test_add_subscriber_idempotent(tmp_path):
    """Adding the same queue twice must not create duplicate entries."""
    pipeline = _make_pipeline(tmp_path)
    q: queue.Queue[bytes] = queue.Queue()
    pipeline.add_subscriber(q)
    pipeline.add_subscriber(q)

    with pipeline._sub_lock:
        count = pipeline._subscribers.count(q)
    assert count == 1


def test_remove_subscriber_missing_does_not_raise(tmp_path):
    """Removing a queue that was never added must be a no-op."""
    pipeline = _make_pipeline(tmp_path)
    q: queue.Queue[bytes] = queue.Queue()
    pipeline.remove_subscriber(q)  # must not raise


def test_multiple_subscribers(tmp_path):
    """Multiple distinct subscribers can be registered simultaneously."""
    pipeline = _make_pipeline(tmp_path)
    queues = [queue.Queue() for _ in range(5)]
    for q in queues:
        pipeline.add_subscriber(q)

    with pipeline._sub_lock:
        for q in queues:
            assert q in pipeline._subscribers
    assert len(pipeline._subscribers) == 5


def test_remove_one_subscriber_leaves_others(tmp_path):
    """Removing one subscriber must not affect the remaining ones."""
    pipeline = _make_pipeline(tmp_path)
    q1: queue.Queue[bytes] = queue.Queue()
    q2: queue.Queue[bytes] = queue.Queue()
    pipeline.add_subscriber(q1)
    pipeline.add_subscriber(q2)

    pipeline.remove_subscriber(q1)

    with pipeline._sub_lock:
        assert q1 not in pipeline._subscribers
        assert q2 in pipeline._subscribers
