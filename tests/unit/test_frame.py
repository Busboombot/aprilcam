"""Tests for aprilcam.frame — FrameEntry and FrameRegistry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

from aprilcam.frame import FrameEntry, FrameRegistry


def _make_image(value: int = 0) -> np.ndarray:
    """Return a small 4x4 BGR test image filled with *value*."""
    return np.full((4, 4, 3), value, dtype=np.uint8)


# ------------------------------------------------------------------
# FrameEntry slot reference tests
# ------------------------------------------------------------------


def test_frame_entry_creation_slot_references() -> None:
    """On creation via the registry, all three slots share the same object."""
    registry = FrameRegistry()
    entry = registry.add(_make_image(42), source="cam_0")

    assert entry.original is entry.deskewed
    assert entry.deskewed is entry.processed


def test_frame_entry_deskew_breaks_reference() -> None:
    """Simulating a deskew replaces deskewed/processed but not original."""
    registry = FrameRegistry()
    entry = registry.add(_make_image(1), source="cam_0")

    new_array = _make_image(2)
    entry.deskewed = new_array
    entry.processed = entry.deskewed
    entry.is_deskewed = True

    assert entry.deskewed is not entry.original
    assert entry.processed is entry.deskewed
    assert entry.is_deskewed is True


# ------------------------------------------------------------------
# FrameRegistry basic operations
# ------------------------------------------------------------------


def test_frame_registry_add_and_get() -> None:
    """add() returns a FrameEntry; get() returns the same object."""
    registry = FrameRegistry()
    entry = registry.add(_make_image(), source="cam_0")

    retrieved = registry.get(entry.frame_id)
    assert retrieved is entry


def test_frame_registry_deterministic_ids() -> None:
    """Frame IDs follow the pattern frm_000, frm_001, frm_002, ..."""
    registry = FrameRegistry()
    ids = [registry.add(_make_image(i), source="cam_0").frame_id for i in range(3)]

    assert ids == ["frm_000", "frm_001", "frm_002"]


def test_frame_registry_auto_eviction() -> None:
    """When capacity is exceeded, the oldest frame is evicted."""
    registry = FrameRegistry(capacity=5)
    for i in range(6):
        registry.add(_make_image(i), source="cam_0")

    # frm_000 should have been evicted
    with pytest.raises(KeyError):
        registry.get("frm_000")

    # frm_001 through frm_005 should still exist
    for i in range(1, 6):
        entry = registry.get(f"frm_{i:03d}")
        assert entry.frame_id == f"frm_{i:03d}"

    assert len(registry) == 5


def test_frame_registry_release() -> None:
    """release() removes a frame; subsequent get() raises KeyError."""
    registry = FrameRegistry()
    entry = registry.add(_make_image(), source="cam_0")
    frame_id = entry.frame_id

    registry.release(frame_id)

    with pytest.raises(KeyError):
        registry.get(frame_id)

    assert len(registry) == 0


def test_frame_registry_release_unknown_raises() -> None:
    """release() on an unknown ID raises KeyError."""
    registry = FrameRegistry()

    with pytest.raises(KeyError):
        registry.release("frm_999")


def test_frame_registry_list() -> None:
    """list_frames() returns summary dicts in insertion order."""
    registry = FrameRegistry()
    registry.add(_make_image(0), source="cam_0", timestamp=100.0)
    registry.add(_make_image(1), source="cam_1", timestamp=200.0)

    summaries = registry.list_frames()

    assert len(summaries) == 2
    assert summaries[0]["frame_id"] == "frm_000"
    assert summaries[0]["source"] == "cam_0"
    assert summaries[0]["timestamp"] == 100.0
    assert summaries[0]["operations_applied"] == []
    assert summaries[0]["is_deskewed"] is False

    assert summaries[1]["frame_id"] == "frm_001"
    assert summaries[1]["source"] == "cam_1"
    assert summaries[1]["timestamp"] == 200.0


def test_frame_registry_clear() -> None:
    """clear() removes all frames."""
    registry = FrameRegistry()
    for i in range(5):
        registry.add(_make_image(i), source="cam_0")

    registry.clear()
    assert len(registry) == 0
    assert registry.list_frames() == []


def test_frame_registry_invalid_capacity() -> None:
    """Capacity < 1 raises ValueError."""
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        FrameRegistry(capacity=0)


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------


def test_frame_registry_thread_safety() -> None:
    """Concurrent adds from 4 threads produce no exceptions and correct length."""
    registry = FrameRegistry(capacity=300)
    num_threads = 4
    frames_per_thread = 25  # 4 * 25 = 100 total

    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        for i in range(frames_per_thread):
            try:
                registry.add(
                    _make_image((thread_id * 25 + i) % 256),
                    source=f"thread_{thread_id}",
                )
            except Exception as exc:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker, t) for t in range(num_threads)]
        for f in as_completed(futures):
            f.result()  # re-raise any exception from the thread

    assert errors == []
    assert len(registry) == num_threads * frames_per_thread
