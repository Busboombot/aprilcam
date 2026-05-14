"""Smoke tests for LiveViewProcess command-pipe IPC (T003).

These tests verify the public API of the command pipe layer without
spawning a real child process or requiring camera hardware.
"""

import pytest

from aprilcam.ui.liveview import LiveViewProcess


# ---------------------------------------------------------------------------
# set_initial_paths stores the list before start()
# ---------------------------------------------------------------------------


def test_live_view_process_set_initial_paths() -> None:
    """set_initial_paths stores the list; no process is spawned."""
    proc = LiveViewProcess(camera_index=0)
    assert proc._initial_paths == []

    sample = [
        {
            "path_id": "path_000",
            "playfield_id": "pf_test",
            "waypoints": [
                {
                    "x": 1.0,
                    "y": 2.0,
                    "size_cm": 5.0,
                    "symbol": "circle",
                    "symbol_color": [255, 0, 0],
                    "line_color": [0, 255, 0],
                }
            ],
        }
    ]
    proc.set_initial_paths(sample)
    assert proc._initial_paths == sample
    # Verify it stored a copy, not the original list reference
    sample.append({"extra": True})
    assert len(proc._initial_paths) == 1


# ---------------------------------------------------------------------------
# send_command is a no-op when not running
# ---------------------------------------------------------------------------


def test_send_command_noop_when_not_running() -> None:
    """send_command must not raise when the live view is not running."""
    proc = LiveViewProcess(camera_index=0)
    assert not proc.is_running
    # None of these should raise
    proc.send_command({"op": "clear"})
    proc.send_command({"op": "add", "path": {"path_id": "path_000"}})
    proc.send_command({"op": "remove", "path_id": "path_000"})
