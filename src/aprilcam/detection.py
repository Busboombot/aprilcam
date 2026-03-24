"""Immutable record types for tag detection results.

TagRecord and FrameRecord are frozen dataclasses used by the detection
loop and ring buffer to store per-frame tag observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from aprilcam.models import AprilTag


@dataclass(frozen=True)
class TagRecord:
    """Immutable snapshot of a single detected tag in one frame."""

    id: int
    center_px: tuple[float, float]
    corners_px: list[list[float]]  # 4x2 as plain lists
    orientation_yaw: float
    world_xy: tuple[float, float] | None
    in_playfield: bool
    vel_px: tuple[float, float] | None
    speed_px: float | None
    vel_world: tuple[float, float] | None
    speed_world: float | None
    heading_rad: float | None
    timestamp: float
    frame_index: int

    def to_dict(self) -> dict:
        """Return a plain dict with all JSON-serializable values."""
        return {
            "id": self.id,
            "center_px": list(self.center_px),
            "corners_px": [list(c) for c in self.corners_px],
            "orientation_yaw": self.orientation_yaw,
            "world_xy": list(self.world_xy) if self.world_xy is not None else None,
            "in_playfield": self.in_playfield,
            "vel_px": list(self.vel_px) if self.vel_px is not None else None,
            "speed_px": self.speed_px,
            "vel_world": list(self.vel_world) if self.vel_world is not None else None,
            "speed_world": self.speed_world,
            "heading_rad": self.heading_rad,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
        }

    @classmethod
    def from_apriltag(
        cls,
        tag: AprilTag,
        *,
        vel_px: tuple[float, float] | None = None,
        speed_px: float | None = None,
        vel_world: tuple[float, float] | None = None,
        speed_world: float | None = None,
        heading_rad: float | None = None,
        timestamp: float,
        frame_index: int,
    ) -> TagRecord:
        """Create a TagRecord from an existing AprilTag model instance.

        Converts numpy arrays to plain Python lists so the record is
        fully serializable without numpy.
        """
        corners_as_lists = [
            [float(x) for x in row] for row in tag.corners_px.tolist()
        ]
        return cls(
            id=tag.id,
            center_px=tag.center_px,
            corners_px=corners_as_lists,
            orientation_yaw=tag.orientation_yaw,
            world_xy=tag.world_xy,
            in_playfield=tag.in_playfield,
            vel_px=vel_px,
            speed_px=speed_px,
            vel_world=vel_world,
            speed_world=speed_world,
            heading_rad=heading_rad,
            timestamp=timestamp,
            frame_index=frame_index,
        )


@dataclass(frozen=True)
class FrameRecord:
    """Immutable snapshot of all tag detections for a single frame."""

    timestamp: float
    frame_index: int
    tags: list[TagRecord]

    def to_dict(self) -> dict:
        """Return a plain dict with tags as list of tag dicts."""
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "tags": [t.to_dict() for t in self.tags],
        }
