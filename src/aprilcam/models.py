from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque

import numpy as np


@dataclass
class AprilTag:
    """Represents a detected AprilTag and its tracked state.

    - id: tag ID
    - corners_px: 4x2 pixel coordinates (order as returned by detector)
    - center_px: pixel center (computed)
    - top_dir_px: unit vector from center toward the top edge midpoint (image coords)
    - world_xy: optional (X,Y) in world units (via homography)
    - orientation_yaw: yaw angle in radians in image plane (from +X toward +Y)
    - last_ts: timestamp of last update
    - frame: video frame index when measured
    - in_playfield: whether the tag center is within the playfield polygon
    """

    id: int
    corners_px: np.ndarray
    center_px: Tuple[float, float]
    top_dir_px: Tuple[float, float]
    orientation_yaw: float
    world_xy: Optional[Tuple[float, float]] = None
    last_ts: Optional[float] = None
    frame: int = 0
    in_playfield: bool = False

    @staticmethod
    def from_corners(
        tag_id: int,
        corners_px: np.ndarray,
        homography: Optional[np.ndarray] = None,
    timestamp: Optional[float] = None,
    frame: int = 0,
    ) -> "AprilTag":
        ptsf = corners_px.astype(np.float32)
        c = ptsf.mean(axis=0)
        p0, p1 = ptsf[0], ptsf[1]
        top_mid = (p0 + p1) / 2.0
        n = top_mid - c
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-6:
            n_unit = (float(n[0]) / n_norm, float(n[1]) / n_norm)
        else:
            # Fallback: perpendicular to first edge
            e = p1 - p0
            perp = np.array([-e[1], e[0]], dtype=np.float32)
            denom = float(np.linalg.norm(perp)) or 1.0
            n_unit = (float(perp[0]) / denom, float(perp[1]) / denom)
        yaw = math.atan2(n_unit[1], n_unit[0])
        world_xy: Optional[Tuple[float, float]] = None
        if homography is not None and homography.size == 9:
            vec = np.array([float(c[0]), float(c[1]), 1.0], dtype=float)
            Xw = homography @ vec
            if abs(float(Xw[2])) > 1e-9:
                world_xy = (float(Xw[0] / Xw[2]), float(Xw[1] / Xw[2]))
        return AprilTag(
            id=int(tag_id),
            corners_px=ptsf.copy(),
            center_px=(float(c[0]), float(c[1])),
            top_dir_px=n_unit,
            orientation_yaw=float(yaw),
            world_xy=world_xy,
            last_ts=timestamp,
            frame=int(frame),
        )

    def update(self, corners_px: np.ndarray, timestamp: float, homography: Optional[np.ndarray] = None) -> None:
        ptsf = corners_px.astype(np.float32)
        c = ptsf.mean(axis=0)
        p0, p1 = ptsf[0], ptsf[1]
        top_mid = (p0 + p1) / 2.0
        n = top_mid - c
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-6:
            n_unit = (float(n[0]) / n_norm, float(n[1]) / n_norm)
        else:
            e = p1 - p0
            perp = np.array([-e[1], e[0]], dtype=np.float32)
            denom = float(np.linalg.norm(perp)) or 1.0
            n_unit = (float(perp[0]) / denom, float(perp[1]) / denom)
        yaw = math.atan2(n_unit[1], n_unit[0])
        self.corners_px = ptsf.copy()
        self.center_px = (float(c[0]), float(c[1]))
        self.top_dir_px = n_unit
        self.orientation_yaw = float(yaw)
        if homography is not None and homography.size == 9:
            vec = np.array([float(c[0]), float(c[1]), 1.0], dtype=float)
            Xw = homography @ vec
            if abs(float(Xw[2])) > 1e-9:
                self.world_xy = (float(Xw[0] / Xw[2]), float(Xw[1] / Xw[2]))
        self.last_ts = float(timestamp)

    def clone(self) -> "AprilTag":
        """Return a deep-ish copy suitable for historical storage in flows."""
        return AprilTag(
            id=int(self.id),
            corners_px=self.corners_px.copy(),
            center_px=(float(self.center_px[0]), float(self.center_px[1])),
            top_dir_px=(float(self.top_dir_px[0]), float(self.top_dir_px[1])),
            orientation_yaw=float(self.orientation_yaw),
            world_xy=(None if self.world_xy is None else (float(self.world_xy[0]), float(self.world_xy[1]))),
            last_ts=(None if self.last_ts is None else float(self.last_ts)),
            frame=int(self.frame),
            in_playfield=bool(self.in_playfield),
        )


class AprilTagFlow:
    """Fixed-size history of AprilTag observations with convenient properties.

    Exposes the same attribute interface as AprilTag, returning values from the
    most recent AprilTag in the deque. Additionally computes vel_px and speed_px
    from the last two observations when available.
    """

    def __init__(self, maxlen: int = 5) -> None:
        self._deque: Deque[AprilTag] = deque(maxlen=maxlen)
        self._id: Optional[int] = None

    def add_tag(self, tag: AprilTag) -> None:
        if self._id is None:
            self._id = int(tag.id)
        self._deque.append(tag)

    # --- core accessors mirroring AprilTag ---
    @property
    def id(self) -> int:
        return int(self._id) if self._id is not None else -1

    def _last(self) -> Optional[AprilTag]:
        return self._deque[-1] if self._deque else None

    @property
    def corners_px(self) -> np.ndarray:
        t = self._last()
        return t.corners_px if t is not None else np.zeros((4, 2), dtype=np.float32)

    @property
    def center_px(self) -> Tuple[float, float]:
        t = self._last()
        return t.center_px if t is not None else (0.0, 0.0)

    @property
    def top_dir_px(self) -> Tuple[float, float]:
        t = self._last()
        return t.top_dir_px if t is not None else (1.0, 0.0)

    @property
    def orientation_yaw(self) -> float:
        t = self._last()
        return t.orientation_yaw if t is not None else 0.0

    @property
    def world_xy(self) -> Optional[Tuple[float, float]]:
        t = self._last()
        return t.world_xy if t is not None else None

    @property
    def last_ts(self) -> Optional[float]:
        t = self._last()
        return t.last_ts if t is not None else None

    @property
    def frame(self) -> int:
        t = self._last()
        return t.frame if t is not None else 0

    @property
    def in_playfield(self) -> bool:
        t = self._last()
        return bool(t.in_playfield) if t is not None else False

    # --- derived motion ---
    @property
    def vel_px(self) -> Tuple[float, float]:
        if len(self._deque) < 2:
            return (0.0, 0.0)
        a = self._deque[-2]
        b = self._deque[-1]
        if a.last_ts is None or b.last_ts is None:
            return (0.0, 0.0)
        dt = max(1e-3, float(b.last_ts - a.last_ts))
        vx = (b.center_px[0] - a.center_px[0]) / dt
        vy = (b.center_px[1] - a.center_px[1]) / dt
        return (float(vx), float(vy))

    @property
    def speed_px(self) -> float:
        vx, vy = self.vel_px
        return float(math.hypot(vx, vy))
