from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

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
    - vel_px: pixel velocity vector (vx, vy) in px/s
    - speed_px: speed magnitude in px/s
    - last_ts: timestamp of last update
    """

    id: int
    corners_px: np.ndarray
    center_px: Tuple[float, float]
    top_dir_px: Tuple[float, float]
    orientation_yaw: float
    world_xy: Optional[Tuple[float, float]] = None
    vel_px: Tuple[float, float] = (0.0, 0.0)
    speed_px: float = 0.0
    last_ts: Optional[float] = None

    @staticmethod
    def from_corners(
        tag_id: int,
        corners_px: np.ndarray,
        homography: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
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
        )

    def update(self, corners_px: np.ndarray, timestamp: float, homography: Optional[np.ndarray] = None) -> None:
        prev_cx, prev_cy = self.center_px
        prev_ts = self.last_ts
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
        # velocity
        if prev_ts is not None and timestamp is not None:
            dt = max(1e-3, float(timestamp - prev_ts))
            vx = (self.center_px[0] - prev_cx) / dt
            vy = (self.center_px[1] - prev_cy) / dt
            self.vel_px = (float(vx), float(vy))
            self.speed_px = float(math.hypot(vx, vy))
        self.last_ts = float(timestamp)
