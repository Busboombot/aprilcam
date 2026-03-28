"""Object detection dataclasses and square detector for non-tag objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class ObjectRecord:
    """A detected non-tag object (e.g. a colored cube)."""

    center_px: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_px: float
    world_xy: Optional[Tuple[float, float]] = None
    color: str = "unknown"
    object_type: str = "cube"
    confidence: float = 1.0


class FrameResult:
    """Result of processing a single frame.

    Backward-compatible with list[TagRecord]: iterating, len(), and
    indexing all delegate to the tags list.
    """

    def __init__(self, tags, objects=None, timestamp=0.0, frame_index=0):
        self.tags = tags
        self.objects = objects or []
        self.timestamp = timestamp
        self.frame_index = frame_index

    def __iter__(self):
        return iter(self.tags)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        return self.tags[idx]


class SquareDetector:
    """Detect square-ish contours in a grayscale image."""

    def __init__(self, min_area: int = 200, max_area: int = 5000):
        self.min_area = min_area
        self.max_area = max_area

    def detect(
        self,
        gray: np.ndarray,
        homography: np.ndarray | None = None,
        tag_corners: list[np.ndarray] | None = None,
        exclusion_point: Tuple[float, float] | None = None,
        exclusion_radius: float = 50,
    ) -> list[ObjectRecord]:
        """Detect square objects in a grayscale image.

        Args:
            gray: Grayscale uint8 image.
            homography: Optional 3x3 homography matrix for world coords.
            tag_corners: List of Nx2 arrays of tag corner polygons to exclude.
            exclusion_point: Optional (x, y) point; detections within
                exclusion_radius of this point are excluded.
            exclusion_radius: Radius around exclusion_point to exclude.

        Returns:
            List of ObjectRecord for each detected square-like contour.
        """
        thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
        )
        contours_a, _ = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        # Also search the inverse to catch bright objects on dark backgrounds.
        contours_b, _ = cv.findContours(
            cv.bitwise_not(thresh), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        contours = list(contours_a) + list(contours_b)

        results: list[ObjectRecord] = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv.boundingRect(cnt)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect >= 2.0:
                continue

            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = area / hull_area
            if solidity <= 0.7:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0

            # Exclude centers inside tag corner polygons.
            if tag_corners:
                inside_tag = False
                for corners in tag_corners:
                    poly = corners.reshape(-1, 1, 2).astype(np.float32)
                    if cv.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        inside_tag = True
                        break
                if inside_tag:
                    continue

            # Exclude centers near the exclusion point.
            if exclusion_point is not None:
                dx = cx - exclusion_point[0]
                dy = cy - exclusion_point[1]
                if (dx * dx + dy * dy) ** 0.5 <= exclusion_radius:
                    continue

            world_xy: Tuple[float, float] | None = None
            if homography is not None:
                pt = homography @ np.array([cx, cy, 1.0])
                world_xy = (float(pt[0] / pt[2]), float(pt[1] / pt[2]))

            results.append(
                ObjectRecord(
                    center_px=(cx, cy),
                    bbox=(x, y, w, h),
                    area_px=float(area),
                    world_xy=world_xy,
                )
            )

        return results
