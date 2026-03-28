"""HSV color classifier with configurable color ranges."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .objects import ObjectRecord

# HSV ranges tuned from real camera data (HD USB CAMERA, overhead lighting).
# OpenCV HSV: H=0-180, S=0-255, V=0-255.
# Ranges are non-overlapping in H to avoid misclassification.
DEFAULT_COLOR_RANGES: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {
    "red": [((0, 80, 80), (8, 255, 255)), ((165, 80, 80), (180, 255, 255))],
    "orange": [((8, 80, 80), (18, 255, 255))],
    "yellow": [((18, 80, 80), (32, 255, 255))],
    "green": [((32, 80, 80), (85, 255, 255))],
    "blue": [((85, 80, 80), (130, 255, 255))],
    "purple": [((130, 60, 60), (165, 255, 255))],
}


class ColorClassifier:
    """Detect and classify colored objects using HSV thresholding."""

    def __init__(
        self,
        color_ranges: dict | None = None,
        min_area: int = 200,
        max_area: int = 5000,
    ):
        self.color_ranges = (
            color_ranges if color_ranges is not None else dict(DEFAULT_COLOR_RANGES)
        )
        self.min_area = min_area
        self.max_area = max_area

    def classify(
        self,
        frame_bgr: np.ndarray,
        homography: np.ndarray | None = None,
    ) -> list[ObjectRecord]:
        """Detect colored objects in a BGR frame using HSV thresholding.

        Args:
            frame_bgr: BGR uint8 image.
            homography: Optional 3x3 homography matrix for world coords.

        Returns:
            List of ObjectRecord for each detected colored region.
        """
        hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        results: list[ObjectRecord] = []

        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv.inRange(
                    hsv,
                    np.array(lo, dtype=np.uint8),
                    np.array(hi, dtype=np.uint8),
                )
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            contours, _ = cv.findContours(
                mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv.contourArea(cnt)
                if area < self.min_area or area > self.max_area:
                    continue
                M = cv.moments(cnt)
                if M["m00"] < 1e-6:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                x, y, w, h = cv.boundingRect(cnt)

                world_xy = None
                if homography is not None:
                    vec = homography @ np.array([cx, cy, 1.0])
                    if abs(vec[2]) > 1e-9:
                        world_xy = (
                            float(vec[0] / vec[2]),
                            float(vec[1] / vec[2]),
                        )

                results.append(
                    ObjectRecord(
                        center_px=(float(cx), float(cy)),
                        bbox=(x, y, w, h),
                        area_px=float(area),
                        world_xy=world_xy,
                        color=color_name,
                        object_type="cube",
                        confidence=1.0,
                    )
                )
        return results

    def classify_at_point(
        self, frame_bgr: np.ndarray, x: float, y: float, radius: int = 20
    ) -> str:
        """Classify the dominant color at a point.

        Args:
            frame_bgr: BGR uint8 image.
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            radius: Radius of the ROI around the point.

        Returns:
            Color name or ``"unknown"``.
        """
        h_img, w_img = frame_bgr.shape[:2]
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(w_img, int(x + radius))
        y2 = min(h_img, int(y + radius))
        if x2 <= x1 or y2 <= y1:
            return "unknown"

        roi = frame_bgr[y1:y2, x1:x2]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
        if total_pixels == 0:
            return "unknown"

        best_color = "unknown"
        best_pct = 0.10  # minimum 10% threshold

        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv.inRange(
                    hsv_roi,
                    np.array(lo, dtype=np.uint8),
                    np.array(hi, dtype=np.uint8),
                )
            pct = float(np.count_nonzero(mask)) / total_pixels
            if pct > best_pct:
                best_pct = pct
                best_color = color_name
        return best_color
