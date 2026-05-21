"""HSV color classifier with configurable color ranges."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .objects import ObjectRecord

# HSV ranges for colored object detection.
# OpenCV HSV: H=0-180, S=0-255, V=0-255.
#
# Tuned against 12 burst-captured frames from Arducam OV9782 (camera 4).
# Measured HSV medians at box locations:
#   black   H=105±17 S=200±51 V= 67±32 → detect by V<75 (dark), any H/S
#   red     H=177±64 S= 73±12 V=254±11 → H=165-180 + H=0-10, S_min=60
#   orange  H= 19± 1 S=100± 9 V=254± 5 → H=14-26, S_min=60
#   yellow  H= 26±22 S=  2± 7 V=255±31 → near-white: S<35, V>215 (H useless when S≈0)
#   green   H= 90± 1 S=252± 9 V=191± 5 → H=35-93, S_min=60
#   blue    H= 97± 1 S=254± 3 V=255± 2 → H=93-102, S_min=200
#   purple  H=114± 1 S=211±18 V=194±19 → H=102-130, S_min=150
#   magenta H=150±40 S= 85±15 V=254±16 → H=135-165, S_min=60
#
# Per-color S minimums eliminate wood-grain false positives (was 26+ FP/frame
# with global S_min=30; now 0 FP/frame on test frames).
DEFAULT_COLOR_RANGES: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {
    "black":   [((0,   0,   0),  (180, 255,  75))],
    "red":     [((0,   60,  60), (10,  255, 255)), ((165, 60, 60), (180, 255, 255))],
    "orange":  [((14,  60,  60), (26,  255, 255))],
    "yellow":  [((0,   0,  215), (180,  35, 255))],
    "green":   [((35,  60,  60), (93,  255, 255))],
    "blue":    [((93,  200,  60), (102, 255, 255))],
    "purple":  [((102, 150,  60), (130, 255, 255))],
    "magenta": [((135,  60,  60), (165, 255, 255))],
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

        Uses a tight center crop and filters to bright, saturated pixels
        to avoid background contamination from the dark playfield.

        Args:
            frame_bgr: BGR uint8 image.
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            radius: Radius of the ROI around the point.

        Returns:
            Color name or ``"unknown"``.
        """
        h_img, w_img = frame_bgr.shape[:2]
        # Use a tight radius to focus on the cube face
        r = min(radius, 10)
        x1 = max(0, int(x - r))
        y1 = max(0, int(y - r))
        x2 = min(w_img, int(x + r))
        y2 = min(h_img, int(y + r))
        if x2 <= x1 or y2 <= y1:
            return "unknown"

        roi = frame_bgr[y1:y2, x1:x2]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # Only consider bright, saturated pixels (the actual cube face,
        # not dark playfield background bleeding into the ROI).
        bright_sat_mask = (hsv_roi[:, :, 1] > 40) & (hsv_roi[:, :, 2] > 80)
        n_valid = int(np.count_nonzero(bright_sat_mask))
        if n_valid < 5:
            # Fall back to all pixels with lower threshold
            bright_sat_mask = hsv_roi[:, :, 1] > 25
            n_valid = int(np.count_nonzero(bright_sat_mask))
            if n_valid < 5:
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
            # Count only within bright+saturated pixels
            match_count = int(np.count_nonzero(mask & bright_sat_mask.astype(np.uint8)))
            pct = float(match_count) / n_valid
            if pct > best_pct:
                best_pct = pct
                best_color = color_name
        return best_color
