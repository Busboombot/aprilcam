"""Image processing functions for MCP tools."""

from __future__ import annotations

from typing import Any

import cv2 as cv
import numpy as np


def process_detect_lines(
    frame: np.ndarray,
    threshold: int = 50,
    min_length: int = 50,
    max_gap: int = 10,
) -> list[dict]:
    """Detect line segments using probabilistic Hough transform."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180, threshold,
        minLineLength=min_length, maxLineGap=max_gap,
    )
    if lines is None:
        return []
    return [
        {"x1": int(l[0][0]), "y1": int(l[0][1]), "x2": int(l[0][2]), "y2": int(l[0][3])}
        for l in lines
    ]


def process_detect_circles(
    frame: np.ndarray,
    min_radius: int = 0,
    max_radius: int = 0,
    param1: float = 100.0,
    param2: float = 30.0,
) -> list[dict]:
    """Detect circles using Hough gradient transform."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, 1, 20,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return []
    return [
        {"center": {"x": int(c[0]), "y": int(c[1])}, "radius": int(c[2])}
        for c in circles[0]
    ]


def process_detect_contours(
    frame: np.ndarray,
    min_area: float = 100.0,
) -> list[dict]:
    """Detect contours using adaptive thresholding."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2,
    )
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv.boundingRect(cnt)
        points = [{"x": int(p[0][0]), "y": int(p[0][1])} for p in approx]
        results.append({
            "points": points,
            "area": float(area),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        })
    return results


def process_detect_motion(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
) -> list[dict]:
    """Detect motion regions by frame differencing.

    *prev_frame* should be a grayscale image (the previous frame).
    Returns an empty list when *prev_frame* is ``None`` (baseline).
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if prev_frame is None:
        return []
    diff = cv.absdiff(gray, prev_frame)
    _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    thresh = cv.dilate(thresh, None, iterations=2)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        results.append({
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "area": float(area),
        })
    return results


def process_detect_qr_codes(frame: np.ndarray) -> list[dict]:
    """Detect and decode QR codes in *frame*."""
    detector = cv.QRCodeDetector()
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)
    if not retval or points is None:
        return []
    results = []
    for i, data in enumerate(decoded_info):
        if not data:
            continue
        corners = [{"x": float(p[0]), "y": float(p[1])} for p in points[i]]
        results.append({"data": data, "corners": corners})
    return results


def process_apply_transform(
    frame: np.ndarray,
    operation: str,
    params: dict | None = None,
) -> np.ndarray:
    """Apply a named image transformation to *frame*."""
    if params is None:
        params = {}

    if operation == "rotate":
        angle = params.get("angle", 90)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, float(angle), 1.0)
        return cv.warpAffine(frame, M, (w, h))

    elif operation == "scale":
        factor = params.get("factor", 0.5)
        return cv.resize(frame, None, fx=float(factor), fy=float(factor))

    elif operation == "threshold":
        value = params.get("value", 127)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, result = cv.threshold(gray, int(value), 255, cv.THRESH_BINARY)
        return cv.cvtColor(result, cv.COLOR_GRAY2BGR)

    elif operation == "canny":
        low = params.get("low", 50)
        high = params.get("high", 150)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, int(low), int(high))
        return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    elif operation == "blur":
        kernel_size = params.get("kernel_size", 5)
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1
        return cv.GaussianBlur(frame, (k, k), 0)

    else:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            "Supported: rotate, scale, threshold, canny, blur"
        )
