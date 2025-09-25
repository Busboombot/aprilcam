from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import cv2 as cv
import numpy as np
from .models import AprilTag, AprilTagFlow


@dataclass
class Playfield:
    """Representation of the playfield extracted from video frames.

    This caches 4x4 ArUco corner detections and exposes a stable playfield polygon.
    Corner IDs are expected to be 0=UL, 1=UR, 2=LL, 3=LR in the rendered simulator.
    We compute a consistent UL,UR,LR,LL order by geometry to tolerate any ID swaps.
    """

    proc_width: int = 960
    detect_inverted: bool = False

    _poly: Optional[np.ndarray] = None  # shape (4,2) float32 in order UL,UR,LR,LL
    _flows: Dict[int, AprilTagFlow] = field(default_factory=dict)

    def _build_aruco4_detector(self):
        d = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        p = cv.aruco.DetectorParameters()
        p.detectInvertedMarker = bool(self.detect_inverted)
        return cv.aruco.ArucoDetector(d, p)

    def _detect_corners(self, frame_bgr: np.ndarray) -> Dict[int, Tuple[float, float]]:
        h, w = frame_bgr.shape[:2]
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        if self.proc_width and w > 0 and self.proc_width < w:
            scale = float(self.proc_width) / float(w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            gray = cv.resize(gray, (new_w, new_h), interpolation=cv.INTER_AREA)
        else:
            scale = 1.0

        detector = self._build_aruco4_detector()
        corners, ids, _ = detector.detectMarkers(gray)
        out: Dict[int, Tuple[float, float]] = {}
        if ids is None:
            return out
        for c, idv in zip(corners, ids.flatten().tolist()):
            pts = c.reshape(-1, 2).astype(np.float32)
            if scale < 1.0 and scale > 1e-9:
                pts = pts / float(scale)
            center = pts.mean(axis=0)
            out[int(idv)] = (float(center[0]), float(center[1]))
        return out

    def _order_poly(self, corners_map: Dict[int, Tuple[float, float]]) -> Optional[np.ndarray]:
        if not all(k in corners_map for k in (0, 1, 2, 3)):
            return None
        pts4 = np.array([
            corners_map[0],
            corners_map[1],
            corners_map[2],
            corners_map[3],
        ], dtype=np.float32)
        idx = np.argsort(pts4[:, 1])  # ascending by y (top first)
        top = pts4[idx[:2]]
        bot = pts4[idx[2:]]
        top = top[np.argsort(top[:, 0])]  # UL, UR
        bot = bot[np.argsort(bot[:, 0])]  # LL, LR
        UL, UR = top[0], top[1]
        LL, LR = bot[0], bot[1]
        return np.array([UL, UR, LR, LL], dtype=np.float32)

    def update(self, frame_bgr: np.ndarray) -> None:
        if self._poly is not None:
            return
        cmap = self._detect_corners(frame_bgr)
        poly = self._order_poly(cmap)
        if poly is not None:
            self._poly = poly

    def get_polygon(self) -> Optional[np.ndarray]:
        return self._poly.copy() if self._poly is not None else None

    def isIn(self, pts: np.ndarray | tuple[float, float]) -> bool:
        """Return True if the given tag points/center lie within the playfield.

        Accepts either:
        - An array of shape (N,2) of tag corners/points; uses their mean as center.
        - A tuple (x, y) representing the center directly.

        If the playfield polygon isn't known yet, returns True (no filtering).
        """
        if self._poly is None:
            return True
        try:
            if isinstance(pts, tuple) or (hasattr(pts, "__len__") and len(pts) == 2 and not hasattr(pts[0], "__len__")):
                u, v = float(pts[0]), float(pts[1])
            else:
                P = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                c = P.mean(axis=0)
                u, v = float(c[0]), float(c[1])
            inside = cv.pointPolygonTest(self._poly.astype(np.float32), (u, v), False)
            return bool(inside >= 0)
        except Exception:
            return True

    def annotate(self, frame_bgr: np.ndarray) -> None:
        if self._poly is None:
            return
        try:
            poly_i = self._poly.astype(int)
            cv.polylines(frame_bgr, [poly_i], True, (255, 255, 255), 2, cv.LINE_AA)
        except Exception:
            pass

    def deskew(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._poly is None:
            return frame_bgr
        UL, UR, LR, LL = self._poly.astype(np.float32)
        w_top = float(np.linalg.norm(UR - UL))
        w_bottom = float(np.linalg.norm(LR - LL))
        h_left = float(np.linalg.norm(LL - UL))
        h_right = float(np.linalg.norm(LR - UR))
        out_w = max(10, int(round(max(w_top, w_bottom))))
        out_h = max(10, int(round(max(h_left, h_right))))
        src = np.array([UL, UR, LR, LL], dtype=np.float32)
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        M = cv.getPerspectiveTransform(src, dst)
        return cv.warpPerspective(frame_bgr, M, (out_w, out_h))

    # --- tag flow integration ---
    def add_tag(self, tag: AprilTag) -> None:
        """Add/Update a tag into the playfield flows, setting in_playfield.

        If the playfield polygon is unknown, in_playfield defaults to True.
        """
        try:
            tag.in_playfield = self.isIn(tag.center_px)
        except Exception:
            tag.in_playfield = True
        flow = self._flows.get(tag.id)
        if flow is None:
            flow = AprilTagFlow(maxlen=5)
            self._flows[tag.id] = flow
        # Store a snapshot so history isn't mutated by future updates
        flow.add_tag(tag.clone())

    def get_flows(self) -> Dict[int, AprilTagFlow]:
        return self._flows

 
