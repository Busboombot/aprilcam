from __future__ import annotations

import argparse
import math
import time
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from .config import AppConfig
from .iohelpers import get_data_dir
from .camutil import list_cameras as _list_cameras, select_camera_by_pattern
from .playfield import Playfield
from .models import AprilTag as AprilTagModel
from .display import PlayfieldDisplay


class AprilCam:
    def __init__(
        self,
        index: int,
        backend: Optional[int],
        speed_alpha: float,
        family: str,
        proc_width: int,
        cap_width: Optional[int] = None,
        cap_height: Optional[int] = None,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.0,
        corner_refine: str = "subpix",
        detect_inverted: bool = True,
        detect_interval: int = 1,
        use_clahe: bool = False,
        use_sharpen: bool = False,
        april_min_wb_diff: float = 3.0,
        april_min_cluster_pixels: int = 5,
        april_max_line_fit_mse: float = 20.0,
        print_tags: bool = False,
        cap: Optional[cv.VideoCapture] = None,
        homography: Optional[np.ndarray] = None,
        headless: bool = False,
        deskew_overlay: bool = False,
        playfield_poly_init: Optional[np.ndarray] = None,
    ) -> None:
        self.index = index
        self.backend = backend
        self.speed_alpha = float(speed_alpha)
        self.family = family
        self.proc_width = int(proc_width)
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.quad_decimate = float(quad_decimate)
        self.quad_sigma = float(quad_sigma)
        self.corner_refine = corner_refine
        self.detect_inverted = bool(detect_inverted)
        self.detect_interval = int(detect_interval)
        self.use_clahe = bool(use_clahe)
        self.use_sharpen = bool(use_sharpen)
        self.april_min_wb_diff = float(april_min_wb_diff)
        self.april_min_cluster_pixels = int(april_min_cluster_pixels)
        self.april_max_line_fit_mse = float(april_max_line_fit_mse)
        self.print_tags = bool(print_tags)
        self.cap = cap
        self.homography = homography
        self.headless = bool(headless)
        self.deskew_overlay = bool(deskew_overlay)
        self.play_poly: Optional[np.ndarray] = None
        if playfield_poly_init is not None and isinstance(playfield_poly_init, np.ndarray) and playfield_poly_init.shape == (4, 2):
            self.play_poly = playfield_poly_init.astype(np.float32)

        # Cached state
        self.detectors = self._build_detectors()
        self.playfield = Playfield(proc_width=self.proc_width or 960, detect_inverted=False)
        self.window = "aprilcam"
        self.arrow_scale, self.arrow_min, self.arrow_max = 0.25, 10, 125
        self.arrow_color, self.arrow_width = (255, 0, 255), 2
        self.M_deskew: Optional[np.ndarray] = None
        self.deskew_size: Optional[Tuple[int, int]] = None  # (w,h)
        self.display = PlayfieldDisplay(
            self.playfield,
            window_name=self.window,
            headless=self.headless,
            deskew_overlay=self.deskew_overlay,
        )

    # ---------- helpers ----------
    @staticmethod
    def draw_text_with_outline(
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.7,
        thickness: int = 1,
    ) -> None:
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv.LINE_AA)

    @staticmethod
    def draw_text_centered_with_outline(
        img: np.ndarray,
        text: str,
        center: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.8,
        thickness: int = 2,
    ) -> None:
        (tw, th), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = int(center[0] - tw / 2)
        y = int(center[1] + th / 2)
        AprilCam.draw_text_with_outline(img, text, (x, y), color=color, font_scale=font_scale, thickness=thickness)

    @staticmethod
    def _get_dict_by_family(name: str):
        m = {
            "16h5": cv.aruco.DICT_APRILTAG_16h5,
            "25h9": cv.aruco.DICT_APRILTAG_25h9,
            "36h10": cv.aruco.DICT_APRILTAG_36h10,
            "36h11": cv.aruco.DICT_APRILTAG_36h11,
        }
        return m.get(name, cv.aruco.DICT_APRILTAG_36h11)

    def _build_detectors(self):
        fams = [self.family] if self.family != "all" else ["16h5", "25h9", "36h10", "36h11"]
        detectors = []
        for f in fams:
            d = cv.aruco.getPredefinedDictionary(self._get_dict_by_family(f))
            p = cv.aruco.DetectorParameters()
            p.cornerRefinementMethod = {
                "none": cv.aruco.CORNER_REFINE_NONE,
                "contour": cv.aruco.CORNER_REFINE_CONTOUR,
                "subpix": cv.aruco.CORNER_REFINE_SUBPIX,
            }.get(self.corner_refine, cv.aruco.CORNER_REFINE_SUBPIX)
            p.aprilTagQuadDecimate = float(max(1.0, self.quad_decimate))
            p.aprilTagQuadSigma = float(max(0.0, self.quad_sigma))
            if hasattr(p, "aprilTagMinWhiteBlackDiff"):
                try:
                    p.aprilTagMinWhiteBlackDiff = int(max(0, int(round(self.april_min_wb_diff))))
                except Exception:
                    p.aprilTagMinWhiteBlackDiff = 3
            if hasattr(p, "aprilTagMinClusterPixels"):
                try:
                    p.aprilTagMinClusterPixels = int(max(1, int(self.april_min_cluster_pixels)))
                except Exception:
                    p.aprilTagMinClusterPixels = 5
            if hasattr(p, "aprilTagMaxLineFitMse"):
                try:
                    p.aprilTagMaxLineFitMse = float(max(1.0, float(self.april_max_line_fit_mse)))
                except Exception:
                    p.aprilTagMaxLineFitMse = 20.0
            p.detectInvertedMarker = bool(self.detect_inverted)
            detectors.append((d, p))
        return detectors

    @staticmethod
    def _maybe_preprocess(gray: np.ndarray, use_clahe: bool, use_sharpen: bool) -> np.ndarray:
        out = gray
        if use_clahe:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = clahe.apply(out)
        if use_sharpen:
            k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            out = cv.filter2D(out, -1, k)
        return out

    def detect_apriltags(self, frame_bgr: np.ndarray, scale: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        h, w = frame_bgr.shape[:2]
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            gray = cv.resize(gray, (new_w, new_h), interpolation=cv.INTER_AREA)
        gray = self._maybe_preprocess(gray, self.use_clahe, self.use_sharpen)

        detections: List[Tuple[np.ndarray, np.ndarray, int]] = []
        for d, p in self.detectors:
            detector = cv.aruco.ArucoDetector(d, p)
            corners, ids, _rej = detector.detectMarkers(gray)
            if ids is None:
                continue
            for c, idv in zip(corners, ids.flatten().tolist()):
                pts = c.reshape(-1, 2).astype(np.float32)
                if scale < 1.0:
                    pts = pts / float(scale)
                detections.append((pts, pts.copy(), int(idv)))
        return detections

    @staticmethod
    def lk_track(prev_gray: np.ndarray, gray: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3,
                                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        if p1 is None or st is None or int(st.sum()) < 4:
            return None
        return p1.reshape(-1, 2)

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Tuple[np.ndarray, np.ndarray, int]]):
        for pts, _raw, tag_id in detections:
            ptsf = pts.astype(np.float32)
            p0, p1, p2, p3 = ptsf[0], ptsf[1], ptsf[2], ptsf[3]
            cv.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2, cv.LINE_AA)
            cv.line(frame, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 0, 255), 2, cv.LINE_AA)
            cv.line(frame, (int(p3[0]), int(p3[1])), (int(p0[0]), int(p0[1])), (0, 0, 255), 2, cv.LINE_AA)
            center = ptsf.mean(axis=0)
            top_mid = (p0 + p1) / 2.0
            h_left = float(np.linalg.norm(p0 - p3))
            h_right = float(np.linalg.norm(p1 - p2))
            tag_h = max(1.0, 0.5 * (h_left + h_right))
            hat_h = 0.30 * tag_h
            n = top_mid - center
            n_norm = float(np.linalg.norm(n))
            if n_norm < 1e-6:
                e = p1 - p0
                n = np.array([-e[1], e[0]], dtype=np.float32)
                n_norm = float(np.linalg.norm(n))
            if n_norm > 1e-6:
                n_unit = n / n_norm
                peak = top_mid + n_unit * hat_h
                cv.line(frame, (int(p0[0]), int(p0[1])), (int(peak[0]), int(peak[1])), (0, 255, 0), 2, cv.LINE_AA)
                cv.line(frame, (int(peak[0]), int(peak[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2, cv.LINE_AA)
            else:
                cv.line(frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2, cv.LINE_AA)

    @staticmethod
    def draw_velocity_vectors(
        frame: np.ndarray,
        detections: List[Tuple[np.ndarray, np.ndarray, int]],
        vel_dirs: dict[int, Tuple[float, float]],
        speeds: dict[int, float],
        scale: float = 0.5,
        min_len: int = 20,
        max_len: int = 250,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 3,
    ) -> None:
        for pts, _raw, tag_id in detections:
            if tag_id not in vel_dirs:
                continue
            vx, vy = vel_dirs[tag_id]
            norm = math.hypot(vx, vy)
            if norm < 1e-6:
                continue
            dir_unit = (vx / norm, vy / norm)
            c = pts.astype(np.float32).mean(axis=0)
            start = (int(c[0]), int(c[1]))
            speed = float(speeds.get(tag_id, 0.0))
            length_px = int(max(min_len, min(max_len, speed * scale)))
            end = (int(c[0] + dir_unit[0] * length_px), int(c[1] + dir_unit[1] * length_px))
            cv.arrowedLine(frame, start, end, color, thickness, tipLength=0.12)
            cv.circle(frame, start, 4, color, -1)
            AprilCam.draw_text_centered_with_outline(frame, f"{int(tag_id)}", start, color=(0, 0, 255), font_scale=0.9, thickness=2)

    # ---------- core loop ----------
    def _init_capture(self) -> Optional[cv.VideoCapture]:
        if self.cap is None:
            self.cap = cv.VideoCapture(int(self.index), 0 if self.backend is None else int(self.backend))
        if not self.cap or not self.cap.isOpened():
            print("Failed to open camera.")
            return None
        if self.cap_width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, int(self.cap_width))
        if self.cap_height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(self.cap_height))
        return self.cap

    def _init_window(self) -> None:
        if self.headless:
            return
        cv.namedWindow(self.window, cv.WINDOW_NORMAL)
        target_w, target_h = 1280, 720
        try:
            cap = self.cap
            if cap is not None and hasattr(cap, "get_bbox") and hasattr(cap, "get_display_rect"):
                cap_l, cap_t, cap_w, cap_h = cap.get_bbox()  # type: ignore[attr-defined]
                mon_l, mon_t, mon_w, mon_h = cap.get_display_rect()  # type: ignore[attr-defined]
                mon_r, mon_b = mon_l + mon_w, mon_t + mon_h
                cap_r, cap_b = cap_l + cap_w, cap_t + cap_h
                pad = 12
                candidates = []
                if cap_r + pad < mon_r:
                    candidates.append((cap_r + pad, cap_t, mon_r - (cap_r + pad), min(cap_h, mon_b - cap_t)))
                if mon_l < cap_l - pad:
                    candidates.append((mon_l, cap_t, (cap_l - pad) - mon_l, min(cap_h, mon_b - cap_t)))
                if cap_b + pad < mon_b:
                    candidates.append((cap_l, cap_b + pad, min(cap_w, mon_r - cap_l), mon_b - (cap_b + pad)))
                if mon_t < cap_t - pad:
                    candidates.append((cap_l, mon_t, min(cap_w, mon_r - cap_l), (cap_t - pad) - mon_t))
                if candidates:
                    bx, by, bw, bh = max(candidates, key=lambda r: r[2] * r[3])
                    target_w = max(320, min(1280, bw))
                    target_h = max(180, min(720, bh))
                    cv.resizeWindow(self.window, int(target_w), int(target_h))
                    cv.moveWindow(self.window, int(bx), int(by))
                else:
                    cv.resizeWindow(self.window, target_w, target_h)
            else:
                cv.resizeWindow(self.window, target_w, target_h)
        except Exception:
            cv.resizeWindow(self.window, target_w, target_h)

    def _update_playfield(self, frame: np.ndarray) -> None:
        if self.play_poly is None:
            try:
                self.playfield.update(frame)
                poly = self.playfield.get_polygon()
                if poly is not None:
                    self.play_poly = poly.astype(np.float32)
            except Exception:
                pass
        # Cache deskew transform once if requested
        if self.deskew_overlay and self.play_poly is not None and self.M_deskew is None:
            UL, UR, LR, LL = self.play_poly.astype(np.float32)
            w_top = float(np.linalg.norm(UR - UL))
            w_bottom = float(np.linalg.norm(LR - LL))
            h_left = float(np.linalg.norm(LL - UL))
            h_right = float(np.linalg.norm(LR - UR))
            out_w = max(10, int(round(max(w_top, w_bottom))))
            out_h = max(10, int(round(max(h_left, h_right))))
            src = np.array([UL, UR, LR, LL], dtype=np.float32)
            dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
            self.M_deskew = cv.getPerspectiveTransform(src, dst)
            self.deskew_size = (out_w, out_h)

    def _overlay_world_coords(self, frame: np.ndarray, detections: List[Tuple[np.ndarray, np.ndarray, int]]) -> None:
        if self.homography is None or not detections:
            return
        for pts, _raw, _tid in detections:
            ptsf = pts.astype(np.float32)
            c = ptsf.mean(axis=0)
            u, v = float(c[0]), float(c[1])
            vec = np.array([u, v, 1.0], dtype=float)
            Xw = self.homography @ vec
            if abs(Xw[2]) > 1e-6:
                Xcm = Xw[0] / Xw[2]
                Ycm = Xw[1] / Xw[2]
                text = f"{Xcm:.1f},{Ycm:.1f}"
                x_coords = ptsf[:, 0]
                y_coords = ptsf[:, 1]
                xmin = float(x_coords.min())
                xmax = float(x_coords.max())
                ymin = float(y_coords.min())
                ymax = float(y_coords.max())
                fw = frame.shape[1]
                fh = frame.shape[0]
                pad = 8
                (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                placed = False
                # Below
                tx = int((xmin + xmax) * 0.5 - tw * 0.5)
                ty = int(ymax + pad + th)
                if ty + base <= fh and tx >= 0 and (tx + tw) <= fw:
                    AprilCam.draw_text_with_outline(frame, text, (tx, ty), color=(0, 255, 0), font_scale=0.5, thickness=1)
                    placed = True
                if not placed:
                    # Above
                    tx = int((xmin + xmax) * 0.5 - tw * 0.5)
                    ty = int(ymin - pad)
                    if ty - th >= 0 and tx >= 0 and (tx + tw) <= fw:
                        AprilCam.draw_text_with_outline(frame, text, (tx, ty), color=(0, 255, 0), font_scale=0.5, thickness=1)
                        placed = True
                if not placed:
                    # Right
                    tx = int(xmax + pad)
                    ty = int((ymin + ymax) * 0.5 + th * 0.5)
                    if (tx + tw) <= fw and ty + base <= fh and (ty - th) >= 0:
                        AprilCam.draw_text_with_outline(frame, text, (tx, ty), color=(0, 255, 0), font_scale=0.5, thickness=1)
                        placed = True
                if not placed:
                    # Left
                    tx = int(xmin - pad - tw)
                    ty = int((ymin + ymax) * 0.5 + th * 0.5)
                    if tx >= 0 and ty + base <= fh and (ty - th) >= 0:
                        AprilCam.draw_text_with_outline(frame, text, (tx, ty), color=(0, 255, 0), font_scale=0.5, thickness=1)
                        placed = True
                if not placed:
                    AprilCam.draw_text_with_outline(frame, text, (int(u) + 8, int(v) + 14), color=(0, 255, 0), font_scale=0.5, thickness=1)

    def _prepare_display(self, frame: np.ndarray) -> np.ndarray:
        display = frame
        if self.play_poly is not None and self.play_poly.shape == (4, 2):
            try:
                if self.deskew_overlay and self.M_deskew is not None and self.deskew_size is not None:
                    w, h = self.deskew_size
                    display = cv.warpPerspective(frame, self.M_deskew, (w, h))
                else:
                    PAD = 24
                    x_coords = self.play_poly[:, 0]
                    y_coords = self.play_poly[:, 1]
                    xmin = max(0, int(math.floor(float(x_coords.min()) - PAD)))
                    ymin = max(0, int(math.floor(float(y_coords.min()) - PAD)))
                    xmax = min(frame.shape[1], int(math.ceil(float(x_coords.max()) + PAD)))
                    ymax = min(frame.shape[0], int(math.ceil(float(y_coords.max()) + PAD)))
                    if xmax > xmin and ymax > ymin:
                        display = frame[ymin:ymax, xmin:xmax]
            except Exception:
                display = frame
        return display

    def _to_models(self, detections: List[Tuple[np.ndarray, np.ndarray, int]], ts: float) -> List[AprilTagModel]:
        out: List[AprilTagModel] = []
        for pts, _raw, tid in detections:
            tag = AprilTagModel.from_corners(tid, pts, homography=self.homography, timestamp=ts)
            out.append(tag)
        return out

    def run(self) -> None:
        cap = self._init_capture()
        if cap is None:
            return
        # Window is managed by PlayfieldDisplay

        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0
        tracks: dict[int, np.ndarray] = {}
        vel_ema: dict[int, float] = {}
        last_seen: dict[int, Tuple[float, float, float]] = {}
        paused = False
        last_display: Optional[np.ndarray] = None
        tag_models: dict[int, AprilTagModel] = {}

        try:
            while True:
                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        print("Camera read failed.")
                        break

                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    now = time.monotonic()
                    detections: List[Tuple[np.ndarray, np.ndarray, int]] = []
                    if self.detect_interval <= 1 or frame_idx % max(1, self.detect_interval) == 0 or prev_gray is None or not tracks:
                        w = frame.shape[1]
                        scale = min(1.0, float(self.proc_width) / float(w)) if (self.proc_width and self.proc_width > 0 and w > 0) else 1.0
                        detections = self.detect_apriltags(frame, scale=scale)
                        tracks = {tid: pts for (pts, _raw, tid) in detections}
                    else:
                        new_tracks: dict[int, np.ndarray] = {}
                        for tid, pts in tracks.items():
                            new_pts = AprilCam.lk_track(prev_gray, gray, pts)
                            if new_pts is not None:
                                new_tracks[tid] = new_pts
                                detections.append((new_pts, new_pts, tid))
                        tracks = new_tracks
                        if len(detections) == 0:
                            w = frame.shape[1]
                            scale = min(1.0, float(self.proc_width) / float(w)) if (self.proc_width and self.proc_width > 0 and w > 0) else 1.0
                            detections = self.detect_apriltags(frame, scale=scale)
                            tracks = {tid: pts for (pts, _raw, tid) in detections}

                    # Update display/playfield and cache deskew transform
                    self._update_playfield(frame)

                    # Filter detections to inside the playfield if defined
                    if self.play_poly is not None and self.play_poly.shape == (4, 2):
                        in_dets: List[Tuple[np.ndarray, np.ndarray, int]] = []
                        for pts, raw, tid in detections:
                            c = pts.astype(np.float32).mean(axis=0)
                            inside = cv.pointPolygonTest(self.play_poly.astype(np.float32), (float(c[0]), float(c[1])), False)
                            if inside >= 0:
                                in_dets.append((pts, raw, tid))
                        detections = in_dets
                        tracks = {tid: pts for (pts, _raw, tid) in detections}

                    # Build/update models and let display draw coords later
                    tags: List[AprilTagModel] = []
                    for pts, _raw, tid in detections:
                        if tid in tag_models:
                            tag_models[tid].update(pts, timestamp=now, homography=self.homography)
                        else:
                            tag_models[tid] = AprilTagModel.from_corners(tid, pts, homography=self.homography, timestamp=now)
                        tags.append(tag_models[tid])
                    # Prune stale models not seen this frame
                    seen_ids = {tid for _pts, _r, tid in detections}
                    for tid in list(tag_models.keys()):
                        if tid not in seen_ids and tag_models[tid].last_ts is not None and (now - float(tag_models[tid].last_ts)) > 1.5:
                            del tag_models[tid]

                    # Velocities (for printing only)
                    speeds: dict[int, float] = {}
                    vel_dirs: dict[int, Tuple[float, float]] = {}
                    for pts, _raw, tag_id in detections:
                        c = pts.mean(axis=0)
                        cx, cy = float(c[0]), float(c[1])
                        if tag_id in last_seen:
                            px, py, pt = last_seen[tag_id]
                            dt = max(1e-3, now - pt)
                            dx, dy = (cx - px), (cy - py)
                            inst = math.hypot(dx, dy) / dt
                            vel_dirs[tag_id] = (dx / dt, dy / dt)
                            prev = vel_ema.get(tag_id)
                            ema = (self.speed_alpha * inst + (1 - self.speed_alpha) * prev) if prev is not None else inst
                            vel_ema[tag_id] = ema
                            speeds[tag_id] = ema
                        last_seen[tag_id] = (cx, cy, now)

                    if self.print_tags and detections:
                        lines = []
                        H = self.homography
                        for pts, _raw, tag_id in detections:
                            ptsf = pts.astype(np.float32)
                            center = ptsf.mean(axis=0)
                            wtxt = ""
                            if H is not None:
                                u, v = float(center[0]), float(center[1])
                                vec = np.array([u, v, 1.0], dtype=float)
                                Xw = H @ vec
                                if abs(Xw[2]) > 1e-6:
                                    Xcm = Xw[0] / Xw[2]
                                    Ycm = Xw[1] / Xw[2]
                                    wtxt = f"  WX:{Xcm:7.1f}cm  WY:{Ycm:7.1f}cm"
                            top_mid = (ptsf[0] + ptsf[1]) / 2.0
                            dir_vec = top_mid - center
                            ori_ang = math.degrees(math.atan2(float(dir_vec[1]), float(dir_vec[0]))) if np.linalg.norm(dir_vec) > 1e-6 else 0.0
                            v = float(speeds.get(tag_id, 0.0))
                            vx, vy = vel_dirs.get(tag_id, (0.0, 0.0))
                            vang = math.degrees(math.atan2(vy, vx)) if (vx != 0.0 or vy != 0.0) else 0.0
                            line = f"ID:{tag_id:4d}  CX:{int(center[0]):4d}  CY:{int(center[1]):4d}  ORI:{ori_ang:+7.1f}°  V:{v:7.1f} px/s  VANG:{vang:+7.1f}°{wtxt}"
                            lines.append(line)
                        print("\n".join(lines))

                    # Update display output and draw overlays using models
                    display = self.display.update(frame)
                    self.display.draw_overlays(display if display is not None else frame, tags, homography=self.homography)
                    last_display = display.copy()
                else:
                    if last_display is None:
                        ok, frame = cap.read()
                        if not ok:
                            print("Camera read failed.")
                            break
                        last_display = frame.copy()
                    display = last_display.copy()
                    if not self.headless:
                        AprilCam.draw_text_with_outline(display, " Paused: Press Space to Run", (10, 30), color=(0, 255, 255), font_scale=0.9, thickness=2)

                if not self.headless:
                    self.display.show(display)
                    key = cv.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                    if key == ord(' '):
                        paused = not paused
                        continue
                else:
                    time.sleep(0.001)
                if not paused:
                    prev_gray = gray
                    frame_idx += 1
        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            try:
                if not self.headless:
                    cv.destroyAllWindows()
            except Exception:
                pass


def save_last_camera(idx: int):
    try:
        p = get_data_dir() / "last_camera"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(int(idx)))
    except Exception:
        pass


def load_last_camera() -> Optional[int]:
    try:
        p = get_data_dir() / "last_camera"
        if p.exists():
            return int(p.read_text().strip())
    except Exception:
        return None
    return None


def run_video(
    index: int,
    backend: Optional[int],
    speed_alpha: float,
    family: str,
    proc_width: int,
    cap_width: Optional[int] = None,
    cap_height: Optional[int] = None,
    quad_decimate: float = 1.0,
    quad_sigma: float = 0.0,
    corner_refine: str = "subpix",
    detect_inverted: bool = True,
    use_aruco3: bool = False,  # unused placeholder to keep CLI stable
    detect_interval: int = 1,
    use_clahe: bool = False,
    use_sharpen: bool = False,
    april_min_wb_diff: float = 3.0,
    april_min_cluster_pixels: int = 5,
    april_max_line_fit_mse: float = 20.0,
    print_tags: bool = False,
    cap: Optional[cv.VideoCapture] = None,
    homography: Optional[np.ndarray] = None,
    headless: bool = False,
    deskew_overlay: bool = False,
    playfield_poly_init: Optional[np.ndarray] = None,
) -> None:
    app = AprilCam(
        index=index,
        backend=backend,
        speed_alpha=speed_alpha,
        family=family,
        proc_width=proc_width,
        cap_width=cap_width,
        cap_height=cap_height,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        corner_refine=corner_refine,
        detect_inverted=detect_inverted,
        detect_interval=detect_interval,
        use_clahe=use_clahe,
        use_sharpen=use_sharpen,
        april_min_wb_diff=april_min_wb_diff,
        april_min_cluster_pixels=april_min_cluster_pixels,
        april_max_line_fit_mse=april_max_line_fit_mse,
        print_tags=print_tags,
        cap=cap,
        homography=homography,
        headless=headless,
        deskew_overlay=deskew_overlay,
        playfield_poly_init=playfield_poly_init,
    )
    app.run()


# CLI main moved to aprilcam.cli.aprilcam_cli


# --- Back-compat wrappers for CLI offline evaluation ---
def build_detectors(
    family: str,
    corner_refine: str,
    quad_decimate: float,
    quad_sigma: float,
    detect_inverted: bool,
    april_min_wb_diff: float = 3.0,
    april_min_cluster_pixels: int = 5,
    april_max_line_fit_mse: float = 20.0,
):
    """Return a list of (dictionary, parameters) configured for the requested family/families.

    Mirrors AprilCam._build_detectors but exposed at module scope for CLI use.
    """
    fams = [family] if family != "all" else ["16h5", "25h9", "36h10", "36h11"]
    detectors = []
    for f in fams:
        d = cv.aruco.getPredefinedDictionary(AprilCam._get_dict_by_family(f))
        p = cv.aruco.DetectorParameters()
        p.cornerRefinementMethod = {
            "none": cv.aruco.CORNER_REFINE_NONE,
            "contour": cv.aruco.CORNER_REFINE_CONTOUR,
            "subpix": cv.aruco.CORNER_REFINE_SUBPIX,
        }.get(corner_refine, cv.aruco.CORNER_REFINE_SUBPIX)
        p.aprilTagQuadDecimate = float(max(1.0, quad_decimate))
        p.aprilTagQuadSigma = float(max(0.0, quad_sigma))
        if hasattr(p, "aprilTagMinWhiteBlackDiff"):
            try:
                p.aprilTagMinWhiteBlackDiff = int(max(0, int(round(april_min_wb_diff))))
            except Exception:
                p.aprilTagMinWhiteBlackDiff = 3
        if hasattr(p, "aprilTagMinClusterPixels"):
            try:
                p.aprilTagMinClusterPixels = int(max(1, int(april_min_cluster_pixels)))
            except Exception:
                p.aprilTagMinClusterPixels = 5
        if hasattr(p, "aprilTagMaxLineFitMse"):
            try:
                p.aprilTagMaxLineFitMse = float(max(1.0, float(april_max_line_fit_mse)))
            except Exception:
                p.aprilTagMaxLineFitMse = 20.0
        p.detectInvertedMarker = bool(detect_inverted)
        detectors.append((d, p))
    return detectors


def detect_apriltags(
    frame_bgr: np.ndarray,
    detectors,
    scale: float = 1.0,
    clahe: bool = False,
    sharpen: bool = False,
):
    """Detect AprilTags in an image using provided detectors.

    Returns a list of tuples: (pts[4x2], raw_pts[4x2], id)
    """
    h, w = frame_bgr.shape[:2]
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        gray = cv.resize(gray, (new_w, new_h), interpolation=cv.INTER_AREA)
    gray = AprilCam._maybe_preprocess(gray, clahe, sharpen)

    detections: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for d, p in detectors:
        detector = cv.aruco.ArucoDetector(d, p)
        corners, ids, _rej = detector.detectMarkers(gray)
        if ids is None:
            continue
        for c, idv in zip(corners, ids.flatten().tolist()):
            pts = c.reshape(-1, 2).astype(np.float32)
            if scale < 1.0:
                pts = pts / float(scale)
            detections.append((pts, pts.copy(), int(idv)))
    return detections
