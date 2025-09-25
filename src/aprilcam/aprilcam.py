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
        """Initialize the AprilCam controller.

        Args:
            index: Camera index (ignored when an explicit cap is provided).
            backend: Preferred OpenCV capture backend constant or None for auto.
            speed_alpha: EMA smoothing factor for printed speeds in [0,1].
            family: AprilTag family name (or 'all').
            proc_width: Processing width for detection downscale (0 disables).
            cap_width: Optional capture width hint for the camera.
            cap_height: Optional capture height hint for the camera.
            quad_decimate: AprilTag decimation (>=1, larger is faster/rougher).
            quad_sigma: AprilTag Gaussian blur sigma in pixels.
            corner_refine: Corner refinement mode: none/contour/subpix.
            detect_inverted: Whether to also detect inverted (white-on-black) tags.
            detect_interval: Detect every N frames; track between.
            use_clahe: Apply CLAHE preprocessing before detection.
            use_sharpen: Apply light sharpening before detection.
            april_min_wb_diff: Min white/black intensity diff for AprilTag.
            april_min_cluster_pixels: Min cluster pixel size for AprilTag.
            april_max_line_fit_mse: Max line fit MSE for AprilTag.
            print_tags: Print per-tag info each frame when detections exist.
            cap: Optional pre-opened cv.VideoCapture or image source.
            homography: 3x3 projective transform to world coords (cm).
            headless: If True, never opens a window.
            deskew_overlay: If True, warp the playfield to a rectangle for display.
            playfield_poly_init: Optional initial 4x2 polygon for playfield.
        """
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
        self.display = PlayfieldDisplay(
            self.playfield,
            window_name=self.window,
            headless=self.headless,
            deskew_overlay=self.deskew_overlay,
        )



    @staticmethod
    def _get_dict_by_family(name: str):
        """Map family string to OpenCV ArUco predefined AprilTag dictionary."""
        m = {
            "16h5": cv.aruco.DICT_APRILTAG_16h5,
            "25h9": cv.aruco.DICT_APRILTAG_25h9,
            "36h10": cv.aruco.DICT_APRILTAG_36h10,
            "36h11": cv.aruco.DICT_APRILTAG_36h11,
        }
        return m.get(name, cv.aruco.DICT_APRILTAG_36h11)

    def _build_detectors(self):
        """Create per-family ArUco detectors configured with AprilTag params."""
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
        """Optionally apply CLAHE and/or sharpening to a grayscale image."""
        out = gray
        if use_clahe:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = clahe.apply(out)
        if use_sharpen:
            k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            out = cv.filter2D(out, -1, k)
        return out

    def detect_apriltags(self, frame_bgr: np.ndarray, scale: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Detect AprilTags in a BGR frame.

        Args:
            frame_bgr: Input color frame in BGR order.
            scale: Optional downscale factor for speed (<1 downscales).

        Returns:
            A list of (pts[4x2], raw_pts[4x2], id) for each detected tag.
        """
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
        """Track points using pyramidal Lucas–Kanade optical flow.

        Returns the new points (Nx2) in the current frame or None on failure.
        """
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3,
                                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        if p1 is None or st is None or int(st.sum()) < 4:
            return None
        return p1.reshape(-1, 2)


    # ---------- core loop ----------
    def _init_capture(self) -> Optional[cv.VideoCapture]:
        """Open the capture device or use the provided cap; apply size hints."""
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

    def _update_playfield(self, frame: np.ndarray) -> None:
        """Update cached playfield polygon via Playfield.

        Deskew is handled by PlayfieldDisplay; this only updates geometry.
        """
        try:
            self.playfield.update(frame)
            poly = self.playfield.get_polygon()
            if poly is not None:
                self.play_poly = poly.astype(np.float32)
        except Exception:
            pass



    def run(self) -> None:
        """Main capture/detect/track loop with display and overlays."""
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
                    # 1) Read next frame
                    ok, frame = cap.read()
                    if not ok:
                        print("Camera read failed.")
                        break

                    # 2) Convert to gray and perform detection or faster LK tracking
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    now = time.monotonic()
                    detections: List[Tuple[np.ndarray, np.ndarray, int]] = []
                    if self.detect_interval <= 1 or frame_idx % max(1, self.detect_interval) == 0 or prev_gray is None or not tracks:
                        w = frame.shape[1]
                        scale = min(1.0, float(self.proc_width) / float(w)) if (self.proc_width and self.proc_width > 0 and w > 0) else 1.0
                        detections = self.detect_apriltags(frame, scale=scale)
                        tracks = {tid: pts for (pts, _raw, tid) in detections}
                    else:
                        # Track existing tag corners forward with LK; fall back to detection on loss
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

                    # 3) Update Playfield cache (polygon) for cropping/deskew
                    self._update_playfield(frame)

                    # 4) Keep only detections inside the current playfield polygon
                    if detections:
                        in_dets: List[Tuple[np.ndarray, np.ndarray, int]] = []
                        for pts, raw, tid in detections:
                            if self.playfield.isIn(pts):
                                in_dets.append((pts, raw, tid))
                        detections = in_dets
                        tracks = {tid: pts for (pts, _raw, tid) in detections}

                    # 5) Update/maintain tag models and playfield flows
                    tags: List[AprilTagModel] = []
                    for pts, _raw, tid in detections:
                        if tid in tag_models:
                            tag_models[tid].update(pts, timestamp=now, homography=self.homography)
                        else:
                            tag_models[tid] = AprilTagModel.from_corners(tid, pts, homography=self.homography, timestamp=now, frame=frame_idx)
                        # Feed into playfield flow map (sets in_playfield)
                        tag_models[tid].frame = frame_idx
                        self.playfield.add_tag(tag_models[tid])
                        tags.append(tag_models[tid])
                    # Prune models not seen recently
                    seen_ids = {tid for _pts, _r, tid in detections}
                    for tid in list(tag_models.keys()):
                        if tid not in seen_ids and tag_models[tid].last_ts is not None and (now - float(tag_models[tid].last_ts)) > 1.5:
                            del tag_models[tid]

                    # 6) Compute per-tag speeds for printing (UI overlay uses models)
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

                    # 7) Optional logging to stdout
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

                    # 8) Prepare display image and draw overlays
                    display = self.display.update(frame)
                    # Use most recent tag states from flows for overlay
                    flows = self.playfield.get_flows()
                    tags_for_overlay = list(flows.values())
                    self.display.draw_overlays(display if display is not None else frame, tags_for_overlay, homography=self.homography)
                    last_display = display.copy()
                else:
                    # Paused branch: reuse last display buffer and show a pause overlay
                    if last_display is None:
                        ok, frame = cap.read()
                        if not ok:
                            print("Camera read failed.")
                            break
                        last_display = frame.copy()
                    display = last_display.copy()
                    if not self.headless:
                            self.display.pause(display)

                # 9) Present frame (if not headless) and process input
                if not self.headless:
                    self.display.show(display)
                    key = cv.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                    if key == ord(' '):
                        paused = not paused
                        continue
                else:
                    # Headless: small sleep to avoid tight loop
                    time.sleep(0.001)
                # 10) Bookkeeping for next iteration
                if not paused:
                    prev_gray = gray
                    frame_idx += 1
        finally:
            # 11) Cleanup resources
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
