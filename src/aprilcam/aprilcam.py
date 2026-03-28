from __future__ import annotations

import argparse
import math
import time
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from .config import AppConfig
from .detection import TagRecord
from .iohelpers import get_data_dir
from .camutil import list_cameras as _list_cameras, select_camera_by_pattern, diagnose_camera_failure
from .errors import CameraError, CameraNotFoundError, CameraInUseError
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
        use_highpass: bool = True,
        highpass_ksize: int = 51,
        april_min_wb_diff: float = 3.0,
        april_min_cluster_pixels: int = 5,
        april_max_line_fit_mse: float = 20.0,
        print_tags: bool = False,
        cap: Optional[cv.VideoCapture] = None,
        homography: Optional[np.ndarray] = None,
        headless: bool = False,
        deskew_overlay: bool = False,
        detect_aruco_4x4: bool = False,
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
        self.use_highpass = bool(use_highpass)
        self.highpass_ksize = int(highpass_ksize)
        self.april_min_wb_diff = float(april_min_wb_diff)
        self.april_min_cluster_pixels = int(april_min_cluster_pixels)
        self.april_max_line_fit_mse = float(april_max_line_fit_mse)
        self.detect_aruco_4x4 = bool(detect_aruco_4x4)
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
        self.playfield = Playfield(
            proc_width=self.proc_width or 960,
            detect_inverted=False,
            ema_alpha=self.speed_alpha,
        )
        self.window = "aprilcam"
        self.display = PlayfieldDisplay(
            self.playfield,
            window_name=self.window,
            headless=self.headless,
            deskew_overlay=self.deskew_overlay,
        )

        # Tracking state (initialized by reset_state)
        self._prev_gray: Optional[np.ndarray] = None
        self._tracks: dict[int, np.ndarray] = {}
        self._track_families: dict[int, str] = {}
        self._tag_models: dict[int, AprilTagModel] = {}
        self._frame_idx: int = 0

        # EMA state for TUI display smoothing
        self._ema: dict[int, dict[str, float]] = {}
        self._ema_alpha: float = 0.05  # smoothing factor (lower = smoother)
        self._tui_initialized: bool = False
        self._tui_last_ids: set[int] = set()

    def reset_state(self) -> None:
        """Reset all tracking state to initial values."""
        self._prev_gray = None
        self._tracks = {}
        self._track_families = {}
        self._tag_models = {}
        self._frame_idx = 0
        self._ema = {}
        self._tui_initialized = False
        self._tui_last_ids = set()

    def _ema_smooth(self, tag_id: int, key: str, value: float) -> float:
        """Apply exponential moving average to a value for a given tag/key."""
        if tag_id not in self._ema:
            self._ema[tag_id] = {}
        state = self._ema[tag_id]
        if key not in state:
            state[key] = value
        else:
            alpha = self._ema_alpha
            state[key] = alpha * value + (1.0 - alpha) * state[key]
        return state[key]

    def _print_tui(self, tag_records: list, has_world: bool) -> None:
        """Print a fixed-position TUI table of tag data with EMA smoothing."""
        import sys

        # Build smoothed rows sorted by tag ID
        rows = []
        for tr in sorted(tag_records, key=lambda t: t.id):
            tag_id = tr.id
            cx = self._ema_smooth(tag_id, "cx", float(tr.center_px[0]))
            cy = self._ema_smooth(tag_id, "cy", float(tr.center_px[1]))
            ori_raw = math.degrees(tr.orientation_yaw)
            ori = self._ema_smooth(tag_id, "ori", ori_raw)
            spd_raw = float(tr.speed_px) if tr.speed_px is not None else 0.0
            spd = self._ema_smooth(tag_id, "spd", spd_raw)
            vx, vy = tr.vel_px if tr.vel_px is not None else (0.0, 0.0)
            vang_raw = math.degrees(math.atan2(vy, vx)) if (vx != 0.0 or vy != 0.0) else 0.0
            vang = self._ema_smooth(tag_id, "vang", vang_raw)

            row = {"id": tag_id, "cx": cx, "cy": cy, "ori": ori, "spd": spd, "vang": vang}

            if has_world:
                H = self.homography
                u, v = float(tr.center_px[0]), float(tr.center_px[1])
                vec = np.array([u, v, 1.0], dtype=float)
                Xw = H @ vec
                if abs(Xw[2]) > 1e-6:
                    wx = self._ema_smooth(tag_id, "wx", Xw[0] / Xw[2])
                    wy = self._ema_smooth(tag_id, "wy", Xw[1] / Xw[2])
                    row["wx"] = wx
                    row["wy"] = wy

            rows.append(row)

        current_ids = {r["id"] for r in rows}

        # Prune EMA state for tags no longer visible
        for old_id in list(self._ema.keys()):
            if old_id not in current_ids:
                del self._ema[old_id]

        # Build output lines
        header = f"{'ID':>4s}  {'CX':>6s}  {'CY':>6s}  {'ORI':>8s}  {'SPEED':>8s}  {'VANG':>8s}"
        if has_world:
            header += f"  {'WX':>8s}  {'WY':>8s}"

        sep = "-" * len(header)

        lines = [sep, header, sep]
        for r in rows:
            line = f"{r['id']:4d}  {r['cx']:6.1f}  {r['cy']:6.1f}  {r['ori']:+7.1f}°  {r['spd']:7.1f}  {r['vang']:+7.1f}°"
            if has_world and "wx" in r:
                line += f"  {r['wx']:7.1f}cm  {r['wy']:7.1f}cm"
            lines.append(line)
        lines.append(sep)

        # Calculate how many lines to clear
        total_lines = len(lines)
        if self._tui_initialized:
            # Move cursor up to overwrite previous output
            # Need to account for previous frame's line count
            prev_count = 4 + len(self._tui_last_ids)  # header(3) + data rows + footer(1)
            sys.stdout.write(f"\033[{prev_count}A")

        # Write new content, clearing each line
        for line in lines:
            sys.stdout.write(f"\033[2K{line}\n")

        sys.stdout.flush()
        self._tui_initialized = True
        self._tui_last_ids = current_ids

    @staticmethod
    def _get_dict_by_family(name: str):
        """Map family string to OpenCV ArUco predefined dictionary."""
        m = {
            "16h5": cv.aruco.DICT_APRILTAG_16h5,
            "25h9": cv.aruco.DICT_APRILTAG_25h9,
            "36h10": cv.aruco.DICT_APRILTAG_36h10,
            "36h11": cv.aruco.DICT_APRILTAG_36h11,
            "aruco_4x4": cv.aruco.DICT_4X4_50,
        }
        return m.get(name, cv.aruco.DICT_APRILTAG_36h11)

    def _build_detectors(self):
        """Create per-family ArUco detectors configured with AprilTag params."""
        fams = [self.family] if self.family != "all" else ["16h5", "25h9", "36h10", "36h11"]
        if self.detect_aruco_4x4:
            fams.append("aruco_4x4")
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
            detectors.append((d, p, f))
        return detectors

    @staticmethod
    def _maybe_preprocess(
        gray: np.ndarray,
        use_clahe: bool,
        use_sharpen: bool,
        use_highpass: bool = True,
        highpass_ksize: int = 51,
    ) -> np.ndarray:
        """Optionally apply preprocessing to a grayscale image.

        High-pass filtering is on by default — it subtracts a blurred
        version of the image, removing low-frequency glare gradients
        while preserving the high-frequency edges that define tags.
        """
        out = gray
        if use_highpass:
            k = highpass_ksize
            if k % 2 == 0:
                k += 1
            blurred = cv.GaussianBlur(out, (k, k), 0)
            hp = cv.subtract(out, blurred)
            out = cv.normalize(hp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        if use_clahe:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = clahe.apply(out)
        if use_sharpen:
            k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            out = cv.filter2D(out, -1, k)
        return out

    def detect_apriltags(self, frame_bgr: np.ndarray, scale: float = 1.0, gray: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray, int, str]]:
        """Detect AprilTags in a BGR frame.

        Args:
            frame_bgr: Input color frame in BGR order.
            scale: Optional downscale factor for speed (<1 downscales).
            gray: Optional pre-computed grayscale image. If None, computed
                from *frame_bgr* internally.

        Returns:
            A list of (pts[4x2], raw_pts[4x2], id, family) for each detected tag.
        """
        h, w = frame_bgr.shape[:2]
        if gray is None:
            gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            gray = cv.resize(gray, (new_w, new_h), interpolation=cv.INTER_AREA)
        gray = self._maybe_preprocess(
            gray, self.use_clahe, self.use_sharpen,
            self.use_highpass, self.highpass_ksize,
        )

        detections: List[Tuple[np.ndarray, np.ndarray, int, str]] = []
        for d, p, fam in self.detectors:
            detector = cv.aruco.ArucoDetector(d, p)
            corners, ids, _rej = detector.detectMarkers(gray)
            if ids is None:
                continue
            for c, idv in zip(corners, ids.flatten().tolist()):
                pts = c.reshape(-1, 2).astype(np.float32)
                if scale < 1.0:
                    pts = pts / float(scale)
                detections.append((pts, pts.copy(), int(idv), fam))
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
            diag = diagnose_camera_failure(int(self.index))
            if not diag.get("exists", True):
                raise CameraNotFoundError(
                    f"Camera at index {self.index} does not exist."
                )
            blocking = diag.get("blocking_processes", [])
            if blocking:
                proc = blocking[0]
                raise CameraInUseError(
                    f"Camera {self.index} is in use by process "
                    f"'{proc['name']}' (PID {proc['pid']}). "
                    f"Kill it with: kill {proc['pid']}",
                    pid=proc["pid"],
                    process_name=proc["name"],
                )
            raise CameraError(f"Failed to open camera {self.index}")
        if self.cap_width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, int(self.cap_width))
        if self.cap_height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(self.cap_height))
        return self.cap

    def _update_playfield(self, frame: np.ndarray, gray: Optional[np.ndarray] = None) -> None:
        """Update cached playfield polygon via Playfield.

        Deskew is handled by PlayfieldDisplay; this only updates geometry.
        """
        try:
            self.playfield.update(frame, gray=gray)
            poly = self.playfield.get_polygon()
            if poly is not None:
                self.play_poly = poly.astype(np.float32)
        except Exception:
            pass



    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> List[TagRecord]:
        """Process a single BGR frame: detect/track tags and return TagRecords.

        This is the stateful detection/tracking core extracted from ``run()``.
        It updates ``self._prev_gray``, ``self._tracks``, ``self._tag_models``,
        and ``self._frame_idx``.

        The method does **not** open windows, call ``waitKey``, print, or read
        from a camera.

        Args:
            frame_bgr: A BGR image (numpy array) to process.
            timestamp: Monotonic timestamp for this frame.

        Returns:
            A list of :class:`TagRecord` for every tag detected/tracked in
            this frame.
        """
        # 2) Convert to gray and perform detection or faster LK tracking
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        detections: List[Tuple[np.ndarray, np.ndarray, int, str]] = []
        if (self.detect_interval <= 1
                or self._frame_idx % max(1, self.detect_interval) == 0
                or self._prev_gray is None
                or not self._tracks):
            w = frame_bgr.shape[1]
            scale = (min(1.0, float(self.proc_width) / float(w))
                     if (self.proc_width and self.proc_width > 0 and w > 0)
                     else 1.0)
            detections = self.detect_apriltags(frame_bgr, scale=scale, gray=gray)
            self._tracks = {tid: pts for (pts, _raw, tid, _fam) in detections}
            self._track_families = {tid: fam for (_pts, _raw, tid, fam) in detections}
        else:
            # Track existing tag corners forward with LK; fall back to detection on loss
            new_tracks: dict[int, np.ndarray] = {}
            for tid, pts in self._tracks.items():
                new_pts = AprilCam.lk_track(self._prev_gray, gray, pts)
                if new_pts is not None:
                    new_tracks[tid] = new_pts
                    fam = self._track_families.get(tid, "36h11")
                    detections.append((new_pts, new_pts, tid, fam))
            self._tracks = new_tracks
            if len(detections) == 0:
                w = frame_bgr.shape[1]
                scale = (min(1.0, float(self.proc_width) / float(w))
                         if (self.proc_width and self.proc_width > 0 and w > 0)
                         else 1.0)
                detections = self.detect_apriltags(frame_bgr, scale=scale, gray=gray)
                self._tracks = {tid: pts for (pts, _raw, tid, _fam) in detections}
                self._track_families = {tid: fam for (_pts, _raw, tid, fam) in detections}

        # 3) Update Playfield cache (polygon) for cropping/deskew
        self._update_playfield(frame_bgr, gray=gray)

        # 4) Keep only detections inside the current playfield polygon
        if detections:
            in_dets: List[Tuple[np.ndarray, np.ndarray, int, str]] = []
            for pts, raw, tid, fam in detections:
                if self.playfield.isIn(pts):
                    in_dets.append((pts, raw, tid, fam))
            detections = in_dets
            self._tracks = {tid: pts for (pts, _raw, tid, _fam) in detections}

        # 5) Update/maintain tag models and playfield flows
        for pts, _raw, tid, fam in detections:
            if tid in self._tag_models:
                self._tag_models[tid].update(pts, timestamp=timestamp, homography=self.homography)
            else:
                self._tag_models[tid] = AprilTagModel.from_corners(
                    tid, pts, homography=self.homography,
                    timestamp=timestamp, frame=self._frame_idx,
                    family=fam,
                )
            self._tag_models[tid].frame = self._frame_idx
            self.playfield.add_tag(self._tag_models[tid])

        # Prune models not seen recently (>1.5s)
        seen_ids = {tid for _pts, _r, tid, _fam in detections}
        for tid in list(self._tag_models.keys()):
            if (tid not in seen_ids
                    and self._tag_models[tid].last_ts is not None
                    and (timestamp - float(self._tag_models[tid].last_ts)) > 1.5):
                del self._tag_models[tid]

        # Build TagRecord objects — velocity is now computed by Playfield.add_tag()
        tag_records: List[TagRecord] = []
        flows = self.playfield.get_flows()
        for pts, _raw, tid, _fam in detections:
            model = self._tag_models.get(tid)
            if model is None:
                continue

            flow = flows.get(tid)
            vel_px_val: Optional[Tuple[float, float]] = flow.vel_px if flow else None
            speed_px_val: Optional[float] = flow.speed_px if flow else None

            tr = TagRecord.from_apriltag(
                model,
                vel_px=vel_px_val,
                speed_px=speed_px_val,
                vel_world=None,
                speed_world=None,
                heading_rad=None,
                timestamp=timestamp,
                frame_index=self._frame_idx,
            )
            tag_records.append(tr)

        # Bookkeeping
        self._frame_idx += 1
        self._prev_gray = gray
        return tag_records

    def run(self) -> None:
        """Main capture/detect/track loop with display and overlays."""
        cap = self._init_capture()
        if cap is None:
            return
        # Window is managed by PlayfieldDisplay

        self.reset_state()
        paused = False
        last_display: Optional[np.ndarray] = None

        try:
            while True:
                if not paused:
                    # 1) Read next frame
                    ok, frame = cap.read()
                    if not ok:
                        print("Camera read failed.")
                        break

                    now = time.monotonic()
                    tag_records = self.process_frame(frame, now)

                    # 6) Optional TUI display (fixed-position, EMA-smoothed)
                    if self.print_tags and tag_records:
                        self._print_tui(tag_records, has_world=self.homography is not None)

                    # 8) Prepare display image and draw overlays
                    display = self.display.update(frame)
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
        finally:
            # Cleanup resources
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
    """Return a list of (dictionary, parameters, family_name) configured for the requested family/families.

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
        detectors.append((d, p, f))
    return detectors


def detect_apriltags(
    frame_bgr: np.ndarray,
    detectors,
    scale: float = 1.0,
    clahe: bool = False,
    sharpen: bool = False,
):
    """Detect AprilTags in an image using provided detectors.

    Returns a list of tuples: (pts[4x2], raw_pts[4x2], id, family)
    """
    h, w = frame_bgr.shape[:2]
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        gray = cv.resize(gray, (new_w, new_h), interpolation=cv.INTER_AREA)
    gray = AprilCam._maybe_preprocess(gray, clahe, sharpen)

    detections: List[Tuple[np.ndarray, np.ndarray, int, str]] = []
    for d, p, fam in detectors:
        detector = cv.aruco.ArucoDetector(d, p)
        corners, ids, _rej = detector.detectMarkers(gray)
        if ids is None:
            continue
        for c, idv in zip(corners, ids.flatten().tolist()):
            pts = c.reshape(-1, 2).astype(np.float32)
            if scale < 1.0:
                pts = pts / float(scale)
            detections.append((pts, pts.copy(), int(idv), fam))
    return detections
