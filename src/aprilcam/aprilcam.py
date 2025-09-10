from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple
import json

import cv2 as cv
import numpy as np
from dotenv import dotenv_values

from .config import AppConfig
from .camutil import list_cameras as _list_cameras, select_camera_by_pattern


# Corner labels for 4x4 ArUco markers used as field corners
ARUCO_CORNER_ABBR = {0: "UL", 1: "UR", 2: "LL", 3: "LR"}


def draw_text_with_outline(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.7,
    thickness: int = 1,
):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv.LINE_AA)


def _get_dict_by_family(name: str):
    m = {
        "16h5": cv.aruco.DICT_APRILTAG_16h5,
        "25h9": cv.aruco.DICT_APRILTAG_25h9,
        "36h10": cv.aruco.DICT_APRILTAG_36h10,
        "36h11": cv.aruco.DICT_APRILTAG_36h11,
    }
    return m.get(name, cv.aruco.DICT_APRILTAG_36h11)


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
    fams = [family] if family != "all" else ["16h5", "25h9", "36h10", "36h11"]
    detectors = []
    for f in fams:
        d = cv.aruco.getPredefinedDictionary(_get_dict_by_family(f))
        p = cv.aruco.DetectorParameters()
        p.cornerRefinementMethod = {
            "none": cv.aruco.CORNER_REFINE_NONE,
            "contour": cv.aruco.CORNER_REFINE_CONTOUR,
            "subpix": cv.aruco.CORNER_REFINE_SUBPIX,
        }.get(corner_refine, cv.aruco.CORNER_REFINE_SUBPIX)
        p.aprilTagQuadDecimate = float(max(1.0, quad_decimate))
        p.aprilTagQuadSigma = float(max(0.0, quad_sigma))
        # Loosen white/black difference slightly to better handle rotated, resampled edges
        if hasattr(p, "aprilTagMinWhiteBlackDiff"):
            try:
                p.aprilTagMinWhiteBlackDiff = int(max(0, int(round(april_min_wb_diff))))
            except Exception:
                # Fallback to a safe default if types are unexpected
                p.aprilTagMinWhiteBlackDiff = 3
        # Allow smaller clusters (small/blurred tags)
        if hasattr(p, "aprilTagMinClusterPixels"):
            try:
                p.aprilTagMinClusterPixels = int(max(1, int(april_min_cluster_pixels)))
            except Exception:
                p.aprilTagMinClusterPixels = 5
        # Tolerate line fit error a bit more for rotated/interpolated edges
        if hasattr(p, "aprilTagMaxLineFitMse"):
            try:
                p.aprilTagMaxLineFitMse = float(max(1.0, float(april_max_line_fit_mse)))
            except Exception:
                p.aprilTagMaxLineFitMse = 20.0
        p.detectInvertedMarker = bool(detect_inverted)
        detectors.append((d, p))
    return detectors


def build_aruco4_detector():
    d = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    p = cv.aruco.DetectorParameters()
    return [(d, p)]


def _maybe_preprocess(gray: np.ndarray, use_clahe: bool, use_sharpen: bool) -> np.ndarray:
    out = gray
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
    if use_sharpen:
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv.filter2D(out, -1, k)
    return out


def detect_apriltags(
    frame_bgr: np.ndarray,
    detectors: List[Tuple[cv.aruco.Dictionary, cv.aruco.DetectorParameters]],
    scale: float = 1.0,
    clahe: bool = False,
    sharpen: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    h, w = frame_bgr.shape[:2]
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        gray = cv.resize(gray, (new_w, new_h), interpolation=cv.INTER_AREA)
    gray = _maybe_preprocess(gray, clahe, sharpen)

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


def lk_track(prev_gray: np.ndarray, gray: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    # Track the 4 corners with LK optical flow
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3,
                                          criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None or st is None or int(st.sum()) < 4:
        return None
    return p1.reshape(-1, 2)


def draw_detections(frame: np.ndarray, detections: List[Tuple[np.ndarray, np.ndarray, int]]):
    for pts, _raw, tag_id in detections:
        # Use float for geometry, then draw as ints
        ptsf = pts.astype(np.float32)
        p0, p1, p2, p3 = ptsf[0], ptsf[1], ptsf[2], ptsf[3]

        # Draw non-top edges (red)
        cv.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2, cv.LINE_AA)
        cv.line(frame, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 0, 255), 2, cv.LINE_AA)
        cv.line(frame, (int(p3[0]), int(p3[1])), (int(p0[0]), int(p0[1])), (0, 0, 255), 2, cv.LINE_AA)

        # Compute "roof" peak above the top edge (p0-p1)
        center = ptsf.mean(axis=0)
        top_mid = (p0 + p1) / 2.0
        # Tag height estimate from left/right edge lengths
        h_left = float(np.linalg.norm(p0 - p3))
        h_right = float(np.linalg.norm(p1 - p2))
        tag_h = max(1.0, 0.5 * (h_left + h_right))
        hat_h = 0.30 * tag_h

        n = top_mid - center
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-6:
            # Fallback: use edge normal if degenerate
            e = p1 - p0
            # Perpendicular vector
            n = np.array([-e[1], e[0]], dtype=np.float32)
            n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-6:
            n_unit = n / n_norm
            peak = top_mid + n_unit * hat_h
            # Draw green "roof" as two segments to the peak
            cv.line(frame, (int(p0[0]), int(p0[1])), (int(peak[0]), int(peak[1])), (0, 255, 0), 2, cv.LINE_AA)
            cv.line(frame, (int(peak[0]), int(peak[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2, cv.LINE_AA)
        else:
            # If still degenerate, draw straight top edge in green
            cv.line(frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2, cv.LINE_AA)

        # ID label near top-left
        x, y = int(p0[0]), int(p0[1])
        draw_text_with_outline(frame, f"ID {tag_id}", (x, y - 8), color=(0, 0, 255), font_scale=0.6, thickness=1)


def draw_orientation_velocity_vectors(
    frame: np.ndarray,
    detections: List[Tuple[np.ndarray, np.ndarray, int]],
    speeds: dict[int, float],
    scale: float = 0.5,
    min_len: int = 20,
    max_len: int = 250,
    color: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 3,
) -> None:
    for pts, _raw, tag_id in detections:
        ptsf = pts.astype(np.float32)
        center = ptsf.mean(axis=0)
        top_mid = (ptsf[0] + ptsf[1]) / 2.0
        dir_vec = top_mid - center
        norm = float(np.linalg.norm(dir_vec))
        if norm < 1e-6:
            continue
        dir_unit = dir_vec / norm
        speed = float(speeds.get(tag_id, 0.0))
        length_px = int(max(min_len, min(max_len, speed * scale)))
        start = (int(center[0]), int(center[1]))
        end = (int(center[0] + dir_unit[0] * length_px), int(center[1] + dir_unit[1] * length_px))
        cv.arrowedLine(frame, start, end, color, thickness, tipLength=0.12)
        cv.circle(frame, start, 4, color, -1)
        draw_text_with_outline(frame, f"v={speed:.1f}px/s", (start[0] + 6, start[1] - 6), color=color, font_scale=0.6, thickness=1)


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


def save_last_camera(idx: int):
    try:
        p = Path.home() / ".aprilcam_last_camera"
        p.write_text(str(int(idx)))
    except Exception:
        pass


def load_last_camera() -> Optional[int]:
    try:
        p = Path.home() / ".aprilcam_last_camera"
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
    window = "aprilcam"
    own_cap = False
    if cap is None:
        cap = cv.VideoCapture(int(index), 0 if backend is None else int(backend))
        own_cap = True
    if not cap or not cap.isOpened():
        print("Failed to open camera.")
        return
    if cap_width:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, int(cap_width))
    if cap_height:
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(cap_height))

    detectors = build_detectors(
        family,
        corner_refine,
        quad_decimate,
        quad_sigma,
        detect_inverted,
        april_min_wb_diff,
        april_min_cluster_pixels,
        april_max_line_fit_mse,
    )
    aruco4_detectors = build_aruco4_detector()

    if not headless:
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        # Default size; may be changed below for screen sources
        target_w, target_h = 1280, 720
        # Best-effort: compute an overlay position and size that stays off the captured playfield and inside the monitor
        try:
            if hasattr(cap, "get_bbox") and hasattr(cap, "get_display_rect"):
                cap_l, cap_t, cap_w, cap_h = cap.get_bbox()  # type: ignore[attr-defined]
                mon_l, mon_t, mon_w, mon_h = cap.get_display_rect()  # type: ignore[attr-defined]
                mon_r, mon_b = mon_l + mon_w, mon_t + mon_h
                cap_r, cap_b = cap_l + cap_w, cap_t + cap_h

                # Preferred region: to the right; fallback: left; then below; then above
                pad = 12
                candidates = []
                # Right side
                if cap_r + pad < mon_r:
                    candidates.append((cap_r + pad, cap_t, mon_r - (cap_r + pad), min(cap_h, mon_b - cap_t)))
                # Left side
                if mon_l < cap_l - pad:
                    candidates.append((mon_l, cap_t, (cap_l - pad) - mon_l, min(cap_h, mon_b - cap_t)))
                # Below
                if cap_b + pad < mon_b:
                    candidates.append((cap_l, cap_b + pad, min(cap_w, mon_r - cap_l), mon_b - (cap_b + pad)))
                # Above
                if mon_t < cap_t - pad:
                    candidates.append((cap_l, mon_t, min(cap_w, mon_r - cap_l), (cap_t - pad) - mon_t))

                # Choose the candidate with the largest area
                if candidates:
                    bx, by, bw, bh = max(candidates, key=lambda r: r[2] * r[3])
                    # Target overlay size fits within candidate box but keeps aspect ~16:9 as a default
                    target_w = max(320, min(1280, bw))
                    target_h = max(180, min(720, bh))
                    cv.resizeWindow(window, int(target_w), int(target_h))
                    cv.moveWindow(window, int(bx), int(by))
                else:
                    cv.resizeWindow(window, target_w, target_h)
            else:
                cv.resizeWindow(window, target_w, target_h)
        except Exception:
            cv.resizeWindow(window, target_w, target_h)

    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0
    tracks: dict[int, np.ndarray] = {}
    vel_ema: dict[int, float] = {}
    last_seen: dict[int, Tuple[float, float, float]] = {}
    paused = False
    last_display: Optional[np.ndarray] = None
    play_poly: Optional[np.ndarray] = None
    if playfield_poly_init is not None and isinstance(playfield_poly_init, np.ndarray) and playfield_poly_init.shape == (4, 2):
        play_poly = playfield_poly_init.astype(np.float32)

    # Arrow styling (velocity vectors). Halved lengths vs prior settings.
    arrow_scale, arrow_min, arrow_max = 0.25, 10, 125
    arrow_color, arrow_width = (255, 0, 255), 2

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.")
                    break

                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                detections: List[Tuple[np.ndarray, np.ndarray, int]] = []
                if detect_interval <= 1 or frame_idx % max(1, detect_interval) == 0 or prev_gray is None or not tracks:
                    w = frame.shape[1]
                    scale = min(1.0, float(proc_width) / float(w)) if (proc_width and proc_width > 0 and w > 0) else 1.0
                    detections = detect_apriltags(frame, detectors, scale=scale, clahe=use_clahe, sharpen=use_sharpen)
                    tracks = {tid: pts for (pts, _raw, tid) in detections}
                else:
                    new_tracks: dict[int, np.ndarray] = {}
                    for tid, pts in tracks.items():
                        new_pts = lk_track(prev_gray, gray, pts)
                        if new_pts is not None:
                            new_tracks[tid] = new_pts
                            detections.append((new_pts, new_pts, tid))
                    tracks = new_tracks
                    if len(detections) == 0:
                        w = frame.shape[1]
                        scale = min(1.0, float(proc_width) / float(w)) if (proc_width and proc_width > 0 and w > 0) else 1.0
                        detections = detect_apriltags(frame, detectors, scale=scale, clahe=use_clahe, sharpen=use_sharpen)
                        tracks = {tid: pts for (pts, _raw, tid) in detections}

                # Detect ArUco 4x4 corner markers and build playfield polygon
                try:
                    w = frame.shape[1]
                    scale = min(1.0, float(proc_width) / float(w)) if (proc_width and proc_width > 0 and w > 0) else 1.0
                    aruco4 = detect_apriltags(frame, aruco4_detectors, scale=scale, clahe=False, sharpen=False)
                    corners_map: dict[int, Tuple[float, float]] = {}
                    for pts, _raw, tid in aruco4:
                        c = pts.astype(np.float32).mean(axis=0)
                        corners_map[tid] = (float(c[0]), float(c[1]))
                        u, v = int(c[0]), int(c[1])
                        cv.circle(frame, (u, v), 4, (255, 255, 0), -1)
                        if tid in ARUCO_CORNER_ABBR:
                            draw_text_with_outline(frame, ARUCO_CORNER_ABBR[tid], (u + 6, v - 6), color=(255, 255, 0), font_scale=0.7, thickness=1)
                    if all(k in corners_map for k in (0, 1, 2, 3)):
                        play_poly = np.array([
                            corners_map[0],
                            corners_map[1],
                            corners_map[3],
                            corners_map[2],
                        ], dtype=np.float32)
                except Exception:
                    pass

                # Filter detections to inside the playfield if defined
                if play_poly is not None and play_poly.shape == (4, 2):
                    cv.polylines(frame, [play_poly.astype(int)], True, (255, 255, 255), 2, cv.LINE_AA)
                    in_dets: List[Tuple[np.ndarray, np.ndarray, int]] = []
                    for pts, raw, tid in detections:
                        c = pts.astype(np.float32).mean(axis=0)
                        inside = cv.pointPolygonTest(play_poly.astype(np.float32), (float(c[0]), float(c[1])), False)
                        if inside >= 0:
                            in_dets.append((pts, raw, tid))
                    detections = in_dets
                    tracks = {tid: pts for (pts, _raw, tid) in detections}

                draw_detections(frame, detections)

                # World coords via homography (cm)
                if homography is not None and len(detections) > 0:
                    for pts, _raw, _tid in detections:
                        c = pts.astype(np.float32).mean(axis=0)
                        u, v = float(c[0]), float(c[1])
                        vec = np.array([u, v, 1.0], dtype=float)
                        Xw = homography @ vec
                        if abs(Xw[2]) > 1e-6:
                            Xcm = Xw[0] / Xw[2]
                            Ycm = Xw[1] / Xw[2]
                            draw_text_with_outline(frame, f"{Xcm:.1f}cm,{Ycm:.1f}cm", (int(u) + 8, int(v) + 16), color=(0, 255, 0), font_scale=0.55, thickness=1)

                # Velocities
                now = time.monotonic()
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
                        ema = (speed_alpha * inst + (1 - speed_alpha) * prev) if prev is not None else inst
                        vel_ema[tag_id] = ema
                        speeds[tag_id] = ema
                    last_seen[tag_id] = (cx, cy, now)

                # Draw only motion-direction vectors; omit the top-of-tag orientation arrow (replaced by green "roof")
                draw_velocity_vectors(frame, detections, vel_dirs, speeds, scale=arrow_scale, min_len=arrow_min, max_len=arrow_max, color=(0, 255, 255), thickness=arrow_width)

                if print_tags and detections:
                    lines = []
                    H = homography
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

                hud = f"tags: {len(detections)}  alpha:{speed_alpha:.2f}"
                cv.putText(frame, hud, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
                cv.putText(frame, hud, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv.LINE_AA)

                # Prepare overlay view: crop to playfield or deskew if requested
                display = frame
                if play_poly is not None and play_poly.shape == (4, 2):
                    try:
                        if deskew_overlay:
                            # Order: UL(0), UR(1), LR(2), LL(3) as constructed earlier
                            UL, UR, LR, LL = play_poly.astype(np.float32)
                            w_top = float(np.linalg.norm(UR - UL))
                            w_bottom = float(np.linalg.norm(LR - LL))
                            h_left = float(np.linalg.norm(LL - UL))
                            h_right = float(np.linalg.norm(LR - UR))
                            out_w = max(10, int(round(max(w_top, w_bottom))))
                            out_h = max(10, int(round(max(h_left, h_right))))
                            src = np.array([UL, UR, LR, LL], dtype=np.float32)
                            dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
                            M = cv.getPerspectiveTransform(src, dst)
                            display = cv.warpPerspective(frame, M, (out_w, out_h))
                        else:
                            # Crop to bounding box with padding
                            PAD = 24
                            x_coords = play_poly[:, 0]
                            y_coords = play_poly[:, 1]
                            xmin = max(0, int(math.floor(float(x_coords.min()) - PAD)))
                            ymin = max(0, int(math.floor(float(y_coords.min()) - PAD)))
                            xmax = min(frame.shape[1], int(math.ceil(float(x_coords.max()) + PAD)))
                            ymax = min(frame.shape[0], int(math.ceil(float(y_coords.max()) + PAD)))
                            if xmax > xmin and ymax > ymin:
                                display = frame[ymin:ymax, xmin:xmax]
                    except Exception:
                        display = frame
                last_display = display.copy()
            else:
                if last_display is None:
                    ok, frame = cap.read()
                    if not ok:
                        print("Camera read failed.")
                        break
                    last_display = frame.copy()
                display = last_display.copy()
                if not headless:
                    draw_text_with_outline(display, " Paused: Press Space to Run", (10, 30), color=(0, 255, 255), font_scale=0.9, thickness=2)
            if not headless:
                cv.imshow(window, display)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == ord(' '):
                    paused = not paused
                    continue
            else:
                # In headless mode, yield CPU briefly
                time.sleep(0.001)
            if not paused:
                prev_gray = gray
                frame_idx += 1
    finally:
        if own_cap and cap is not None:
            cap.release()
        try:
            if not headless:
                cv.destroyAllWindows()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="aprilcam", description="Detect AprilTags using input specified by homography JSON.")
    parser.add_argument("--backend", type=str, choices=["auto", "avfoundation", "v4l2", "msmf", "dshow"], default="auto",
                        help="Capture backend to prefer for camera sources (homography may include backend)")
    parser.add_argument("--speed-alpha", type=float, default=0.3,
                        help="EMA smoothing factor for speed in [0,1]; higher is more responsive, lower is smoother")
    parser.add_argument("--quiet", action="store_true", help="Reduce OpenCV logging to suppress probe warnings")
    parser.add_argument("--family", type=str, choices=["16h5", "25h9", "36h10", "36h11", "all"], default="36h11",
                        help="AprilTag family to detect (fewer families = faster)")
    parser.add_argument("--proc-width", type=int, default=0,
                        help="Resize width for detection processing (0 = no downscale; downscales for speed, corners scaled back)")
    parser.add_argument("--quad-decimate", type=float, default=1.0, help="AprilTag quad decimate (>=1, larger is faster but less accurate)")
    parser.add_argument("--quad-sigma", type=float, default=0.0, help="AprilTag quad sigma (Gaussian blur) in pixels")
    parser.add_argument("--corner-refine", type=str, choices=["none", "contour", "subpix"], default="subpix",
                        help="Corner refinement method (subpix is slower but more accurate)")
    parser.add_argument("--use-aruco3", action="store_true", help="Use ArUco3 detection (if available)")
    parser.add_argument("--april-min-wb-diff", type=float, default=3.0,
                        help="Minimum white-black intensity difference (lower tolerates blur, may increase false positives)")
    parser.add_argument("--april-min-cluster-pixels", type=int, default=5,
                        help="Minimum connected-pixel cluster size (lower helps small/blurred tags)")
    parser.add_argument("--april-max-line-fit-mse", type=float, default=20.0,
                        help="Max line fit MSE (higher tolerates rotated/interpolated edges)")
    parser.add_argument("--detect-interval", type=int, default=1,
                        help="Run full detection every N frames; others use fast optical flow tracking (1 = detect every frame)")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast enhancement before detection")
    parser.add_argument("--sharpen", action="store_true", help="Apply unsharp mask before detection")
    parser.add_argument("--print-tags", action="store_true", help="Print per-tag ID, center, orientation, and velocity (fixed-width)")
    # Homography file (source configuration)
    parser.add_argument("--homography", type=str, default="homography.json", help="Homography JSON filename (in data dir unless absolute)")
    # Overlay options
    parser.add_argument("--deskew-overlay", action="store_true", help="Warp the playfield quadrilateral to a rectangle in the overlay window")
    args = parser.parse_args(argv)

    # Reduce OpenCV logging if requested
    if getattr(args, "quiet", False):
        try:
            if hasattr(cv, "utils") and hasattr(cv.utils, "logging"):
                cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

    # Backend mapping
    be_map = {
        "auto": None,
        "avfoundation": getattr(cv, "CAP_AVFOUNDATION", 1200),
        "v4l2": getattr(cv, "CAP_V4L2", 200),
        "msmf": getattr(cv, "CAP_MSMF", 1400),
        "dshow": getattr(cv, "CAP_DSHOW", 700),
    }
    be_value = be_map.get(args.backend)

    # Load config and homography JSON (also holds input source info now)
    cfg = None
    try:
        cfg = AppConfig.load()
    except Exception:
        cfg = None

    # No camera probing args in this mode; homography provides the source.

    # Open source based on homography JSON metadata
    # Resolve path
    H_path = Path(args.homography)
    if cfg and not H_path.is_absolute():
        H_path = cfg.data_dir / H_path
    H_meta = None
    H_pix = None
    H = None
    cap = None
    try:
        if H_path.exists():
            data = json.loads(H_path.read_text())
            H = np.array(data.get("homography", []), dtype=float) if data.get("homography") is not None else None
            H_meta = data.get("source")
            H_pix = data.get("pixel_points")
    except Exception:
        H_meta = None
        H = None

    if not H_meta:
        print("Homography file missing or missing 'source' config. Run homocal to generate it.")
        return 2

    src_type = str(H_meta.get("type", "camera"))
    if src_type == "screen":
        region = H_meta.get("region")
        try:
            from .screencap import ScreenCaptureMSS
            cap = ScreenCaptureMSS(
                monitor=int(H_meta.get("monitor", 1)),
                fps=float(H_meta.get("fps", 30.0)),
                region=tuple(region) if region else None,
            )
        except Exception as e:
            print(f"Failed to initialize screen capture: {e}")
            return 2
    else:
        # Camera path using AppConfig
        cam_idx = H_meta.get("index")
        cap = cfg.get_camera(arg=int(cam_idx) if cam_idx is not None else None, backend=args.backend, quiet=bool(args.quiet)) if cfg else None
        if cap and cap.isOpened():
            cw = H_meta.get("cap_width")
            ch = H_meta.get("cap_height")
            if cw:
                cap.set(cv.CAP_PROP_FRAME_WIDTH, int(cw))
            if ch:
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(ch))

    if cap is None or not cap.isOpened():
        print("No camera available.")
        return 1

    # No last-camera persistence; source comes from homography JSON.

    # H already parsed from JSON above if present.

    run_video(
        index=0,
        backend=be_value,
        speed_alpha=max(0.0, min(1.0, float(args.speed_alpha))),
        family=args.family,
        proc_width=int(args.proc_width),
        cap_width=None,
        cap_height=None,
        quad_decimate=float(args.quad_decimate),
        quad_sigma=float(args.quad_sigma),
        corner_refine=str(args.corner_refine),
        detect_inverted=bool((H_meta or {}).get("detect_inverted", True)),
        use_aruco3=bool(args.use_aruco3),
        detect_interval=max(1, int(args.detect_interval)),
        use_clahe=bool(args.clahe),
        use_sharpen=bool(args.sharpen),
        print_tags=bool(args.print_tags),
        cap=cap,
        homography=H,
        headless=False,
        deskew_overlay=bool(args.deskew_overlay),
    april_min_wb_diff=float(args.april_min_wb_diff),
    april_min_cluster_pixels=int(args.april_min_cluster_pixels),
    april_max_line_fit_mse=float(args.april_max_line_fit_mse),
        playfield_poly_init=(
            np.array([H_pix[i] for i in (0, 1, 3, 2)], dtype=np.float32) if isinstance(H_pix, list) and len(H_pix) == 4 else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
