from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple, List

import cv2 as cv
import numpy as np

from .playfield import Playfield
from .models import AprilTag


class PlayfieldDisplay:
    """Display and overlay manager for the playfield.

    - Can deskew the frame using the Playfield polygon once available.
    - Draws detections, world coords, and velocity vectors.
    """

    def __init__(
        self,
        playfield: Playfield,
        window_name: str = "aprilcam",
        headless: bool = False,
        deskew_overlay: bool = False,
    ) -> None:
        # references and flags
        self.playfield = playfield
        self.window = window_name
        self.headless = bool(headless)
        self.deskew_overlay = bool(deskew_overlay)

        # perspective (deskew) cache
        self.M_deskew = None
        self.deskew_size = None

        # display mode bookkeeping so overlays map correctly
        self._mode = "full"  # one of: 'full', 'crop', 'deskew'
        self._crop_xy = (0, 0)  # (xmin, ymin)
        self._crop_wh = (0, 0)  # (w, h)

        # window bookkeeping
        self._win_created = False
        self._last_size = (0, 0)  # (w, h)

    def _ensure_window(self) -> None:
        if self.headless:
            return
        if not self._win_created:
            try:
                cv.namedWindow(self.window, cv.WINDOW_NORMAL)
                self._win_created = True
            except Exception:
                pass

    def _update_deskew(self, frame: np.ndarray) -> None:
        poly = self.playfield.get_polygon()
        if not self.deskew_overlay or poly is None or self.M_deskew is not None:
            return
        UL, UR, LR, LL = poly.astype(np.float32)
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

    def prepare_display(self, frame: np.ndarray) -> np.ndarray:
        # Reset mode by default
        self._mode = "full"
        self._crop_xy = (0, 0)
        self._crop_wh = (frame.shape[1], frame.shape[0])
        poly = self.playfield.get_polygon()
        if poly is None:
            return frame
        try:
            # Deskewed view
            if self.deskew_overlay and self.M_deskew is not None and self.deskew_size is not None:
                w, h = self.deskew_size
                self._mode = "deskew"
                self._crop_xy = (0, 0)
                self._crop_wh = (w, h)
                return cv.warpPerspective(frame, self.M_deskew, (w, h))
            # Cropped view
            PAD = 8
            x_coords = poly[:, 0]
            y_coords = poly[:, 1]
            xmin = max(0, int(math.floor(float(x_coords.min()) - PAD)))
            ymin = max(0, int(math.floor(float(y_coords.min()) - PAD)))
            xmax = min(frame.shape[1], int(math.ceil(float(x_coords.max()) + PAD)))
            ymax = min(frame.shape[0], int(math.ceil(float(y_coords.max()) + PAD)))
            if xmax > xmin and ymax > ymin:
                self._mode = "crop"
                self._crop_xy = (xmin, ymin)
                self._crop_wh = (xmax - xmin, ymax - ymin)
                return frame[ymin:ymax, xmin:xmax]
        except Exception:
            pass
        return frame

    def _map_points_to_display(self, pts: np.ndarray) -> np.ndarray:
        """Transform points from source-frame coords into display-image coords.

        Accepts an array of shape (N, 2) float32/float64 and returns float32.
        """
        if pts is None or len(pts) == 0:
            return pts
        P = pts.astype(np.float32)
        if self._mode == "deskew" and self.M_deskew is not None:
            # perspectiveTransform expects shape (N,1,2)
            P3 = P.reshape(-1, 1, 2)
            Q = cv.perspectiveTransform(P3, self.M_deskew).reshape(-1, 2)
            return Q
        if self._mode == "crop":
            ox, oy = self._crop_xy
            Q = P.copy()
            Q[:, 0] -= float(ox)
            Q[:, 1] -= float(oy)
            return Q
        return P

    @staticmethod
    def _draw_text_with_outline(img: np.ndarray, text: str, org: Tuple[int, int], color=(255, 255, 255), font_scale=0.7, thickness=1):
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv.LINE_AA)

    def draw_overlays(self, frame: np.ndarray, tags: Iterable[AprilTag], homography: Optional[np.ndarray] = None) -> None:
        # playfield outline (transform into display coords)
        poly = self.playfield.get_polygon()
        if poly is not None:
            try:
                poly_disp = self._map_points_to_display(poly.astype(np.float32))
                cv.polylines(frame, [poly_disp.astype(int)], True, (255, 255, 255), 2, cv.LINE_AA)
            except Exception:
                pass
        # tag boxes, ids, velocity, world coords text
        for tag in tags:
            # map corners and center into display coords
            pts_src = tag.corners_px.astype(np.float32)
            ptsf = self._map_points_to_display(pts_src)
            pts = ptsf.astype(np.int32)
            p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
            # draw a peaked "roof" indicating the outward top direction
            try:
                # Compute apex in source coords using the tag's top direction
                pts_src4 = tag.corners_px.astype(np.float32)
                top_mid_src = (pts_src4[0] + pts_src4[1]) * 0.5
                nux, nuy = getattr(tag, "top_dir_px", (1.0, 0.0))
                top_len = float(np.linalg.norm(pts_src4[1] - pts_src4[0]))
                roof_len = max(6.0, min(80.0, 0.35 * top_len))
                apex_src = np.array([[top_mid_src[0] + nux * roof_len, top_mid_src[1] + nuy * roof_len]], dtype=np.float32)
                apex = self._map_points_to_display(apex_src).reshape(2).astype(int)
                cv.line(frame, tuple(p0), tuple(apex), (0, 255, 0), 2, cv.LINE_AA)
                cv.line(frame, tuple(p1), tuple(apex), (0, 255, 0), 2, cv.LINE_AA)
            except Exception:
                # fallback: flat green top edge
                cv.line(frame, tuple(p0), tuple(p1), (0, 255, 0), 2, cv.LINE_AA)
            cv.line(frame, tuple(p1), tuple(p2), (0, 0, 255), 2, cv.LINE_AA)
            cv.line(frame, tuple(p2), tuple(p3), (0, 0, 255), 2, cv.LINE_AA)
            cv.line(frame, tuple(p3), tuple(p0), (0, 0, 255), 2, cv.LINE_AA)
            c_src = np.array([tag.center_px], dtype=np.float32)
            c_map = self._map_points_to_display(c_src).reshape(2)
            cx, cy = int(c_map[0]), int(c_map[1])
            # velocity arrow (supports AprilTagFlow with vel_px property)
            vx, vy = (0.0, 0.0)
            try:
                vx, vy = getattr(tag, "vel_px", (0.0, 0.0))
            except Exception:
                vx, vy = (0.0, 0.0)
            norm = math.hypot(vx, vy)
            if norm > 1e-6:
                length_px = int(max(12, min(250, norm * 0.5)))
                ux, uy = (vx / norm, vy / norm)
                # build arrow in source coords, then map both points
                start_src = np.array([[tag.center_px[0], tag.center_px[1]]], dtype=np.float32)
                end_src = np.array([[tag.center_px[0] + ux * length_px, tag.center_px[1] + uy * length_px]], dtype=np.float32)
                start_map = self._map_points_to_display(start_src).reshape(2)
                end_map = self._map_points_to_display(end_src).reshape(2)
                end = (int(end_map[0]), int(end_map[1]))
                cx, cy = int(start_map[0]), int(start_map[1])
                cv.arrowedLine(frame, (cx, cy), end, (0, 255, 255), 2, tipLength=0.12)
                cv.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            # ID label (centered on the tag center)
            id_text = f"{tag.id}"
            (tw, th), base = cv.getTextSize(id_text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = int(cx - tw * 0.5)
            ty = int(cy + th * 0.5)
            self._draw_text_with_outline(frame, id_text, (tx, ty), color=(0, 0, 255), font_scale=0.8, thickness=2)
            # world coords small label outside bbox
            if tag.world_xy is not None:
                Xw, Yw = tag.world_xy
                text = f"{Xw:.1f},{Yw:.1f}"
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                xmin = int(x_coords.min())
                xmax = int(x_coords.max())
                ymin = int(y_coords.min())
                ymax = int(y_coords.max())
                (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                pad = 8
                fw, fh = frame.shape[1], frame.shape[0]
                placed = False
                tx = int((xmin + xmax) * 0.5 - tw * 0.5)
                ty = int(ymax + pad + th)
                if ty + base <= fh and 0 <= tx <= fw - tw:
                    self._draw_text_with_outline(frame, text, (tx, ty), color=(0, 255, 0), font_scale=0.5, thickness=1)
                    placed = True
                if not placed:
                    tx = int((xmin + xmax) * 0.5 - tw * 0.5)
                    ty = int(ymin - pad)
                    if ty - th >= 0 and 0 <= tx <= fw - tw:
                        self._draw_text_with_outline(frame, text, (tx, ty), color=(0, 255, 0), font_scale=0.5, thickness=1)
                        placed = True
                if not placed:
                    tx = int(xmax + pad)
                    ty = int((ymin + ymax) * 0.5 + th * 0.5)
                    if tx + tw <= fw and 0 <= ty - th and ty + base <= fh:
                        self._draw_text_with_outline(frame, text, (tx, ty), color=(0, 255, 0), font_scale=0.5, thickness=1)
                        placed = True
                if not placed:
                    tx = int(xmin - pad - tw)
                    ty = int((ymin + ymax) * 0.5 + th * 0.5)
                    if tx >= 0 and 0 <= ty - th and ty + base <= fh:
                        self._draw_text_with_outline(frame, text, (tx, ty), color=(0, 255, 0), font_scale=0.5, thickness=1)
                        placed = True
                if not placed:
                    self._draw_text_with_outline(frame, text, (cx + 8, cy + 14), color=(0, 255, 0), font_scale=0.5, thickness=1)

    def update(self, frame: np.ndarray) -> np.ndarray:
        # ensure playfield cache and deskew once
        self.playfield.update(frame)
        self._update_deskew(frame)
        self._ensure_window()
        return self.prepare_display(frame)

    def show(self, display: np.ndarray) -> None:
        if self.headless:
            return
        # Resize window to match the current display image to avoid whitespace
        try:
            h, w = display.shape[:2]
            if (w, h) != self._last_size and w > 0 and h > 0:
                cv.resizeWindow(self.window, int(w), int(h))
                self._last_size = (int(w), int(h))
        except Exception:
            pass
        cv.imshow(self.window, display)

    def pause(self, frame: np.ndarray, text: str = " Paused: Press Space to Run") -> None:
        """Overlay a paused message onto the given frame."""
        if frame is None:
            return
        try:
            self._draw_text_with_outline(frame, text, (10, 30), color=(0, 255, 255), font_scale=0.9, thickness=2)
        except Exception:
            pass
