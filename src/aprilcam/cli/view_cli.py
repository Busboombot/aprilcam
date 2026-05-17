"""CLI subcommand: aprilcam view — Live view via the AprilCam daemon."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Optional


def _tag_record_to_dict(tag_record) -> dict:
    """Convert a TagRecord Pydantic model to a legacy-format dict for display helpers.

    The panel formatters and _tag_dict_to_aprilcam() were written against the
    old FrameMessage.tags dict schema.  This shim maps TagRecord fields to
    the expected keys so the display code needs no further changes.
    """
    vel_px_raw = tag_record.vel_px
    vel_world_raw = tag_record.vel_world
    world_xy = tag_record.world_xy

    # corners_px in TagRecord is list[tuple[float, float]]; flatten to list[list]
    corners = [[c[0], c[1]] for c in tag_record.corners_px]

    return {
        "id": tag_record.id,
        "center_px": list(tag_record.center_px),
        "corners_px": corners,
        "orientation_yaw": tag_record.yaw,
        "world_xy": list(world_xy) if world_xy is not None else None,
        "in_playfield": tag_record.in_playfield,
        "vel_px": list(vel_px_raw) if vel_px_raw is not None else [0.0, 0.0],
        "vel_world": list(vel_world_raw) if vel_world_raw is not None else None,
    }


def _tag_dict_to_aprilcam(tag_dict: dict):
    """Convert a TagRecord dict into an AprilTag object."""
    import numpy as np
    from aprilcam.core.models import AprilTag

    corners_raw = tag_dict.get("corners_px", [[0, 0]] * 4)
    corners_px = np.array(corners_raw, dtype=np.float32)

    center_raw = tag_dict.get("center_px", [0.0, 0.0])
    center_px = (float(center_raw[0]), float(center_raw[1]))

    c = corners_px.mean(axis=0)
    p0, p1 = corners_px[0], corners_px[1]
    top_mid = (p0 + p1) / 2.0
    n = top_mid - c
    n_norm = float(np.linalg.norm(n))
    if n_norm > 1e-6:
        top_dir_px = (float(n[0]) / n_norm, float(n[1]) / n_norm)
    else:
        top_dir_px = (1.0, 0.0)

    orientation_yaw = float(tag_dict.get("orientation_yaw", 0.0))

    world_raw = tag_dict.get("world_xy")
    world_xy = (float(world_raw[0]), float(world_raw[1])) if world_raw is not None else None

    tag = AprilTag(
        id=int(tag_dict["id"]),
        family="36h11",
        corners_px=corners_px,
        center_px=center_px,
        top_dir_px=top_dir_px,
        orientation_yaw=orientation_yaw,
        world_xy=world_xy,
        in_playfield=bool(tag_dict.get("in_playfield", False)),
    )

    vel_raw = tag_dict.get("vel_px")
    tag.vel_px = (float(vel_raw[0]), float(vel_raw[1])) if vel_raw is not None else (0.0, 0.0)

    return tag


def _load_paths(paths_file: Path) -> dict:
    try:
        raw = json.loads(paths_file.read_text())
    except Exception:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        result = {}
        for item in raw:
            if isinstance(item, dict):
                pid = item.get("path_id") or item.get("id") or str(len(result))
                result[str(pid)] = item
        return result
    return {}


def _decode_frame(frame_bytes, np, cv):
    if isinstance(frame_bytes, (bytes, bytearray)):
        buf = np.frombuffer(frame_bytes, dtype=np.uint8)
    else:
        buf = np.array(frame_bytes, dtype=np.uint8)
    return cv.imdecode(buf, cv.IMREAD_COLOR)


# ── Tag panel formatting ─────────────────────────────────────────────────────

def _vel_mag(t: dict) -> float:
    vp = t.get("vel_px", [0, 0])
    return math.hypot(float(vp[0]), float(vp[1]))



_MOB_HDR = (
    f"{'ID':>2} {'PxX':>4} {'PxY':>4} {'WldX':>6} {'WldY':>6} {'Ang':>4} {'VelX':>5} {'VelY':>5}\n"
    + "-" * 43 + "\n"
)

_STAT_HDR = (
    f"{'ID':>2} {'PxX':>4} {'PxY':>4} {'WldX':>6} {'WldY':>6} {'Ang':>4}\n"
    + "-" * 31 + "\n"
)


def _fmt_mobile_row(t: dict) -> str:
    tid = int(t.get("id", 0))
    cx, cy = t.get("center_px", [0, 0])
    wxy = t.get("world_xy")
    wx = f"{float(wxy[0]):6.1f}" if wxy else "    --"
    wy = f"{float(wxy[1]):6.1f}" if wxy else "    --"
    ang = math.degrees(float(t.get("orientation_yaw", 0.0)))
    vw = t.get("vel_world")
    vp = t.get("vel_px", [0, 0])
    vx, vy = (float(vw[0]), float(vw[1])) if vw is not None else (float(vp[0]), float(vp[1]))
    return f"{tid:>2} {int(cx):>4} {int(cy):>4} {wx} {wy} {ang:>4.0f} {vx:>5.1f} {vy:>5.1f}\n"


def _fmt_stat_row(t: dict) -> str:
    tid = int(t.get("id", 0))
    cx, cy = t.get("center_px", [0, 0])
    wxy = t.get("world_xy")
    wx = f"{float(wxy[0]):6.1f}" if wxy else "    --"
    wy = f"{float(wxy[1]):6.1f}" if wxy else "    --"
    ang = math.degrees(float(t.get("orientation_yaw", 0.0)))
    return f"{tid:>2} {int(cx):>4} {int(cy):>4} {wx} {wy} {ang:>4.0f}\n"


# ── main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aprilcam view",
        description="Open a live view window fed by the AprilCam daemon",
    )
    parser.add_argument("camera", metavar="CAMERA", help="Camera name or integer index")
    parser.add_argument(
        "--unix-path",
        default=None,
        metavar="PATH",
        help="Unix socket path for the daemon control socket",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=None,
        metavar="N",
        help="TCP port the daemon is listening on",
    )
    args = parser.parse_args(argv)

    import numpy as np
    import cv2 as cv

    from aprilcam.config import Config
    from aprilcam.client.control import DaemonControl
    from aprilcam.core.playfield import PlayfieldBoundary
    from aprilcam.ui.display import PlayfieldDisplay

    config = Config.load()
    dc = DaemonControl.connect_default(
        config, unix_path=args.unix_path, tcp_port=args.tcp_port
    )

    cam_name: Optional[str] = None
    try:
        camera_arg = args.camera
        try:
            cam_index = int(camera_arg)
            cam_name = dc.open_camera(cam_index)
        except ValueError:
            # camera_arg is a name, not an index — verify it is open
            info = dc.get_camera_info(camera_arg)
            cam_name = info.cam_name

        if cam_name is None:
            print(f"Error: could not resolve camera '{camera_arg}'", file=sys.stderr)
            dc.close()
            return 1

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        dc.close()
        return 1

    # Open image and tag streams
    try:
        image_consumer = dc.get_image_stream(cam_name)
        tag_consumer = dc.get_tag_stream(cam_name)
    except Exception as exc:
        print(f"Error: could not open streams for camera '{cam_name}': {exc}", file=sys.stderr)
        dc.close()
        return 1
    finally:
        # DaemonControl is no longer needed after streams are open
        dc.close()

    # Read first image frame
    try:
        first_frame = image_consumer.read()
    except (EOFError, RuntimeError) as exc:
        print(f"Error: could not read first frame: {exc}", file=sys.stderr)
        image_consumer.close()
        tag_consumer.close()
        return 1

    # Read first tag frame (non-blocking with a short timeout via separate read)
    first_tag_frame = None
    try:
        first_tag_frame = tag_consumer.read()
    except (EOFError, RuntimeError):
        pass

    _DISPLAY_W = 1000  # canvas is always this wide; height scales proportionally

    boundary = PlayfieldBoundary()
    display = PlayfieldDisplay(
        playfield=boundary,
        window_name="aprilcam view",
        deskew_overlay=True,
    )

    # State shared between reader threads and main (Tk) thread
    _latest_tag_frame: list = [first_tag_frame]  # mutable container for thread sharing
    _tag_lock = threading.Lock()

    def _process_frame_and_tags(frame_bgr: "np.ndarray", tag_frame):
        """Apply tag overlay to frame_bgr; return (disp, status_dict, raw_tags_dicts)."""
        nonlocal boundary

        if tag_frame is not None and tag_frame.playfield_corners:
            raw_corners = tag_frame.playfield_corners
            if len(raw_corners) == 4:
                poly = np.array([[c[0], c[1]] for c in raw_corners], dtype=np.float32)
                boundary.polygon = poly
                boundary._poly = poly
                display._update_deskew(frame_bgr)

        homography: Optional[np.ndarray] = None
        if tag_frame is not None and tag_frame.homography is not None:
            try:
                homography = np.array(tag_frame.homography, dtype=np.float64)
                if homography.shape != (3, 3):
                    homography = None
            except Exception:
                homography = None

        disp = display.prepare_display(frame_bgr)

        raw_tags_dicts: list[dict] = []
        tags = []
        if tag_frame is not None:
            for tr in tag_frame.tags:
                try:
                    td = _tag_record_to_dict(tr)
                    raw_tags_dicts.append(td)
                    tags.append(_tag_dict_to_aprilcam(td))
                except Exception:
                    pass

        display.draw_overlays(disp, tags, homography)

        fps_val = tag_frame.fps if tag_frame is not None else 0.0
        calibrated = homography is not None
        deskew_mode = getattr(display, "_mode", "full") == "deskew"
        status_dict = {
            "fps": fps_val,
            "tag_count": len(tags),
            "calibrated": calibrated,
            "deskew_mode": deskew_mode,
        }
        return disp, status_dict, raw_tags_dicts

    first_disp, first_status, first_raw_tags = _process_frame_and_tags(
        first_frame, first_tag_frame
    )

    # Compute initial canvas height from the first display frame's aspect ratio
    _dh, _dw = first_disp.shape[:2]
    _display_h = int(round(_dh * _DISPLAY_W / _dw))

    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    frame_queue.put_nowait((first_disp, first_status, first_raw_tags))

    def _image_reader_thread():
        """Continuously read image frames and push processed results to frame_queue."""
        while not stop_event.is_set():
            try:
                frame_bgr = image_consumer.read()
            except (EOFError, RuntimeError):
                print("Image stream closed — daemon may have stopped.", file=sys.stderr)
                stop_event.set()
                break

            with _tag_lock:
                current_tag_frame = _latest_tag_frame[0]

            disp, status_dict, raw_tags = _process_frame_and_tags(
                frame_bgr, current_tag_frame
            )
            try:
                frame_queue.put_nowait((disp, status_dict, raw_tags))
            except queue.Full:
                pass

    def _tag_reader_thread():
        """Continuously read tag frames and update _latest_tag_frame."""
        while not stop_event.is_set():
            try:
                tf = tag_consumer.read()
                with _tag_lock:
                    _latest_tag_frame[0] = tf
            except (EOFError, RuntimeError):
                stop_event.set()
                break

    image_reader = threading.Thread(target=_image_reader_thread, daemon=True)
    tag_reader = threading.Thread(target=_tag_reader_thread, daemon=True)

    # ── Build tkinter window ──────────────────────────────────────────────
    import tkinter as tk
    import tkinter.font as tkfont
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"aprilcam view — {cam_name}")
    root.configure(bg="#111")
    root.resizable(False, True)

    # Top-level split: left (video, fixed size) | right (info panel, expands)
    left_frame = tk.Frame(root, bg="#111")
    left_frame.pack(side=tk.LEFT, fill=tk.Y)

    right_frame = tk.Frame(root, bg="#1e1e1e")
    right_frame.pack(side=tk.LEFT, fill=tk.Y)  # fixed width — no horizontal expansion

    # ── Left: canvas — always DISPLAY_W wide, height proportional ────────
    canvas = tk.Canvas(
        left_frame, width=_DISPLAY_W, height=_display_h,
        bg="black", highlightthickness=0,
    )
    canvas.pack()
    img_item = canvas.create_image(0, 0, anchor=tk.NW)

    # ── Right panel layout ────────────────────────────────────────────────
    mono = tkfont.Font(family="Courier", size=11)
    label_font = ("Helvetica", 10)
    value_font = ("Helvetica", 10, "bold")
    PANEL_BG = "#1e1e1e"
    FG = "#dddddd"
    MOB_FG = "#ffcc44"
    STAT_FG = "#88ccff"

    # ── Status block (top of right panel) ────────────────────────────────
    status_frame = tk.LabelFrame(
        right_frame, text="Camera Status",
        font=("Helvetica", 10, "bold"),
        fg="#aaaaaa", bg=PANEL_BG, padx=8, pady=6,
    )
    status_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

    def _kv_row(parent, row, key, init="--"):
        tk.Label(parent, text=key, font=label_font, fg="#aaaaaa", bg=PANEL_BG,
                 anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 12), pady=1)
        var = tk.StringVar(value=init)
        tk.Label(parent, textvariable=var, font=value_font, fg=FG, bg=PANEL_BG,
                 anchor="w").grid(row=row, column=1, sticky="w")
        return var

    var_fps = _kv_row(status_frame, 0, "FPS")
    var_tags = _kv_row(status_frame, 1, "Tags")
    var_cal = _kv_row(status_frame, 2, "Calibrated")
    var_deskew = _kv_row(status_frame, 3, "Deskew")

    # ── Mobile tags section ───────────────────────────────────────────────
    mob_frame = tk.LabelFrame(
        right_frame, text="Mobile Tags",
        font=("Helvetica", 10, "bold"),
        fg=MOB_FG, bg=PANEL_BG, padx=4, pady=4,
    )
    mob_frame.pack(fill=tk.X, padx=8, pady=(4, 2))

    mobile_text = tk.Text(
        mob_frame, font=mono, bg="#111", fg=MOB_FG,
        state=tk.DISABLED, height=8, width=44,
        relief=tk.FLAT, padx=4, pady=2, wrap=tk.NONE,
    )
    mob_sb = tk.Scrollbar(mob_frame, command=mobile_text.yview)
    mobile_text.configure(yscrollcommand=mob_sb.set)
    mob_sb.pack(side=tk.RIGHT, fill=tk.Y)
    mobile_text.pack(fill=tk.BOTH, expand=True)

    # ── Stationary tags section ───────────────────────────────────────────
    stat_outer = tk.LabelFrame(
        right_frame, text="Stationary Tags",
        font=("Helvetica", 10, "bold"),
        fg=STAT_FG, bg=PANEL_BG, padx=4, pady=4,
    )
    stat_outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))

    stat_text = tk.Text(
        stat_outer, font=mono, bg="#111", fg=STAT_FG,
        state=tk.DISABLED, height=8, width=44,
        relief=tk.FLAT, padx=4, pady=2, wrap=tk.NONE,
    )
    stat_sb = tk.Scrollbar(stat_outer, command=stat_text.yview)
    stat_text.configure(yscrollcommand=stat_sb.set)
    stat_sb.pack(side=tk.RIGHT, fill=tk.Y)
    stat_text.pack(fill=tk.BOTH, expand=True)

    # ── Mobility tracking (main-thread only) ──────────────────────────────
    _vel_counts: dict[int, int] = {}
    _perm_mobile: set[int] = set()
    _VEL_THRESHOLD = 1.0   # px/s
    _PROMOTE_FRAMES = 10

    def _set_text(widget, text: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    def _update_tag_panel(raw_tags: list[dict]) -> None:
        for t in raw_tags:
            tid = int(t.get("id", -1))
            if _vel_mag(t) > _VEL_THRESHOLD:
                _vel_counts[tid] = _vel_counts.get(tid, 0) + 1
                if _vel_counts[tid] >= _PROMOTE_FRAMES:
                    _perm_mobile.add(tid)

        mobile, stationary = [], []
        for t in sorted(raw_tags, key=lambda x: int(x.get("id", 0))):
            tid = int(t.get("id", -1))
            if tid in _perm_mobile or _vel_mag(t) > _VEL_THRESHOLD:
                mobile.append(t)
            else:
                stationary.append(t)

        mob_str = _MOB_HDR + "".join(_fmt_mobile_row(t) for t in mobile) if mobile else "(none)\n"
        st_str = _STAT_HDR + "".join(_fmt_stat_row(t) for t in stationary) if stationary else "(none)\n"
        _set_text(mobile_text, mob_str)
        _set_text(stat_text, st_str)

    # ── Window close handlers ─────────────────────────────────────────────
    def _on_close() -> None:
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.bind("<q>", lambda _e: _on_close())
    root.bind("<Escape>", lambda _e: _on_close())

    # ── Poll callback (tkinter main thread) ───────────────────────────────
    def _poll() -> None:
        if stop_event.is_set():
            try:
                root.destroy()
            except tk.TclError:
                pass
            return

        try:
            frame_bgr, status_dict, raw_tags = frame_queue.get_nowait()
        except queue.Empty:
            root.after(33, _poll)
            return

        # Scale to fixed display width, maintaining aspect ratio
        fh, fw = frame_bgr.shape[:2]
        dh = int(round(fh * _DISPLAY_W / fw))
        frame_disp = cv.resize(frame_bgr, (_DISPLAY_W, dh), interpolation=cv.INTER_AREA)

        # Resize canvas + window if the display height changed (e.g. homography engaged)
        if abs(dh - canvas.winfo_height()) > 2:
            canvas.config(height=dh)
            root.update_idletasks()
            _locked_w = _DISPLAY_W + right_frame.winfo_reqwidth()
            root.geometry(f"{_locked_w}x{root.winfo_reqheight()}")

        rgb = cv.cvtColor(frame_disp, cv.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.itemconfig(img_item, image=photo)
        canvas._photo_ref = photo

        fps_val = status_dict.get("fps")
        var_fps.set(f"{fps_val:.1f}" if isinstance(fps_val, (int, float)) else "--")
        var_tags.set(str(status_dict.get("tag_count", 0)))
        var_cal.set("Yes" if status_dict.get("calibrated") else "No")
        var_deskew.set("On" if status_dict.get("deskew_mode") else "Off")

        _update_tag_panel(raw_tags)

        root.after(33, _poll)

    # Snap window to exact content size: locked width = canvas + right panel.
    root.update_idletasks()
    _locked_w = _DISPLAY_W + right_frame.winfo_reqwidth()
    root.geometry(f"{_locked_w}x{_display_h}")
    root.resizable(False, True)

    image_reader.start()
    tag_reader.start()
    root.after(33, _poll)

    try:
        root.mainloop()
    finally:
        stop_event.set()
        try:
            image_consumer.close()
        except Exception:
            pass
        try:
            tag_consumer.close()
        except Exception:
            pass

    return 0
