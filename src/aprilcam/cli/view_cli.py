"""CLI subcommand: aprilcam view — Live view via the AprilCam daemon."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import socket
import sys
import threading
from pathlib import Path
from typing import Optional


def _tag_dict_to_aprilcam(tag_dict: dict):
    """Convert a TagRecord dict (from FrameMessage.tags) into an AprilTag object."""
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
    args = parser.parse_args(argv)

    import numpy as np
    import cv2 as cv

    from aprilcam.config import Config
    from aprilcam.daemon.client import ensure_running
    from aprilcam.daemon.protocol import read_frame
    from aprilcam.core.playfield import PlayfieldBoundary
    from aprilcam.ui.display import PlayfieldDisplay

    config = Config.load()
    client = ensure_running(config)

    cam_name: Optional[str] = None
    try:
        camera_arg = args.camera
        try:
            cam_index = int(camera_arg)
            resp = client.rpc("open_camera", index=cam_index)
            cam_name = resp.get("cam_name") or resp.get("name")
        except ValueError:
            resp = client.rpc("get_camera_info", cam_name=camera_arg)
            cam_name = resp.get("cam_name") or camera_arg

        if cam_name is None:
            print(f"Error: could not resolve camera '{camera_arg}'", file=sys.stderr)
            client.close()
            return 1

        info_resp = client.rpc("get_camera_info", cam_name=cam_name)
        info_data = info_resp.get("info", {})
        data_socket_path: Optional[str] = info_data.get("data_socket")
        paths_file_path: Optional[str] = info_data.get("paths_file")

        if not data_socket_path:
            print(
                f"Error: no data socket found for camera '{cam_name}'. "
                "Is the detection loop running?",
                file=sys.stderr,
            )
            client.close()
            return 1

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        client.close()
        return 1
    finally:
        client.close()

    data_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        data_sock.connect(data_socket_path)
    except OSError as exc:
        print(f"Error: cannot connect to data socket '{data_socket_path}': {exc}", file=sys.stderr)
        return 1

    paths: dict = {}
    paths_file: Optional[Path] = Path(paths_file_path) if paths_file_path else None
    paths_mtime: Optional[float] = None
    if paths_file is not None and paths_file.exists():
        paths = _load_paths(paths_file)
        try:
            paths_mtime = os.stat(paths_file).st_mtime
        except OSError:
            pass

    try:
        first_msg = read_frame(data_sock)
    except ConnectionError:
        print("Data socket closed before first frame — daemon may have stopped.", file=sys.stderr)
        data_sock.close()
        return 1

    first_frame = _decode_frame(first_msg.frame_jpeg, np, cv)
    if first_frame is None:
        print("Error: could not decode first frame from daemon.", file=sys.stderr)
        data_sock.close()
        return 1

    frame_h, frame_w = first_frame.shape[:2]

    boundary = PlayfieldBoundary()
    display = PlayfieldDisplay(
        playfield=boundary,
        window_name="aprilcam view",
        deskew_overlay=True,
    )

    def _process_msg(msg, frame_bgr):
        nonlocal paths, paths_mtime

        if paths_file is not None and paths_file.exists():
            try:
                mtime = os.stat(paths_file).st_mtime
                if mtime != paths_mtime:
                    paths = _load_paths(paths_file)
                    paths_mtime = mtime
            except OSError:
                pass

        if msg.playfield_corners:
            poly = np.array(msg.playfield_corners, dtype=np.float32)
            if poly.shape == (4, 2):
                boundary.polygon = poly
                boundary._poly = poly
                display._update_deskew(frame_bgr)

        homography: Optional[np.ndarray] = None
        if msg.homography is not None:
            try:
                homography = np.array(msg.homography, dtype=np.float64)
                if homography.shape != (3, 3):
                    homography = None
            except Exception:
                homography = None

        disp = display.prepare_display(frame_bgr)

        tags = []
        for td in msg.tags:
            try:
                tags.append(_tag_dict_to_aprilcam(td))
            except Exception:
                pass

        display.draw_overlays(disp, tags, homography)
        if paths and homography is not None:
            display.draw_paths(disp, paths, boundary, homography)

        calibrated = homography is not None
        deskew_mode = getattr(display, "_deskew_active", False)
        status_dict = {
            "fps": msg.fps,
            "tag_count": len(tags),
            "calibrated": calibrated,
            "deskew_mode": deskew_mode,
        }
        return disp, status_dict, list(msg.tags)

    first_disp, first_status, first_tags = _process_msg(first_msg, first_frame)

    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    frame_queue.put_nowait((first_disp, first_status, first_tags))

    def _reader_thread():
        while not stop_event.is_set():
            try:
                msg = read_frame(data_sock)
            except ConnectionError:
                print("Data socket closed — daemon may have stopped.", file=sys.stderr)
                stop_event.set()
                break
            frame_bgr = _decode_frame(msg.frame_jpeg, np, cv)
            if frame_bgr is None:
                continue
            disp, status_dict, raw_tags = _process_msg(msg, frame_bgr)
            try:
                frame_queue.put_nowait((disp, status_dict, raw_tags))
            except queue.Full:
                pass

    reader = threading.Thread(target=_reader_thread, daemon=True)

    # ── Build tkinter window ──────────────────────────────────────────────
    import tkinter as tk
    import tkinter.font as tkfont
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"aprilcam view — {cam_name}")
    root.configure(bg="#111")
    root.resizable(True, True)

    # Top-level split: left (video, fixed size) | right (info panel, expands)
    left_frame = tk.Frame(root, bg="#111")
    left_frame.pack(side=tk.LEFT, fill=tk.Y)

    right_frame = tk.Frame(root, bg="#1e1e1e")
    right_frame.pack(side=tk.LEFT, fill=tk.Y)  # fixed width — no horizontal expansion

    # ── Left: canvas only (camera has fixed resolution) ──────────────────
    canvas = tk.Canvas(
        left_frame, width=frame_w, height=frame_h,
        bg="black", highlightthickness=0,
    )
    canvas.pack()
    img_item = canvas.create_image(0, 0, anchor=tk.NW)

    # ── Right panel layout ────────────────────────────────────────────────
    mono = tkfont.Font(family="Courier", size=11)
    label_font = ("Helvetica", 10)
    value_font = ("Helvetica", 10, "bold")
    PANEL_BG = "#1e1e1e"
    SECT_BG = "#252525"
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

        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
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

    # Snap window to natural content size so it doesn't default to full-screen.
    # Camera canvas has a fixed resolution; there's no reason for the window
    # to be wider than canvas + right panel.
    root.update_idletasks()
    root.geometry(f"{root.winfo_reqwidth()}x{root.winfo_reqheight()}")
    root.resizable(False, False)

    reader.start()
    root.after(33, _poll)

    try:
        root.mainloop()
    finally:
        stop_event.set()
        try:
            data_sock.close()
        except OSError:
            pass

    return 0
