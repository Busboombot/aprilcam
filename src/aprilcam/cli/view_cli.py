"""CLI subcommand: aprilcam view — Live view via the AprilCam daemon."""

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import threading
from pathlib import Path
from typing import Optional


def _tag_dict_to_aprilcam(tag_dict: dict):
    """Convert a TagRecord dict (from FrameMessage.tags) into an AprilTag object.

    Only the fields that draw_overlays actually reads are populated:
      - id, corners_px, center_px, top_dir_px, orientation_yaw, world_xy, vel_px
    """
    import math
    import numpy as np
    from aprilcam.core.models import AprilTag

    corners_raw = tag_dict.get("corners_px", [[0, 0]] * 4)
    corners_px = np.array(corners_raw, dtype=np.float32)

    center_raw = tag_dict.get("center_px", [0.0, 0.0])
    center_px = (float(center_raw[0]), float(center_raw[1]))

    # Recompute top_dir_px from corners (same logic as AprilTag.from_corners)
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

    # Attach vel_px as an attribute so draw_overlays can use it for the arrow
    vel_raw = tag_dict.get("vel_px")
    if vel_raw is not None:
        tag.vel_px = (float(vel_raw[0]), float(vel_raw[1]))
    else:
        tag.vel_px = (0.0, 0.0)

    return tag


def _load_paths(paths_file: Path) -> dict:
    """Load paths JSON and return a dict keyed by path_id.

    Handles both dict format  {path_id: {path dict}, ...}
    and list format           [{path dict}, ...].
    """
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
    """Decode JPEG bytes to a BGR numpy array. Returns None on failure."""
    if isinstance(frame_bytes, (bytes, bytearray)):
        buf = np.frombuffer(frame_bytes, dtype=np.uint8)
    else:
        buf = np.array(frame_bytes, dtype=np.uint8)
    return cv.imdecode(buf, cv.IMREAD_COLOR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aprilcam view",
        description="Open a live view window fed by the AprilCam daemon",
    )
    parser.add_argument(
        "camera",
        metavar="CAMERA",
        help="Camera name or integer index",
    )
    args = parser.parse_args(argv)

    import numpy as np
    import cv2 as cv

    from aprilcam.config import Config
    from aprilcam.daemon.client import ensure_running
    from aprilcam.daemon.protocol import read_frame
    from aprilcam.core.playfield import PlayfieldBoundary
    from aprilcam.ui.display import PlayfieldDisplay

    # 1. Load config and ensure the daemon is running
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

        # 3. Get camera info (data socket path, paths file path)
        info_resp = client.rpc("get_camera_info", cam_name=cam_name)
        # Daemon returns {"ok": True, "info": {data_socket, paths_file, ...}}
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
        # We no longer need the control connection for the view loop
        client.close()

    # 5. Connect to the data socket (AF_UNIX)
    data_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        data_sock.connect(data_socket_path)
    except OSError as exc:
        print(f"Error: cannot connect to data socket '{data_socket_path}': {exc}", file=sys.stderr)
        return 1

    # 6. Pre-load paths if the paths file already exists
    paths: dict = {}
    paths_file: Optional[Path] = Path(paths_file_path) if paths_file_path else None
    paths_mtime: Optional[float] = None

    if paths_file is not None and paths_file.exists():
        paths = _load_paths(paths_file)
        try:
            paths_mtime = os.stat(paths_file).st_mtime
        except OSError:
            pass

    # 7. Bootstrap: read ONE frame synchronously to get frame dimensions
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

    # 8. Build PlayfieldDisplay using the first frame
    boundary = PlayfieldBoundary()
    display = PlayfieldDisplay(
        playfield=boundary,
        window_name="aprilcam view",
        deskew_overlay=True,
    )

    def _process_msg(msg, frame_bgr):
        """Apply display pipeline to a frame+msg, return (disp_frame, status_dict)."""
        nonlocal paths, paths_mtime

        # Reload paths if the file has changed
        if paths_file is not None and paths_file.exists():
            try:
                mtime = os.stat(paths_file).st_mtime
                if mtime != paths_mtime:
                    paths = _load_paths(paths_file)
                    paths_mtime = mtime
            except OSError:
                pass

        # Update playfield polygon from message corners
        if msg.playfield_corners:
            poly = np.array(msg.playfield_corners, dtype=np.float32)
            if poly.shape == (4, 2):
                boundary.polygon = poly
                boundary._poly = poly
                display._update_deskew(frame_bgr)

        # Convert homography
        homography: Optional[np.ndarray] = None
        if msg.homography is not None:
            try:
                homography = np.array(msg.homography, dtype=np.float64)
                if homography.shape != (3, 3):
                    homography = None
            except Exception:
                homography = None

        # Prepare display frame (crop or deskew to playfield)
        disp = display.prepare_display(frame_bgr)

        # Convert tag dicts to AprilTag objects
        tags = []
        for td in msg.tags:
            try:
                tags.append(_tag_dict_to_aprilcam(td))
            except Exception:
                pass

        # Draw overlays onto display frame
        display.draw_overlays(disp, tags, homography)

        # Draw paths (only when homography is available)
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

        return disp, status_dict

    # Process the first frame
    first_disp, first_status = _process_msg(first_msg, first_frame)

    # 9. Set up shared state for reader <-> tkinter communication
    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Put the first frame in the queue immediately
    frame_queue.put_nowait((first_disp, first_status))

    # 10. Reader thread: blocks on read_frame, decodes, processes, enqueues
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

            disp, status_dict = _process_msg(msg, frame_bgr)

            try:
                frame_queue.put_nowait((disp, status_dict))
            except queue.Full:
                pass  # Drop frame — tkinter is catching up

    reader = threading.Thread(target=_reader_thread, daemon=True)

    # 11. Build tkinter window
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"aprilcam view — {cam_name}")

    canvas = tk.Canvas(root, width=frame_w, height=frame_h, bg="black", highlightthickness=0)
    canvas.pack()

    # Status bar below canvas
    status_bar = tk.Frame(root, bg="#222")
    status_bar.pack(fill=tk.X)

    lbl_fps = tk.Label(status_bar, text="FPS: --", fg="white", bg="#222", padx=8)
    lbl_fps.pack(side=tk.LEFT)

    lbl_tags = tk.Label(status_bar, text="Tags: --", fg="white", bg="#222", padx=8)
    lbl_tags.pack(side=tk.LEFT)

    lbl_cal = tk.Label(status_bar, text="Calibrated: --", fg="white", bg="#222", padx=8)
    lbl_cal.pack(side=tk.LEFT)

    lbl_deskew = tk.Label(status_bar, text="Deskew: --", fg="white", bg="#222", padx=8)
    lbl_deskew.pack(side=tk.LEFT)

    # Place initial blank image on canvas
    img_item = canvas.create_image(0, 0, anchor=tk.NW)

    def _on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.bind("<q>", lambda _e: _on_close())
    root.bind("<Escape>", lambda _e: _on_close())

    def _poll():
        if stop_event.is_set():
            try:
                root.destroy()
            except tk.TclError:
                pass
            return

        try:
            frame_bgr, status_dict = frame_queue.get_nowait()
        except queue.Empty:
            root.after(33, _poll)
            return

        # Convert BGR -> RGB -> PIL -> PhotoImage
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)
        canvas.itemconfig(img_item, image=photo)
        canvas._photo_ref = photo  # prevent GC

        # Update status labels
        fps_val = status_dict.get("fps")
        fps_str = f"{fps_val:.1f}" if isinstance(fps_val, (int, float)) else "--"
        lbl_fps.config(text=f"FPS: {fps_str}")
        lbl_tags.config(text=f"Tags: {status_dict.get('tag_count', 0)}")
        lbl_cal.config(text=f"Calibrated: {'Yes' if status_dict.get('calibrated') else 'No'}")
        lbl_deskew.config(text=f"Deskew: {'On' if status_dict.get('deskew_mode') else 'Off'}")

        root.after(33, _poll)

    # 12. Start reader thread and tkinter main loop
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
