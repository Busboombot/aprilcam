"""Live visualization subprocess for AprilCam.

Spawns a child process that opens a camera, runs tag detection with
PlayfieldDisplay overlays (house shape, direction arrow, velocity arrow,
tag ID), and shows a deskewed live view in an OpenCV window.

Three OS pipes connect parent and child:

- **data pipe** (child → parent): detection results sent back as JSON
  lines so the MCP server can populate its ring buffer.
- **stop pipe** (parent → child): parent writes ``"stop\\n"`` to signal
  the child to exit cleanly.
- **command pipe** (parent → child): parent writes line-delimited JSON
  commands (``add``, ``remove``, ``clear``) to mutate the set of
  agent-drawn paths rendered by the child each frame.

macOS requires OpenCV GUI (imshow/waitKey) on the main thread, so
the visualization must run in a separate process — not just a thread.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2 as cv
import numpy as np


def _child_main(
    camera_index: int,
    backend: Optional[int],
    pipe_fd: int,
    deskew: bool,
    family: str,
    proc_width: int,
    use_clahe: bool,
    use_sharpen: bool,
    stop_fd: int,
    robot_tag_id: Optional[int] = None,
    gripper_offset_cm: float = 14.0,
    cmd_fd: int = -1,
    initial_paths_json: str = "[]",
) -> None:
    """Entry point for the live-view child process.

    Runs on the child's main thread so OpenCV GUI works on macOS.
    Writes one JSON line per frame to *pipe_fd* with detection data.
    Exits when the window is closed (q/Esc) or *stop_fd* becomes readable.
    Reads JSON-line commands from *cmd_fd* to mutate the paths overlay.
    """
    # Late imports so the parent process doesn't load OpenCV GUI subsystem
    from aprilcam.core.aprilcam import AprilCam
    from aprilcam.ui.display import PlayfieldDisplay
    from aprilcam.core.playfield import Playfield

    pipe_out = os.fdopen(pipe_fd, "w", buffering=1)  # line-buffered
    stop_in = os.fdopen(stop_fd, "r")
    cmd_in = os.fdopen(cmd_fd, "r", buffering=1)  # line-buffered command pipe

    # Seed the paths dict from initial_paths_json (keyed by path_id)
    try:
        _initial = json.loads(initial_paths_json)
        paths: dict[str, dict] = {p["path_id"]: p for p in _initial}
    except Exception:
        paths = {}

    # Open camera
    cap = cv.VideoCapture(camera_index, 0 if backend is None else int(backend))
    if not cap.isOpened():
        pipe_out.write(json.dumps({"error": "Failed to open camera"}) + "\n")
        pipe_out.close()
        return

    # Load calibration homography for world coords and gripper overlay
    homography = None
    try:
        from aprilcam.calibration.calibration import load_calibration_for_camera
        from aprilcam.camera.camutil import get_device_name
        dev_name = get_device_name(camera_index)
        cal = load_calibration_for_camera(dev_name)
        if cal is not None:
            homography = cal.homography
    except Exception:
        pass

    cam = AprilCam(
        index=camera_index,
        backend=backend,
        speed_alpha=0.3,
        family=family,
        proc_width=proc_width,
        use_clahe=use_clahe,
        use_sharpen=use_sharpen,
        headless=True,  # we manage the window ourselves
        deskew_overlay=deskew,
        cap=cap,
        homography=homography,
    )

    display = PlayfieldDisplay(
        cam.playfield,
        window_name="aprilcam-live",
        headless=False,
        deskew_overlay=deskew,
        robot_tag_id=robot_tag_id,
        gripper_offset_cm=gripper_offset_cm,
    )

    cam.reset_state()

    def _drain_commands() -> bool:
        """Drain pending commands and check the stop signal.

        Uses a single select call on both pipes so neither starves the other.
        Returns True if the stop signal is readable (child should exit).
        Mutates *paths* in place for any command lines available on cmd_in.
        """
        import select
        r, _, _ = select.select([stop_in, cmd_in], [], [], 0)
        if stop_in in r:
            return True
        if cmd_in in r:
            try:
                line = cmd_in.readline()
                if line:
                    msg = json.loads(line.strip())
                    op = msg.get("op")
                    if op == "add":
                        p = msg["path"]
                        paths[p["path_id"]] = p
                    elif op == "remove":
                        paths.pop(msg["path_id"], None)
                    elif op == "clear":
                        paths.clear()
            except Exception:
                pass
        return False

    # Pipeline view modes: number keys switch the displayed image
    PIPE_MODES = {
        ord("0"): ("color",     "0:Color (raw)"),
        ord("1"): ("gray",      "1:Grayscale"),
        ord("2"): ("highpass",  "2:High-pass"),
        ord("3"): ("clahe",     "3:CLAHE"),
        ord("4"): ("hp+clahe",  "4:HP+CLAHE"),
        ord("5"): ("threshold", "5:Threshold"),
    }
    pipe_mode = "color"
    pipe_labels = [v[1] for v in PIPE_MODES.values()]

    try:
        while True:
            if _drain_commands():
                break

            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            tag_records = cam.process_frame(frame, now)

            # Build pipeline debug images from the raw frame
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            illum = cv.GaussianBlur(gray, (51, 51), 0).astype(np.float32)
            illum = np.maximum(illum, 1.0)
            flat = np.clip((gray.astype(np.float32) / illum) * 128.0, 0, 255).astype(np.uint8)
            clahe_img = cv.createCLAHE(3.0, (8, 8)).apply(gray)
            flat_clahe = cv.createCLAHE(3.0, (8, 8)).apply(flat)
            _, thresh_img = cv.threshold(flat, 150, 255, cv.THRESH_BINARY)

            # Always update playfield detection from the raw color frame
            display.playfield.update(frame)
            display._update_deskew(frame)
            display._ensure_window()

            # Select which image to display based on pipeline mode
            if pipe_mode == "color":
                view_frame = frame
            elif pipe_mode == "gray":
                view_frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            elif pipe_mode == "highpass":
                view_frame = cv.cvtColor(flat, cv.COLOR_GRAY2BGR)
            elif pipe_mode == "clahe":
                view_frame = cv.cvtColor(clahe_img, cv.COLOR_GRAY2BGR)
            elif pipe_mode == "hp+clahe":
                view_frame = cv.cvtColor(flat_clahe, cv.COLOR_GRAY2BGR)
            elif pipe_mode == "threshold":
                view_frame = cv.cvtColor(thresh_img, cv.COLOR_GRAY2BGR)
            else:
                view_frame = frame

            # Prepare display (deskew/crop) using the selected view
            disp = display.prepare_display(view_frame)

            # Draw overlays using playfield flows (which have velocity)
            flows = cam.playfield.get_flows()
            tags_for_overlay = list(flows.values())
            display.draw_overlays(disp, tags_for_overlay, homography=cam.homography)

            # Draw agent-defined paths (stub: T004 implements the body)
            display.draw_paths(disp, paths, cam.playfield, cam.homography)

            # Status panel on right side
            display.draw_status_panel(disp, tags_for_overlay, homography=cam.homography)

            # Pipeline mode menu at bottom-left
            fh, fw = disp.shape[:2]
            menu_y = fh - 8
            for label in reversed(pipe_labels):
                # Highlight active mode
                mode_key = label.split(":")[1].lower().replace(" (raw)", "").replace(" ", "")
                # Map label back to mode name
                mode_map = {"color": "color", "grayscale": "gray", "high-pass": "highpass",
                            "clahe": "clahe", "hp+clahe": "hp+clahe", "threshold": "threshold"}
                is_active = mode_map.get(mode_key, "") == pipe_mode
                color = (0, 255, 255) if is_active else (160, 160, 160)
                cv.putText(disp, label, (8, menu_y), cv.FONT_HERSHEY_SIMPLEX,
                           0.35, (0, 0, 0), 2, cv.LINE_AA)
                cv.putText(disp, label, (8, menu_y), cv.FONT_HERSHEY_SIMPLEX,
                           0.35, color, 1, cv.LINE_AA)
                menu_y -= 14

            # Show the frame
            display.show(disp)

            # Write detection data to pipe
            frame_data = {
                "timestamp": now,
                "frame_index": cam._frame_idx - 1,
                "tags": [tr.to_dict() for tr in tag_records],
            }
            try:
                pipe_out.write(json.dumps(frame_data) + "\n")
            except BrokenPipeError:
                break

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in PIPE_MODES:
                pipe_mode = PIPE_MODES[key][0]
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv.destroyAllWindows()
        except Exception:
            pass
        try:
            pipe_out.close()
        except Exception:
            pass
        try:
            stop_in.close()
        except Exception:
            pass
        try:
            cmd_in.close()
        except Exception:
            pass


class LiveViewProcess:
    """Manages a live-view child process and reads its detection data.

    The child process runs the OpenCV window on its main thread.
    Detection results flow back to the parent via a pipe and can be
    consumed by the MCP server's ring buffer.
    """

    def __init__(
        self,
        camera_index: int = 0,
        backend: Optional[int] = None,
        deskew: bool = True,
        family: str = "36h11",
        proc_width: int = 0,
        use_clahe: bool = False,
        use_sharpen: bool = False,
        robot_tag_id: Optional[int] = None,
        gripper_offset_cm: float = 14.0,
    ) -> None:
        self._camera_index = camera_index
        self._backend = backend
        self._deskew = deskew
        self._family = family
        self._proc_width = proc_width
        self._use_clahe = use_clahe
        self._use_sharpen = use_sharpen
        self._robot_tag_id = robot_tag_id
        self._gripper_offset_cm = gripper_offset_cm

        self._process: Optional[multiprocessing.Process] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._pipe_read_fd: Optional[int] = None
        self._stop_write_fd: Optional[int] = None
        self._cmd_write_fd: Optional[int] = None
        self._initial_paths: list[dict] = []
        self._callback: Optional[Any] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._process is not None and self._process.is_alive()

    def set_initial_paths(self, paths: list[dict]) -> None:
        """Store the initial paths to be sent to the child when start() is called.

        Must be called before :meth:`start`.  Each element should be the
        dict returned by ``Path.to_dict()``.
        """
        self._initial_paths = list(paths)

    def send_command(self, msg: dict) -> None:
        """Send a JSON command line to the child process.

        Is a no-op if the live view is not running or the command pipe
        file descriptor is not available.

        Args:
            msg: A dict with an ``"op"`` key (``"add"``, ``"remove"``,
                 or ``"clear"``) and any associated payload.
        """
        if self._cmd_write_fd is None or not self._running:
            return
        try:
            line = (json.dumps(msg) + "\n").encode()
            os.write(self._cmd_write_fd, line)
        except OSError:
            pass

    def start(self, on_frame: Optional[Any] = None) -> None:
        """Start the live view subprocess.

        Args:
            on_frame: Optional callback(frame_dict) called for each frame
                      of detection data received from the child.
        """
        if self.is_running:
            raise RuntimeError("Live view is already running")

        self._callback = on_frame

        # Pipe for detection data: child writes, parent reads
        data_r, data_w = os.pipe()
        # Pipe for stop signal: parent writes, child reads
        stop_r, stop_w = os.pipe()
        # Pipe for commands: parent writes, child reads
        cmd_r, cmd_w = os.pipe()

        self._pipe_read_fd = data_r
        self._stop_write_fd = stop_w
        self._cmd_write_fd = cmd_w

        self._process = multiprocessing.Process(
            target=_child_main,
            args=(
                self._camera_index,
                self._backend,
                data_w,
                self._deskew,
                self._family,
                self._proc_width,
                self._use_clahe,
                self._use_sharpen,
                stop_r,
                self._robot_tag_id,
                self._gripper_offset_cm,
                cmd_r,
                json.dumps(self._initial_paths),
            ),
            daemon=True,
        )
        self._process.start()
        self._running = True

        # Close the child's ends in the parent
        os.close(data_w)
        os.close(stop_r)
        os.close(cmd_r)

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._read_pipe, daemon=True
        )
        self._reader_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the child to stop and wait for it to exit."""
        self._running = False

        # Signal child via stop pipe
        if self._stop_write_fd is not None:
            try:
                os.write(self._stop_write_fd, b"stop\n")
            except OSError:
                pass
            try:
                os.close(self._stop_write_fd)
            except OSError:
                pass
            self._stop_write_fd = None

        # Close command pipe write end
        if self._cmd_write_fd is not None:
            try:
                os.close(self._cmd_write_fd)
            except OSError:
                pass
            self._cmd_write_fd = None

        # Wait for process
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None

        # Close read end of data pipe
        if self._pipe_read_fd is not None:
            try:
                os.close(self._pipe_read_fd)
            except OSError:
                pass
            self._pipe_read_fd = None

        # Wait for reader thread
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

    def _read_pipe(self) -> None:
        """Background thread: read JSON lines from the child's pipe."""
        if self._pipe_read_fd is None:
            return
        pipe_in = os.fdopen(self._pipe_read_fd, "r")
        self._pipe_read_fd = None  # fdopen takes ownership
        try:
            for line in pipe_in:
                if not self._running:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if self._callback:
                        self._callback(data)
                except (json.JSONDecodeError, Exception):
                    continue
        except Exception:
            pass
        finally:
            try:
                pipe_in.close()
            except Exception:
                pass


class LoopingVideoCapture:
    """Wraps cv2.VideoCapture to loop a video file continuously."""

    def __init__(self, path: str, fps: Optional[float] = None) -> None:
        self._path = path
        self._cap = cv.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._fps = fps or self._cap.get(cv.CAP_PROP_FPS) or 30.0
        self._delay = 1.0 / self._fps
        self._last_read = 0.0

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def read(self):
        # Throttle to video FPS so playback is realtime
        now = time.monotonic()
        elapsed = now - self._last_read
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_read = time.monotonic()

        ok, frame = self._cap.read()
        if not ok:
            # Loop: seek back to start
            self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return ok, frame

    def get(self, prop_id):
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        return self._cap.set(prop_id, value)

    def release(self):
        self._cap.release()


def run_live_view(
    camera_index: int = 0,
    backend: Optional[int] = None,
    video_path: Optional[str] = None,
    loop: bool = True,
    deskew: bool = True,
    family: str = "36h11",
    detect_aruco: bool = False,
    proc_width: int = 0,
    use_highpass: bool = True,
    use_clahe: bool = False,
    use_sharpen: bool = False,
    homography: Optional[np.ndarray] = None,
    color_camera: Optional[int] = None,
    robot_tag_id: Optional[int] = None,
    gripper_offset_cm: float = 14.0,
) -> None:
    """Run the live view directly (blocking) — for CLI use.

    Args:
        video_path: If provided, play this video file instead of a camera.
        loop: If True and video_path is set, loop the video continuously.
        color_camera: Optional camera index for color classification
            when the user presses 'd' for object detection.
        robot_tag_id: Tag ID of the robot. Draws a blue gripper circle.
        gripper_offset_cm: Distance from robot tag center to gripper (default 14).
    """
    from aprilcam.core.aprilcam import AprilCam

    if video_path is not None:
        import os
        if not os.path.isfile(video_path):
            print(f"Video file not found: {video_path}")
            return
        if loop:
            cap = LoopingVideoCapture(video_path)
        else:
            cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        source_label = f"video:{video_path}"
    else:
        cap = cv.VideoCapture(camera_index, 0 if backend is None else int(backend))
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            return
        source_label = f"camera:{camera_index}"

    print(f"Live view: {source_label} {'(looping)' if video_path and loop else ''}")

    cam = AprilCam(
        index=camera_index,
        backend=backend,
        speed_alpha=0.3,
        family=family,
        proc_width=proc_width,
        use_clahe=use_clahe,
        use_sharpen=use_sharpen,
        detect_aruco_4x4=detect_aruco,
        use_highpass=use_highpass,
        headless=False,
        deskew_overlay=deskew,
        print_tags=True,
        cap=cap,
        homography=homography,
        robot_tag_id=robot_tag_id,
        gripper_offset_cm=gripper_offset_cm,
    )

    cam.run(color_camera=color_camera)
