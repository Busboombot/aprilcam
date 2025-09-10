from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict
import subprocess
import re
import shutil
import os


import cv2 as cv


@dataclass
class CameraInfo:
    index: int
    name: str
    backend: Optional[str] = None


def default_backends() -> List[int]:
    if os.name == "nt":  # Windows
        return [getattr(cv, "CAP_MSMF", 1400), getattr(cv, "CAP_DSHOW", 700), getattr(cv, "CAP_ANY", 0)]
    elif sys.platform == "darwin":  # macOS
        return [getattr(cv, "CAP_AVFOUNDATION", 1200), getattr(cv, "CAP_ANY", 0)]
    else:  # Linux/Unix
        return [getattr(cv, "CAP_V4L2", 200), getattr(cv, "CAP_ANY", 0)]


class _SilenceStderr:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._orig_fd = None
        self._devnull_fd = None

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            # Duplicate original stderr fd
            self._orig_fd = os.dup(2)
            # Open devnull and redirect fd 2 there
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._devnull_fd, 2)
        except Exception:
            # If anything fails, best-effort: mark as disabled
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        try:
            if self._orig_fd is not None:
                os.dup2(self._orig_fd, 2)
        finally:
            if self._devnull_fd is not None:
                try:
                    os.close(self._devnull_fd)
                except Exception:
                    pass
            if self._orig_fd is not None:
                try:
                    os.close(self._orig_fd)
                except Exception:
                    pass
        return False


def _macos_avfoundation_device_names() -> Dict[int, str]:
    names: Dict[int, str] = {}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return names
    try:
        # ffmpeg prints device list to stderr
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        text = proc.stderr
        # Parse lines under "AVFoundation video devices:" like "[0] FaceTime HD Camera"
        in_video = False
        for line in text.splitlines():
            if "AVFoundation video devices:" in line:
                in_video = True
                continue
            if "AVFoundation audio devices:" in line:
                in_video = False
            if not in_video:
                continue
            m = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
            if m:
                idx = int(m.group(1))
                nm = m.group(2).strip()
                names[idx] = nm
    except Exception:
        pass
    return names


def macos_avfoundation_device_names() -> Dict[int, str]:
    """Public helper to fetch AVFoundation device names via ffmpeg.
    Keys are the AVFoundation indices used by ffmpeg/imagesnap, not necessarily cv2 index mapping for CAP_ANY.
    """
    if sys.platform != "darwin":
        return {}
    return _macos_avfoundation_device_names()


def list_cameras(max_index: int = 10, backends: Optional[List[int]] = None, stop_after_failures: int = 4, quiet: bool = False, detailed_names: bool = False) -> List[CameraInfo]:
    cameras: List[CameraInfo] = []
    backends = backends or default_backends()
    backend_failures: Dict[int, int] = {be: 0 for be in backends}
    av_names: Dict[int, str] = {}
    if sys.platform == "darwin" and detailed_names:
        av_names = _macos_avfoundation_device_names()
    for idx in range(max_index):
        for be in list(backends):
            # Skip overly chatty/invalid indices for AVFoundation, but allow CAP_ANY for >2 to support multiple cams.
            if sys.platform == "darwin" and be == getattr(cv, "CAP_AVFOUNDATION", 1200) and idx >= 2:
                continue
            # If this backend has too many consecutive failures, skip further attempts to reduce noise
            if backend_failures.get(be, 0) >= max(1, int(stop_after_failures)) and idx > 1:
                continue
            with _SilenceStderr(quiet):
                cap = cv.VideoCapture(idx, be)
                try:
                    if cap.isOpened():
                        backend_name = None
                        try:
                            backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else None
                        except Exception:
                            pass
                        # Prefer detailed AVFoundation device name if available
                        pretty = None
                        if sys.platform == "darwin" and backend_name == "AVFOUNDATION" and idx in av_names:
                            pretty = av_names.get(idx)
                        name = (pretty or f"Camera {idx}") + (f" ({backend_name})" if backend_name else "")
                        cameras.append(CameraInfo(index=idx, name=name, backend=backend_name))
                        backend_failures[be] = 0
                        break
                    else:
                        backend_failures[be] = backend_failures.get(be, 0) + 1
                finally:
                    cap.release()
    return cameras


def select_camera_by_pattern(pattern: Optional[str], cameras: List[CameraInfo]) -> Optional[int]:
    if not pattern:
        return None
    pat = pattern.strip().lower()
    # Direct index forms: "@2", "#2", or plain "2"
    if pat.startswith("@") or pat.startswith("#"):
        try:
            return int(pat[1:])
        except ValueError:
            pass
    try:
        # If the whole string is an int, use as index
        return int(pat)
    except ValueError:
        pass
    for cam in cameras:
        if pat in cam.name.lower():
            return cam.index
    return None
