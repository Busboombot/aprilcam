from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import dotenv_values
import cv2 as cv

from .camutil import (
    list_cameras,
    default_backends,
    select_camera_by_pattern,
)
import json
import numpy as np


@dataclass
class AppConfig:
    """
    Loads configuration from a .env file discovered by walking up from CWD.

    Required vars in .env:
      - JTL_APRILCAM=1  (guard to ensure the correct environment)
      - ROOT_DIR=<path|envdir>  (root of project; if 'envdir', use parent of .env file)

    Derived:
      - root_dir: Path
      - data_dir: Path (defaults to <root_dir>/data)
    """

    env_path: Path
    env: Dict[str, str]
    root_dir: Path
    data_dir: Path

    @staticmethod
    def find_env(start: Optional[Path] = None) -> Path:
        start = start or Path.cwd()
        cur = start.resolve()
        while True:
            candidate = cur / ".env"
            if candidate.exists():
                return candidate
            if cur.parent == cur:
                raise FileNotFoundError("No .env file found when walking up from CWD")
            cur = cur.parent

    @classmethod
    def load(cls, start: Optional[Path] = None) -> "AppConfig":
        env_path = cls.find_env(start)
        env_map: Dict[str, str] = {k: v for k, v in dotenv_values(env_path).items() if v is not None}

        # Guard variable required
        if env_map.get("JTL_APRILCAM") != "1":
            raise RuntimeError(".env missing required JTL_APRILCAM=1")

        # Resolve ROOT_DIR
        raw_root = env_map.get("ROOT_DIR")
        env_dir = env_path.parent
        if not raw_root or raw_root.strip().lower() == "envdir":
            root_dir = env_dir
        else:
            p = Path(raw_root)
            root_dir = p if p.is_absolute() else (env_dir / p)
        root_dir = root_dir.resolve()

        # Data directory (default ROOT_DIR/data)
        data_dir = Path(env_map.get("DATA_DIR", str(root_dir / "data"))).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        return cls(env_path=env_path, env=env_map, root_dir=root_dir, data_dir=data_dir)

    # --- Camera helpers ---
    def get_camera(
        self,
        arg: Optional[object] = None,
        *,
        backend: Optional[str] = None,
        max_cams: int = 10,
        quiet: bool = False,
    ) -> Optional[cv.VideoCapture]:
        """
        Resolve a camera from --camera arg (int or pattern) or .env CAMERA, open and return cv.VideoCapture.

        - arg may be int or str; if str and numeric, it's treated as an index; otherwise substring pattern.
        - backend can be one of: None/"auto", "avfoundation", "v4l2", "msmf", "dshow".
        - Falls back to the first available camera if none specified.
        Returns an opened VideoCapture or None on failure.
        """
        # Decode backend string to API preference
        be_map = {
            None: None,
            "auto": None,
            "avfoundation": getattr(cv, "CAP_AVFOUNDATION", 1200),
            "v4l2": getattr(cv, "CAP_V4L2", 200),
            "msmf": getattr(cv, "CAP_MSMF", 1400),
            "dshow": getattr(cv, "CAP_DSHOW", 700),
        }
        be_value = be_map.get(backend if backend is None or isinstance(backend, str) else None)
        # Choose backends list for enumeration
        backends = None if be_value is None else [be_value, getattr(cv, "CAP_ANY", 0)]
        # Parse input
        index: Optional[int] = None
        pattern: Optional[str] = None
        if arg is not None:
            if isinstance(arg, int):
                index = int(arg)
            else:
                s = str(arg).strip()
                try:
                    index = int(s)
                except ValueError:
                    pattern = s
        else:
            cam_env = self.env.get("CAMERA")
            if cam_env:
                s = str(cam_env).strip()
                try:
                    index = int(s)
                except ValueError:
                    pattern = s

        # If we have an index, try to open directly
        if index is not None:
            cap = cv.VideoCapture(int(index), 0 if be_value is None else int(be_value))
            if cap.isOpened():
                return cap
            cap.release()

        # Else enumerate and select by pattern or pick first
        cams = list_cameras(max_index=max_cams, backends=backends, quiet=quiet)
        if pattern:
            sel = select_camera_by_pattern(pattern, cams)
            if sel is not None:
                cap = cv.VideoCapture(int(sel), 0 if be_value is None else int(be_value))
                if cap.isOpened():
                    return cap
                cap.release()
        # Fallback to first camera if any
        if cams:
            cap = cv.VideoCapture(int(cams[0].index), 0 if be_value is None else int(be_value))
            if cap.isOpened():
                return cap
            cap.release()
        return None

    # --- Homography helpers ---
    def load_homography(self, path: Optional[Path] = None) -> Optional[np.ndarray]:
        """Load a 3x3 homography matrix from JSON. Defaults to <DATA_DIR>/homography.json.
        Returns numpy array (float64) or None if not found/invalid.
        """
        p = Path(path) if path else (self.data_dir / "homography.json")
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            H = np.array(data.get("homography", []), dtype=float)
            if H.shape == (3, 3):
                return H
        except Exception:
            return None
        return None
