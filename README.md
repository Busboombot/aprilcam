# aprilcam

CLI to list available cameras and detect AprilTags from a USB video stream using OpenCV.

## Setup

This project is configured for uv (https://docs.astral.sh/uv/), but works with any PEP 517 tool as it uses a standard `pyproject.toml`.

### With uv

1. Create/activate a virtual environment (uv will do this automatically if you use `uv run`).
2. Install dependencies:

```bash
uv pip install -e .
```

Or run directly without installing:

```bash
uv run aprilcam --help
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

List and select a camera (default remembers the previous selection; press Enter to accept):

```bash
aprilcam
```

Skip prompt and use a specific camera index:

```bash
aprilcam --camera 0
```

Adjust how many indices to probe (0..N-1):

```bash
aprilcam --max-cams 15
```

Specify a capture backend explicitly (helpful if detection fails):

```bash
aprilcam --backend avfoundation   # macOS
aprilcam --backend v4l2           # Linux
aprilcam --backend msmf           # Windows
```

Controls:
- Press `q` or `Esc` to quit the video window.

Notes:
- Requires `opencv-contrib-python` for the ArUco AprilTag detectors.
- On macOS you may need to grant camera permission to Terminal/VS Code (System Settings > Privacy & Security > Camera). The first probe may trigger a permission prompt.

## How it works

- Enumerates cameras by trying indices from 0 up to `--max-cams` and checking if they open.
- Uses OpenCV's ArUco AprilTag dictionaries (16h5, 25h9, 36h10, 36h11) to detect tags.
- Draws a red box around each detected tag and overlays the detected ID.
- Remembers the last selected camera in `~/.config/aprilcam/config.json`.

### Generate AprilTag images (36h11)

Create 800x800 px tags with a label line below into `april-tag-images/`:

```bash
taggen --out-dir april-tag-images --start 0 --end 586 --size 800
```

This produces files like `tag36h11_<id>.png` with a white background, black tag, and a centered "ID: <id>" text below the tag.

## License

MIT
