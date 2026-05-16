# AprilCam Daemon Interface Specification

`aprilcamd` is a long-running background process that owns all cameras,
runs AprilTag detection, and publishes per-frame data to any number of
subscribers over Unix domain sockets.

---

## Starting the Daemon

The daemon auto-spawns when any client (the viewer or MCP server) first
connects. You can also start it explicitly:

```
uv run python -m aprilcam.daemon
```

A CLI subcommand (`aprilcam daemon`) is planned but not yet implemented.

The daemon will not start a second instance — it uses an exclusive `flock`
on a pidfile. Log output goes to `<data_dir>/aprilcamd.log`.

---

## File Paths

All paths derive from `Config` (see `src/aprilcam/config.py`).  Defaults:

| Path | Default |
|------|---------|
| Control socket | `/tmp/aprilcam/control.sock` |
| Per-camera data socket | `/tmp/aprilcam/<cam_name>/data.sock` |
| Pidfile | `/tmp/aprilcam/aprilcamd.pid` |
| Per-camera info file | `<data_dir>/<cam_name>/info.json` |
| Per-camera paths file | `<data_dir>/<cam_name>/paths.json` |
| Daemon log | `<data_dir>/aprilcamd.log` |

`<data_dir>` defaults to `./data/runtime/` resolved from the working
directory.  Override with `APRILCAM_DATA_DIR` or `APRILCAM_SOCKET_DIR`.

---

## Control Socket

**Address:** `<socket_dir>/control.sock` (AF_UNIX, SOCK_STREAM)

**Protocol:** One request per connection.  The client connects, sends one
newline-terminated JSON object, reads one newline-terminated JSON object,
and the connection closes.

**Request envelope:**
```json
{"cmd": "<command>", ...args}
```

**Response envelope:**
```json
{"ok": true, ...fields}
{"ok": false, "error": "<message>"}
```

### Commands

#### `list_cameras`
Returns names of all currently open cameras.

Request:
```json
{"cmd": "list_cameras"}
```
Response:
```json
{"ok": true, "cameras": ["cam_0", "cam_2"]}
```

---

#### `open_camera`
Opens a camera by OpenCV device index and starts its capture/detection
pipeline.  If the camera is already open, returns immediately without
restarting.

Request:
```json
{"cmd": "open_camera", "index": 2}
```
Response:
```json
{
  "ok": true,
  "cam_name": "cam_2",
  "info_json_path": "/path/to/data/runtime/cam_2/info.json"
}
```

The `cam_name` is always `cam_<index>`.  After a successful response,
`info.json` and `data.sock` are ready.

---

#### `close_camera`
Stops the pipeline and releases the camera.

Request:
```json
{"cmd": "close_camera", "cam_name": "cam_2"}
```
Response:
```json
{"ok": true}
```

---

#### `get_camera_info`
Returns the current contents of `info.json` for a camera.

Request:
```json
{"cmd": "get_camera_info", "cam_name": "cam_2"}
```
Response:
```json
{
  "ok": true,
  "info": {
    "data_socket": "/tmp/aprilcam/cam_2/data.sock",
    "paths_file": "/abs/path/data/runtime/cam_2/paths.json",
    "device_name": "cam_2",
    "homography": [[...], [...], [...]],
    "calibrated": true,
    "frame_size": [1280, 800]
  }
}
```

---

#### `capture_frame`
Returns the most recent raw camera frame as a base64-encoded JPEG.
Does not run detection — this is the raw frame before processing.

Request:
```json
{"cmd": "capture_frame", "cam_name": "cam_2"}
```
Response:
```json
{"ok": true, "frame_b64": "<base64 JPEG>"}
```

---

#### `reload_calibration`
Reloads calibration from disk and updates the active pipeline.

Request:
```json
{"cmd": "reload_calibration", "cam_name": "cam_2"}
```
Response:
```json
{"ok": true}
```

---

#### `get_calibration_save_path`
Returns the path where `aprilcam calibrate` will write calibration data.

Request:
```json
{"cmd": "get_calibration_save_path"}
```
Response:
```json
{"ok": true, "path": "/abs/path/data/calibration.json"}
```

---

#### `shutdown`
Gracefully stops all pipelines and exits the daemon process.

Request:
```json
{"cmd": "shutdown"}
```
Response:
```json
{"ok": true}
```

---

## Data Socket (per camera)

**Address:** `<socket_dir>/<cam_name>/data.sock` (AF_UNIX, SOCK_STREAM)

**Protocol:** Connect and read a continuous stream of length-prefixed
msgpack messages.  The daemon fans out every frame to all connected
subscribers.  Slow subscribers have frames silently dropped (queue depth
capped at 2) — the subscriber always gets the latest available frame, not
a backlog.

### Wire Format

Each message is:
```
[4 bytes: big-endian uint32 length][<length> bytes: msgpack payload]
```

Read exactly 4 bytes to get the payload length, then read exactly that
many bytes and unpack with `msgpack.unpackb(data, raw=False)`.

### FrameMessage Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema` | int | Protocol version. Currently `1`. |
| `frame_id` | int | Monotonically increasing frame counter. |
| `ts_mono_ns` | int | `time.monotonic_ns()` at capture time. |
| `ts_wall_ms` | int | `time.time() * 1000` at capture time. |
| `frame_jpeg` | bytes | JPEG-encoded frame at quality 85. |
| `frame_w` | int | Frame width in pixels. |
| `frame_h` | int | Frame height in pixels. |
| `tags` | list[dict] | Detected AprilTag records (see below). |
| `homography` | list[list[float]] \| null | 3×3 homography matrix, or null if uncalibrated. |
| `playfield_corners` | list[[x,y]] | Playfield polygon corners in pixel coordinates, or `[]` if not detected. |
| `paths_file` | str | Absolute path to `paths.json`. Poll this file's mtime to detect updates. |
| `fps` | float | Rolling frame rate over the last 30 frames. |

### TagRecord Dict Fields

Each entry in `tags` is a dict with:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | AprilTag ID. |
| `corners_px` | [[x,y] × 4] | Corner pixel coordinates (UL, UR, LR, LL). |
| `center_px` | [x, y] | Center pixel coordinate. |
| `orientation_yaw` | float | Tag heading in radians. |
| `world_xy` | [x, y] \| null | World position in cm (null if uncalibrated). |
| `vel_px` | [vx, vy] | Velocity in pixels/second (smoothed). |
| `in_playfield` | bool | Whether the tag center is inside the playfield polygon. |

### Example Subscriber (Python)

```python
import socket
import struct
import msgpack

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/aprilcam/cam_2/data.sock")

while True:
    # Read length prefix
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("socket closed")
        raw_len += chunk
    length = struct.unpack(">I", raw_len)[0]

    # Read payload
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("socket closed")
        payload += chunk

    msg = msgpack.unpackb(payload, raw=False)
    print(f"frame {msg['frame_id']}: {len(msg['tags'])} tags at {msg['fps']:.1f} fps")
```

Or use the built-in helper:

```python
from aprilcam.daemon.protocol import read_frame
msg = read_frame(sock)   # returns a FrameMessage dataclass
```

---

## info.json

Written atomically to `<data_dir>/<cam_name>/info.json` when a camera is
opened.  Clients can read this file to discover the data socket path,
paths file path, and current calibration state without connecting to the
control socket.

```json
{
  "data_socket": "/tmp/aprilcam/cam_2/data.sock",
  "paths_file": "/abs/path/data/runtime/cam_2/paths.json",
  "device_name": "cam_2",
  "homography": [[0.135, 0.003, -37.6], [-0.003, 0.138, -6.3], [0.0, 0.0, 1.0]],
  "calibrated": true,
  "frame_size": [1280, 800]
}
```

---

## paths.json

Written atomically by the MCP server (not the daemon) whenever paths are
created, deleted, or cleared.  Subscribers poll this file's `mtime` each
frame and reload when it changes.

The path to this file is announced in every `FrameMessage` (`paths_file`
field) and in `info.json`.  The daemon never reads or writes this file.

**Format:** JSON array of path objects, or `[]` when no paths are active.

```json
[
  {
    "path_id": "path_000",
    "playfield_id": "pf_cam_2",
    "waypoints": [
      {
        "x": 20.0, "y": 15.0, "size_cm": 5.0,
        "symbol": "filled_circle",
        "symbol_color": [255, 0, 0],
        "line_color": [0, 200, 0]
      }
    ]
  }
]
```

Valid symbols: `square`, `filled_square`, `circle`, `filled_circle`,
`triangle`, `filled_triangle`, `x`, `none`.  Colors are RGB `[0..255]`.

---

## Configuration

Priority order (highest wins):

| Priority | Source |
|----------|--------|
| 4 (highest) | Environment variables prefixed `APRILCAM_` |
| 3 | `.env` file (walk up from CWD) |
| 2 | `.aprilcam` project dotfile (walk up from CWD) |
| 1 (lowest) | `~/.aprilcam` user dotfile |

| Key | Default | Description |
|-----|---------|-------------|
| `APRILCAM_SOCKET_DIR` | `/tmp/aprilcam/` | Directory for all Unix sockets and pidfile. |
| `APRILCAM_DATA_DIR` | `./data/runtime/` | Directory for info.json, paths.json, logs. |
| `APRILCAM_CALIBRATION_SOURCE` | `./data/calibration.json` | Calibration file path. |
| `APRILCAM_LOG_LEVEL` | `INFO` | Python logging level. |
| `APRILCAM_DAEMON_PIDFILE` | `<socket_dir>/aprilcamd.pid` | Pidfile path. |

---

## Known Limitations

- **Dead pipeline not auto-restarted.** If the camera is unplugged, the
  capture thread exits but the daemon still considers the camera registered.
  Calling `open_camera` again does nothing. Workaround: restart the daemon.

- **No `aprilcam daemon` CLI subcommand.** Use `python -m aprilcam.daemon`
  until the subcommand is implemented.

- **Daemon does not exit when idle.** It runs until explicitly shut down
  or killed, even when no cameras are open and no subscribers are connected.
