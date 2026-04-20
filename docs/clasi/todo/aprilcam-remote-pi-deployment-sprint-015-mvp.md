---
status: pending
---

# AprilCam Remote-Pi Deployment — Sprint 015 MVP

## Context

Camera enumeration is fragile on the Mac (USB hubs shuffle indices), and
tuning exposure on a Mac-attached USB camera is currently impossible —
`Camera` only sets width/height. The user wants to move cameras onto a
Raspberry Pi (Pi 4/5, USB + CSI ribbon) that the Mac reaches over LAN,
so the rig is portable and the Mac stops owning the cameras.

Three concrete needs fall out of that move:

1. **Programmatic camera tuning** (exposure first, gain/white-balance/etc.
   close behind) exposed as MCP tools.
2. **Bandwidth-efficient frame return** — the default 1920×1080 BGR JPEG
   (~250 KB) is too heavy to poll over Wi-Fi. Need `scale` (1/2/4/8 int
   downsample) and `grayscale` options; scale=4 grayscale ≈ 15 KB.
3. **Remote viewing of the processed feed** ("IP-camera for the annotated
   playfield view") — an MJPEG stream that taps the detection loop's
   `last_frame`, renders tag overlays, and pushes multipart JPEG.

A lot of the infrastructure already exists. The new sprint wires these
three features into it and adds CSI/libcamera support so ribbon cameras
work alongside USB on the Pi. Trusted-LAN, no auth. Settings persistence
deferred to Sprint 016.

## What already exists (don't rebuild)

- `aprilcam web --host 0.0.0.0 --port 17439` — Starlette + Uvicorn
  HTTP/WebSocket server at [src/aprilcam/server/web_server.py](src/aprilcam/server/web_server.py),
  with FastMCP mounted at `/mcp/sse` (line 755,761).
- WebSocket tag stream at `/ws/tags/{source_id}` (web_server.py:693-736).
- `DetectionLoop.last_frame` is already a public `@property` at
  [src/aprilcam/core/detection.py:245-248](src/aprilcam/core/detection.py#L245-L248) — the MJPEG endpoint can read it cleanly.
- `format_image_output()` at [src/aprilcam/server/mcp_server.py:241-268](src/aprilcam/server/mcp_server.py#L241-L268)
  is the single JPEG-encode chokepoint — extend here, not in each handler.
- Platform-aware capture backends already present
  ([src/aprilcam/camera/camutil.py](src/aprilcam/camera/camutil.py)):
  V4L2 on Linux, AVFoundation on Mac. Pi USB works today.
- `VideoCamera` test rig (Sprint 014) reads `.mov` files — Pi-side
  regression harness without hardware. [src/aprilcam/camera/video_camera.py](src/aprilcam/camera/video_camera.py).
- `pyproject.toml` deps (opencv-contrib-python, numpy, fastmcp,
  starlette, uvicorn, websockets) all have ARM64 wheels. No blockers.

## Deployment architecture

**Primary transport: HTTP/SSE.** Pi runs `aprilcam web --host 0.0.0.0`
as a systemd service. Mac's MCP client connects to
`http://pi.local:17439/mcp/sse`. Detection loops live in that long-lived
process, so camera captures stay warm across Mac reconnects (re-opening
V4L2/libcamera is slow and sometimes races). Browser on Mac also hits
the same host for the MJPEG live view.

**Stdio+SSH remains documented** as a cold-start debug path
(`ssh pi@... aprilcam mcp`) but isn't the expected daily mode — it
respawns the server on every connect and loses detection state.

## Implementation plan

### Ticket 1 — Camera-property MCP tools (USB/V4L2 path)

Extend `Camera` at [src/aprilcam/camera/camera.py](src/aprilcam/camera/camera.py):

- Add a whitelist dict mapping canonical names → cv2 property IDs:
  `exposure`, `gain`, `brightness`, `contrast`, `saturation`,
  `white_balance`, `auto_exposure`, `auto_white_balance`, `fps`,
  `sharpness`, `backlight`.
- `Camera.set_property(name, value) -> {requested, applied, supported}`
  — round-trip check: `get`, `set`, `get`; `supported=True` only if the
  second `get` reflects the request.
- `Camera.get_property(name)` and `Camera.list_properties()`.
- V4L2 quirk: setting `exposure` requires `auto_exposure` in manual
  mode (value 1 or 3 depending on driver). `set_camera_settings`
  auto-disables auto-exposure when caller passes `exposure=` without
  `auto_exposure=`.

New MCP tools in [src/aprilcam/server/mcp_server.py](src/aprilcam/server/mcp_server.py)
(and REST mirrors in [src/aprilcam/server/web_server.py](src/aprilcam/server/web_server.py)
via `_TOOL_SPECS` + `_HANDLERS`):

```
set_camera_property(camera_id, property, value)
get_camera_property(camera_id, property)
list_camera_properties(camera_id)
set_camera_settings(camera_id, *, exposure=None, gain=None,
                    white_balance=None, auto_exposure=None,
                    auto_white_balance=None, brightness=None,
                    contrast=None, saturation=None, fps=None,
                    sharpness=None)
```

`set_camera_settings` returns `{applied: {...}, unsupported: [...]}` so
the agent learns which knobs the hardware accepts.

### Ticket 2 — CSI / libcamera backend (Pi ribbon cameras)

`cv2.VideoCapture` on the Pi does not drive CSI cameras reliably —
they need `picamera2` (which uses libcamera under the hood). Add a
parallel camera class so the rest of the stack doesn't care.

- New file `src/aprilcam/camera/csi_camera.py` — `CSICamera` with the
  same `.read() / .release() / .set_property() / .list_properties()`
  surface as `Camera`. Wraps `picamera2.Picamera2`. Property names map
  to picamera2 controls: `exposure` → `ExposureTime` (µs),
  `gain` → `AnalogueGain`, `auto_exposure` → `AeEnable`,
  `white_balance` → `ColourGains`, `auto_white_balance` → `AwbEnable`,
  etc.
- Extend [src/aprilcam/camera/camutil.py](src/aprilcam/camera/camutil.py)
  `list_cameras()` to append CSI entries via
  `Picamera2.global_camera_info()` when the import succeeds; tag them
  `backend="libcamera"` and use indices that don't collide with V4L2.
- `open_camera` tool picks the class by `backend`: `"libcamera"` →
  `CSICamera`, everything else → existing `Camera`. Downstream
  (`DetectionLoop`, `format_image_output`, handlers) is unchanged
  because it only calls `.read()`.
- `pyproject.toml`: add an optional extra `[project.optional-dependencies] pi = ["picamera2"]`
  so `pipx install 'aprilcam[pi]'` pulls it on the Pi; Mac installs
  skip it. Guard the import with try/except → graceful no-CSI on Mac.

### Ticket 3 — Frame scale + grayscale

Extend `format_image_output()` at
[src/aprilcam/server/mcp_server.py:241](src/aprilcam/server/mcp_server.py#L241):

```python
def format_image_output(frame, format="base64", quality=85,
                       scale: int = 1, grayscale: bool = False):
    if scale not in (1, 2, 4, 8):
        raise ValueError("scale must be one of 1, 2, 4, 8")
    if scale > 1:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (w // scale, h // scale),
                           interpolation=cv2.INTER_AREA)
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # existing JPEG encode path handles 1-channel fine
```

Add `scale` and `grayscale` params to `capture_frame`, `get_frame`,
`get_frame_image`, and `get_composite_frame` tool signatures. REST
mirrors in `web_server.py` get matching query params. Annotation order:
overlay on full-res, then downsample (preserves tag-corner legibility).

### Ticket 4 — MJPEG live processed-frame stream

New endpoint in [src/aprilcam/server/web_server.py](src/aprilcam/server/web_server.py):

```
GET /api/stream/{source_id}?annotate=true&scale=2&grayscale=false
                           &fps=10&quality=75
```

Content-Type `multipart/x-mixed-replace; boundary=frame`.
Starlette `StreamingResponse` with an async generator that:

1. Looks up the `DetectionLoop` for `source_id`. 404 if none.
2. Loops: read `loop.last_frame`, if new frame (compare `frame_count`),
   run the same overlay helper that `_handle_get_frame` uses for
   `annotate=True`, pass through `format_image_output` (scale/grayscale/
   quality), yield multipart chunk. If frame unchanged,
   `await asyncio.sleep(1/fps)`.
3. Pace with `asyncio.sleep(1/fps)` between writes — drop frames rather
   than queue. Back-pressure: if client is slow, the `await` on write
   naturally blocks and the sleep loop drops the next frame.

**Refactor prereq:** extract the overlay logic currently inside
`_handle_get_frame` (when `annotate=True`) into a shared
`render_annotated(frame, source_id)` function in `mcp_server.py` or
a new `src/aprilcam/vision/annotate.py`. The MJPEG generator and the
single-frame path both call it. Non-negotiable — duplicating overlay
code splits the truth.

Rewire the built-in HTML UI's `<img id="liveImg">` to use the stream
URL instead of polling `/api/get_frame`. One-line change in the
HTML template inside `web_server.py`.

### Ticket 5 — Pi deployment artifacts

New directory `deploy/pi/`:

- `aprilcam-web.service` — systemd unit:
  - `User=aprilcam`, member of `video` group for V4L2 access.
  - `ExecStart=/home/aprilcam/.local/bin/aprilcam web --host 0.0.0.0 --port 17439`.
  - `Restart=on-failure`, `RestartSec=5`.
  - `Environment=` for `APRILCAM_BIND_HOST`, `APRILCAM_BIND_PORT`.
- `install.sh` — idempotent: `apt install python3-venv v4l-utils libgl1 python3-picamera2 avahi-daemon`,
  create `aprilcam` user, add to `video` group, `pipx install 'aprilcam[pi]'`,
  copy + `systemctl enable --now aprilcam-web.service`.
- `README.md` — one-page runbook: flash, install, check
  `journalctl -u aprilcam-web -f`, confirm `pi.local` resolves from
  the Mac, Mac-side `.mcp.json` snippet pointing at the SSE URL.
- `smoke_test.sh` — from the Mac, curl `/api/list_cameras`,
  `/api/get_frame?scale=4&grayscale=true`, tail 5s of `/api/stream/{id}`.

Wire `APRILCAM_BIND_HOST` / `APRILCAM_BIND_PORT` into
[src/aprilcam/cli/web_cli.py](src/aprilcam/cli/web_cli.py) as
argparse defaults pulled from `os.environ`.

### Ticket 6 — Pi regression + version bump

- Run `tests/system/test_video_pipeline.py` on the Pi post-install
  (VideoCamera path — no hardware needed) to prove the ARM64 install
  works end-to-end.
- New `tests/integration/test_camera_properties.py` — opens a real
  camera, enumerates properties, sets exposure to two values, asserts
  `supported` round-trip + frame changes. `pytest.mark.skipif` if no
  camera.
- Bump `pyproject.toml` version to `0.20260419.1` (today, first build).

## Deferred to Sprint 016

- Camera-settings persistence (write last-applied settings to
  `~/.config/aprilcam/camera_defaults.json`, auto-apply on `open()`).
- `APRILCAM_TOKEN` middleware (if the Pi ever leaves the trusted LAN).
- MJPEG multi-client shared-encode optimization (single encode,
  fan-out to N clients).
- Optional Dockerfile for reproducible deploys.
- V4L2 property probe test suite against real hardware.

## Critical files

- [src/aprilcam/camera/camera.py](src/aprilcam/camera/camera.py) — property whitelist + set/get methods
- [src/aprilcam/camera/csi_camera.py](src/aprilcam/camera/csi_camera.py) (new) — picamera2 wrapper
- [src/aprilcam/camera/camutil.py](src/aprilcam/camera/camutil.py) — extend `list_cameras()` for libcamera
- [src/aprilcam/server/mcp_server.py](src/aprilcam/server/mcp_server.py) — new tools, `format_image_output` extension, `render_annotated` extract
- [src/aprilcam/server/web_server.py](src/aprilcam/server/web_server.py) — REST mirrors, MJPEG endpoint, HTML UI rewire
- [src/aprilcam/cli/web_cli.py](src/aprilcam/cli/web_cli.py) — env-var defaults
- [src/aprilcam/core/detection.py](src/aprilcam/core/detection.py) — no change, `last_frame` already public
- `deploy/pi/` (new) — systemd unit, install.sh, README, smoke_test.sh
- `tests/integration/test_camera_properties.py` (new)
- `pyproject.toml` — `[pi]` extra + version bump

## Verification

1. **Mac-side sanity first** (before touching the Pi):
   - `uv run pytest` green, including the new property test (skipped
     without hardware).
   - `aprilcam mcp` still starts and serves existing tools (regression).
   - `aprilcam web` locally: hit `/api/get_frame?scale=4&grayscale=true`
     and confirm ~15 KB response; open `/api/stream/{id}` in a browser
     tab and see live overlays.

2. **Pi install:**
   - Flash fresh Pi OS, `deploy/pi/install.sh`, reboot.
   - `systemctl status aprilcam-web` → active.
   - `journalctl -u aprilcam-web -f` shows camera enumeration on boot.
   - `curl http://pi.local:17439/api/list_cameras` lists both USB and
     CSI devices.

3. **Mac-to-Pi end-to-end:**
   - Add SSE MCP entry to Mac's `.mcp.json` pointing at
     `http://pi.local:17439/mcp/sse`. Call `list_cameras`,
     `open_camera`, `start_detection`, `get_tags` — tag stream works.
   - `set_camera_settings(camera_id, exposure=500, auto_exposure=False)`
     → visible change in `/api/stream/{id}` in browser.
   - Run `tests/system/test_video_pipeline.py` on the Pi.
   - Execute `deploy/pi/smoke_test.sh` from the Mac.

4. **Bandwidth spot-check:** time and size three `/api/get_frame`
   responses: default, scale=4, scale=4 grayscale. Confirm the scale=4
   grayscale payload is under 25 KB.
