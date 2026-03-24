---
id: '005'
title: detect_qr_codes and apply_transform MCP tools
status: done
use-cases:
- SUC-006
- SUC-008
depends-on:
- '001'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# detect_qr_codes and apply_transform MCP tools

## Description

Add QR code detection and image transformation to `image_processing.py`,
plus their MCP tool registrations:

1. **`process_detect_qr_codes(frame)`** — Uses `cv2.QRCodeDetector`
   to detect and decode QR codes in the frame. Returns a list of
   dicts with `data` (decoded string), `points` (corner coordinates),
   and `bounding_box`.

2. **`process_apply_transform(frame, operation, params)`** — Applies
   a named image transformation. Supported operations:
   - `rotate` — rotate by `angle` degrees (params: `{"angle": 90}`)
   - `scale` — resize by factor (params: `{"factor": 0.5}`)
   - `threshold` — binary threshold (params: `{"value": 127}`)
   - `canny` — Canny edge detection (params: `{"low": 50, "high": 150}`)
   - `blur` — Gaussian blur (params: `{"ksize": 5}`)
   Returns the transformed frame.

3. **MCP tools `detect_qr_codes` and `apply_transform`** —
   `detect_qr_codes` returns structured JSON. `apply_transform`
   returns the transformed image via `format_image_output`.

## Acceptance Criteria

- [ ] `process_detect_qr_codes()` added to `image_processing.py`
- [ ] `process_detect_qr_codes()` returns list of `{"data", "points", "bounding_box"}` dicts
- [ ] `process_detect_qr_codes()` returns empty list when no QR codes found
- [ ] `process_apply_transform()` added to `image_processing.py`
- [ ] `rotate` operation rotates frame by specified angle
- [ ] `scale` operation resizes frame by specified factor
- [ ] `threshold` operation applies binary threshold
- [ ] `canny` operation applies Canny edge detection
- [ ] `blur` operation applies Gaussian blur
- [ ] `process_apply_transform()` raises error for unknown operation
- [ ] `detect_qr_codes` MCP tool registered with `source_id`
- [ ] `apply_transform` MCP tool registered with `source_id`, `operation`, `params`, and optional `format`
- [ ] All existing tests continue to pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**:
  - Unit tests in `tests/test_image_processing.py`:
    - `test_detect_qr_synthetic` — generate QR code image, verify detection and decode
    - `test_detect_qr_empty` — blank image, verify empty list
    - `test_transform_rotate` — verify frame dimensions change for 90-degree rotation
    - `test_transform_scale` — verify output dimensions match scale factor
    - `test_transform_threshold` — verify output is binary (only 0 and 255 values)
    - `test_transform_canny` — verify edge output is single-channel
    - `test_transform_blur` — verify output is smoother than input
    - `test_transform_unknown` — unknown operation raises error
- **Verification command**: `uv run pytest`
