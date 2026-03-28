---
id: "005"
title: "Public API exports and documentation"
status: done
use-cases: [SUC-011-004]
depends-on: ["001", "002", "003", "004"]
github-issue: ""
todo: ""
---

# Public API exports and documentation

## Description

Export key classes from `__init__.py` and update agent-facing docs.

### Changes

1. **`__init__.py`**: Add exports:
   ```python
   from aprilcam.aprilcam import AprilCam
   from aprilcam.stream import detect_tags
   from aprilcam.detection import TagRecord, DetectionLoop
   from aprilcam.models import AprilTag
   from aprilcam.playfield import Playfield
   from aprilcam.errors import (
       CameraError, CameraInUseError,
       CameraNotFoundError, CameraPermissionError,
   )
   ```

2. **Agent documentation**: Update or create a doc describing:
   - The `data/` directory convention for homography files
   - The `detect_tags()` generator as recommended API
   - Camera contention handling
   - Available public imports

## Acceptance Criteria

- [ ] `from aprilcam import detect_tags, AprilCam, TagRecord` works
- [ ] `from aprilcam import CameraInUseError` works
- [ ] Agent documentation covers the library API
- [ ] No circular import issues

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: Import smoke tests verifying all exports
- **Verification command**: `uv run pytest`
