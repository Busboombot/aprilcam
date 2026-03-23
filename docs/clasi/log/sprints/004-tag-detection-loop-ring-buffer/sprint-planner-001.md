---
timestamp: '2026-03-23T20:48:28'
parent: team-lead
child: sprint-planner
scope: docs/clasi/sprints/004-tag-detection-loop-ring-buffer/
sprint: 004-tag-detection-loop-ring-buffer
context_documents:
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/sprint.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/architecture-update.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/usecases.md
result: success - sprint.md (197 lines), usecases.md (150 lines), architecture-update.md
  (143 lines) written
files_modified:
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/sprint.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/usecases.md
- docs/clasi/sprints/004-tag-detection-loop-ring-buffer/architecture-update.md
---

# Dispatch: team-lead → sprint-planner

Write the planning documents for Sprint 004: Tag Detection Loop & Ring Buffer. You may only create or modify files under docs/clasi/sprints/004-tag-detection-loop-ring-buffer/. You may read files from any location.

Context: Persistent detection loop controlled by agent (start/stop). 300-frame ring buffer storing per-frame tag records. MCP tools: start_detection, stop_detection, get_tags, get_tag_history. Each record: id, center_px, corners_px, orientation_yaw, world_xy, in_playfield, velocity, speed, heading. Background thread, agents query on demand. Existing code: aprilcam.py (detection engine), models.py (AprilTag, AprilTagFlow).

Files to write: sprint.md, usecases.md, architecture-update.md.

## Context Documents

- `docs/clasi/sprints/004-tag-detection-loop-ring-buffer/sprint.md`
- `docs/clasi/sprints/004-tag-detection-loop-ring-buffer/architecture-update.md`
- `docs/clasi/sprints/004-tag-detection-loop-ring-buffer/usecases.md`
