---
timestamp: '2026-03-24T04:51:12'
parent: team-lead
child: technical-lead
scope: /Volumes/Proj/proj/RobotProjects/AprilTags/docs/clasi/sprints/006-multi-camera-compositing
sprint: 006-multi-camera-compositing
context_documents:
- docs/clasi/sprints/006-multi-camera-compositing/sprint.md
- docs/clasi/sprints/006-multi-camera-compositing/architecture-update.md
- docs/clasi/sprints/006-multi-camera-compositing/usecases.md
result: "success \u2014 4 tickets created"
files_modified: []
---

# Dispatch: team-lead → technical-lead

Create tickets for Sprint 006: Multi-Camera Compositing. Use create_ticket with sprint_id="006".

### Ticket 001: Composite class and cross-camera homography
- Create `src/aprilcam/composite.py` with `Composite` dataclass and `CompositeManager`
- `compute_cross_camera_homography(primary_points, secondary_points)` function
- `map_tags_to_primary(tags, homography)` function
- depends-on: []
- Use cases: SUC-001, SUC-002, SUC-004

### Ticket 002: create_composite MCP tool
- Auto-detect ArUco markers in both cameras for homography
- Manual correspondence_points fallback
- Register composite in CompositeManager
- depends-on: ["001"]
- Use cases: SUC-001, SUC-002

### Ticket 003: get_composite_frame and get_composite_tags MCP tools
- `get_composite_frame(composite_id, format?)` — primary frame with tag overlays from secondary
- `get_composite_tags(composite_id)` — tags in primary camera coordinates
- `render_tag_overlay(frame, mapped_tags)` function
- depends-on: ["002"]
- Use cases: SUC-003, SUC-004

### Ticket 004: Integration tests
- End-to-end: create_composite -> get_composite_tags -> get_composite_frame
- Tests with synthetic image pairs (ArUco markers at known positions)
- Verify individual cameras still work independently
- depends-on: ["003"]
- Use cases: SUC-001 through SUC-005

Write substantive acceptance criteria for each.

## Context Documents

- `docs/clasi/sprints/006-multi-camera-compositing/sprint.md`
- `docs/clasi/sprints/006-multi-camera-compositing/architecture-update.md`
- `docs/clasi/sprints/006-multi-camera-compositing/usecases.md`
