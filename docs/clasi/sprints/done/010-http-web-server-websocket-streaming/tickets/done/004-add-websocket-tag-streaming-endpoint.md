---
id: '004'
title: Add WebSocket tag streaming endpoint
status: done
use-cases:
- SUC-010-004
depends-on:
- '002'
github-issue: ''
todo: ''
---

# Add WebSocket tag streaming endpoint

## Description

Add a WebSocket route at `/ws/tags/{source_id}` to the Starlette app.
When a client connects:

1. Server checks that a detection loop is running on `source_id`.
2. Server starts an asyncio task that polls the detection ring buffer
   at ~30fps.
3. New tag frames are pushed to the client as JSON messages (same
   format as `get_tags` response).
4. Backpressure: drop old frames, always send latest.
5. Server handles client disconnection gracefully.
6. Multiple clients can subscribe to the same source_id.

Add `websockets` to pyproject.toml dependencies.

## Acceptance Criteria

- [ ] WebSocket endpoint at `/ws/tags/{source_id}` accepts connections
- [ ] JSON messages match `get_tags` response format
- [ ] Messages stream continuously while detection is running
- [ ] Slow clients get latest frame (old frames dropped)
- [ ] Graceful handling of client disconnect
- [ ] Multiple simultaneous WebSocket clients supported

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: WebSocket integration test with mock detection data
- **Verification command**: `uv run pytest`
