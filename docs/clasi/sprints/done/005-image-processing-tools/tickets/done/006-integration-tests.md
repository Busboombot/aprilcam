---
id: '006'
title: Integration tests
status: done
use-cases:
- SUC-001
- SUC-002
- SUC-003
- SUC-004
- SUC-005
- SUC-006
- SUC-007
- SUC-008
depends-on:
- '002'
- '003'
- '004'
- '005'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Integration tests

## Description

Write full MCP tool round-trip integration tests for all 8 image
processing tools. Each test creates a mock camera with a synthetic
test image, registers it with the server, calls the MCP tool through
the server's tool dispatch, and verifies the response structure and
content.

These tests validate the complete pipeline: MCP tool entry point ->
resolve_source -> processing function -> format_image_output -> MCP
response. They complement the unit tests in tickets 001-005 by
exercising the tools as an MCP client would use them.

## Acceptance Criteria

- [ ] `tests/test_mcp_image_tools.py` test file created
- [ ] Integration test for `get_frame` — verifies full round-trip with mock camera
- [ ] Integration test for `crop_region` — verifies cropped output dimensions and format
- [ ] Integration test for `detect_lines` — verifies structured JSON response with line data
- [ ] Integration test for `detect_circles` — verifies structured JSON response with circle data
- [ ] Integration test for `detect_contours` — verifies contour list structure
- [ ] Integration test for `detect_motion` — verifies two-call sequence (empty then regions)
- [ ] Integration test for `detect_qr_codes` — verifies QR detection and decode
- [ ] Integration test for `apply_transform` — verifies at least 2 operations (e.g., rotate, canny)
- [ ] All tests use a shared fixture for mock camera setup
- [ ] Tests verify both success cases and error cases (invalid source_id)
- [ ] All existing tests continue to pass

## Testing

- **Existing tests to run**: `uv run pytest tests/`
- **New tests to write**: All tests in `tests/test_mcp_image_tools.py` as described above
- **Verification command**: `uv run pytest`
