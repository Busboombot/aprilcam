---
id: '007'
title: Refactor existing MCP tools as backward-compat wrappers
status: done
use-cases:
- SUC-002
depends-on:
- '004'
- '005'
github-issue: ''
todo: ''
---
<!-- CLASI: Before changing code or making plans, review the SE process in CLAUDE.md -->

# Refactor existing MCP tools as backward-compat wrappers

## Description

Refactor the existing per-operation MCP tools (`detect_lines`, `detect_circles`,
`detect_contours`, `detect_qr_codes`, `deskew_image`, etc.) to become thin
wrappers around the new frame pipeline. This ensures backward compatibility --
no external API changes -- while eliminating duplicated processing logic.

### Pattern for each wrapper

Each existing tool follows the same pattern internally:

1. Create a transient FrameEntry (capture from source or load from file)
2. Run the single relevant operation via the batch pipeline
3. Extract the results from `frame.results[operation_name]`
4. Return the results in the same format as before
5. The transient frame can be left in the ring buffer (auto-evicts) or
   released immediately

### Tools to refactor

- `detect_lines(source_id, ...)` -- wraps `["detect_lines"]`
- `detect_circles(source_id, ...)` -- wraps `["detect_circles"]`
- `detect_contours(source_id, ...)` -- wraps `["detect_contours"]`
- `detect_qr_codes(source_id, ...)` -- wraps `["detect_qr"]`
- `deskew_image(source_id, ...)` -- wraps `["deskew"]`
- Any other per-operation tools that currently do capture + process + return

### External API unchanged

The MCP tool signatures, parameter names, and response formats must remain
exactly the same. Agents using these tools should see no difference.

## Acceptance Criteria

- [ ] `detect_lines` internally uses create_frame + process_frame pipeline
- [ ] `detect_circles` internally uses create_frame + process_frame pipeline
- [ ] `detect_contours` internally uses create_frame + process_frame pipeline
- [ ] `detect_qr_codes` internally uses create_frame + process_frame pipeline
- [ ] `deskew_image` internally uses create_frame + process_frame pipeline
- [ ] All existing tool signatures unchanged (same parameters, same return format)
- [ ] Response format identical to pre-refactor output
- [ ] No duplicated processing logic -- all tools go through batch pipeline
- [ ] Transient frames are handled appropriately (auto-evict or release)

## Implementation Notes

### Key files
- `src/aprilcam/mcp_server.py` -- all existing MCP tool functions to refactor

### Design decisions
- Wrapper functions remain as the MCP-registered tools (same `@mcp.tool()` decorators)
- Internal implementation changes from direct OpenCV calls to pipeline calls
- Transient frames left in ring buffer (they'll auto-evict; explicit release
  would add complexity for no benefit since the ring buffer is bounded)
- The `resolve_source()` helper is still used to validate source_id before
  passing to `create_frame`

### Approach
- Refactor one tool at a time, verifying response format after each
- Use the existing test suite as a regression check
- Compare response dicts before/after to ensure format parity

## Testing

- **Existing tests to run**: `uv run pytest` (full suite -- this is the primary
  regression check; existing tool tests validate backward compatibility)
- **New tests to write**:
  - `test_detect_lines_uses_pipeline` -- verify internal pipeline path (mock check)
  - `test_wrapper_response_format_parity` -- verify output matches pre-refactor format
- **Verification command**: `uv run pytest`
