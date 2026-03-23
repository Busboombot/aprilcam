---
id: "001-001"
title: "Update pyproject.toml: single entry point and optional dependencies"
status: todo
use-cases: [SUC-001]
depends-on: []
github-issue: ""
todo: ""
---

# Update pyproject.toml: single entry point and optional dependencies

## Description

Replace the 9 separate `[project.scripts]` entries with a single entry
point: `aprilcam = "aprilcam.cli:main"`. Move `pygame` from core
dependencies to `[project.optional-dependencies]` under a `playfield`
extra. Add an `mcp` optional dependency group with `mcp>=1.0`.

This is a pure configuration change to `pyproject.toml`. The
`aprilcam.cli:main` function created in ticket 001-002 must land in the
same commit or immediately after for the package to remain installable.

## Acceptance Criteria

- [ ] `[project.scripts]` has exactly one entry: `aprilcam = "aprilcam.cli:main"`
- [ ] `pygame>=2.5` is removed from `[project.dependencies]`
- [ ] `[project.optional-dependencies]` section exists with `playfield = ["pygame>=2.5"]`
- [ ] `[project.optional-dependencies]` section includes `mcp = ["mcp>=1.0"]`
- [ ] All other core dependencies (`opencv-contrib-python`, `numpy`, `python-dotenv`, `mss`) remain unchanged
- [ ] No other files are modified in this ticket

## Implementation Notes

Replace the existing `[project.scripts]` block (9 entries) with:

```toml
[project.scripts]
aprilcam = "aprilcam.cli:main"
```

Remove `"pygame>=2.5"` from the `dependencies` list and add:

```toml
[project.optional-dependencies]
playfield = ["pygame>=2.5"]
mcp = ["mcp>=1.0"]
```

## Testing

- **Existing tests to run**: None (no tests exist yet).
- **New tests to write**: None for this ticket.
- **Verification command**: `python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb')))"` to validate TOML syntax.
