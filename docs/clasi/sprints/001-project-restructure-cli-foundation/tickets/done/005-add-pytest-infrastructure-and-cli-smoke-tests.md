---
id: "001-005"
title: "Add pytest infrastructure and CLI smoke tests"
status: todo
use-cases: [SUC-001, SUC-002, SUC-003]
depends-on: [001-004]
github-issue: ""
todo: ""
---

# Add pytest infrastructure and CLI smoke tests

## Description

Create the pytest test infrastructure and write smoke tests that verify
the CLI entry point and all subcommands work correctly. This is the
safety net for all future refactoring.

## Acceptance Criteria

- [ ] `tests/__init__.py` exists (empty file, makes tests a package)
- [ ] `tests/conftest.py` exists with shared fixtures (e.g., a `cli_runner` fixture that invokes `main()` with captured output)
- [ ] `tests/test_cli_smoke.py` exists with parametrized smoke tests
- [ ] Smoke test verifies `aprilcam --help` exits 0 and contains "usage"
- [ ] Smoke test verifies `aprilcam <subcommand> --help` exits 0 for all 9 subcommands: mcp, taggen, arucogen, cameras, homocal, screencap, detect, capture, test
- [ ] Smoke test verifies `aprilcam mcp` prints the placeholder message and exits 0
- [ ] Import test verifies `import aprilcam` succeeds without pygame installed
- [ ] Import test verifies `from aprilcam.cli import main` is importable
- [ ] `pytest` (or `uv run pytest`) runs from repo root and all tests pass
- [ ] `pyproject.toml` has `[tool.pytest.ini_options]` with `testpaths = ["tests"]` (if not already present)

## Implementation Notes

Suggested `conftest.py` fixture:

```python
import pytest
from aprilcam.cli import main

@pytest.fixture
def cli_runner():
    """Run the CLI main() with given argv and return exit code."""
    def _run(argv):
        try:
            return main(argv)
        except SystemExit as e:
            return e.code
    return _run
```

Suggested `test_cli_smoke.py` structure:

```python
import pytest

SUBCOMMANDS = ["mcp", "taggen", "arucogen", "cameras", "homocal",
               "screencap", "detect", "capture", "test"]

@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_subcommand_help(cli_runner, subcmd):
    """Each subcommand's --help should exit 0."""
    code = cli_runner([subcmd, "--help"])
    assert code == 0

def test_main_help(cli_runner):
    code = cli_runner(["--help"])
    assert code == 0

def test_import_aprilcam():
    import aprilcam  # should not raise
```

The tests should NOT require a camera, display, or pygame.

## Testing

- **Existing tests to run**: This IS the test infrastructure.
- **New tests to write**: All tests described in acceptance criteria above.
- **Verification command**: `uv run pytest -v`
