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
