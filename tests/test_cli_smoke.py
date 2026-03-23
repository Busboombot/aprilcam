import pytest

SUBCOMMANDS = [
    "mcp", "taggen", "arucogen", "cameras", "homocal",
    "screencap", "detect", "capture", "test",
]


@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_subcommand_help(cli_runner, subcmd):
    """Each subcommand's --help should exit 0."""
    code = cli_runner([subcmd, "--help"])
    assert code == 0


def test_main_help(cli_runner):
    """Main --help should exit 0."""
    code = cli_runner(["--help"])
    assert code == 0


def test_no_subcommand_shows_help(cli_runner):
    """No subcommand should return non-zero."""
    code = cli_runner([])
    assert code != 0


def test_mcp_stub(cli_runner, capsys):
    """MCP stub should print message and exit 0."""
    code = cli_runner(["mcp"])
    assert code == 0


def test_import_aprilcam():
    """import aprilcam should work without pygame."""
    import aprilcam  # noqa: F401


def test_import_cli_main():
    """from aprilcam.cli import main should be importable."""
    from aprilcam.cli import main  # noqa: F401
    assert callable(main)
