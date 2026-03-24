"""Tests for aprilcam init subcommand."""

import json

from aprilcam.cli.init_cli import main, CLAUDE_CODE_ENTRY, VSCODE_ENTRY


def test_init_creates_mcp_json(tmp_path, monkeypatch):
    """Fresh directory gets a .mcp.json with the aprilcam entry."""
    monkeypatch.chdir(tmp_path)
    main()
    data = json.loads((tmp_path / ".mcp.json").read_text())
    assert data["mcpServers"]["aprilcam"] == CLAUDE_CODE_ENTRY


def test_init_creates_vscode_mcp_json(tmp_path, monkeypatch):
    """Fresh directory gets .vscode/mcp.json with the aprilcam entry."""
    monkeypatch.chdir(tmp_path)
    main()
    path = tmp_path / ".vscode" / "mcp.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["servers"]["aprilcam"] == VSCODE_ENTRY


def test_init_preserves_existing_entries(tmp_path, monkeypatch):
    """Pre-existing server entries in both files survive the upsert."""
    monkeypatch.chdir(tmp_path)

    # Seed .mcp.json with another server
    mcp = tmp_path / ".mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"other": {"command": "other-server"}}}) + "\n")

    # Seed .vscode/mcp.json with another server
    vscode_dir = tmp_path / ".vscode"
    vscode_dir.mkdir()
    vscode_mcp = vscode_dir / "mcp.json"
    vscode_mcp.write_text(json.dumps({"servers": {"other": {"command": "other-server"}}}) + "\n")

    main()

    data_cc = json.loads(mcp.read_text())
    assert data_cc["mcpServers"]["other"] == {"command": "other-server"}
    assert data_cc["mcpServers"]["aprilcam"] == CLAUDE_CODE_ENTRY

    data_vs = json.loads(vscode_mcp.read_text())
    assert data_vs["servers"]["other"] == {"command": "other-server"}
    assert data_vs["servers"]["aprilcam"] == VSCODE_ENTRY


def test_init_updates_existing_aprilcam(tmp_path, monkeypatch):
    """An outdated aprilcam entry gets replaced."""
    monkeypatch.chdir(tmp_path)

    mcp = tmp_path / ".mcp.json"
    old_entry = {"command": "old-aprilcam", "args": ["--legacy"]}
    mcp.write_text(json.dumps({"mcpServers": {"aprilcam": old_entry}}) + "\n")

    main()

    data = json.loads(mcp.read_text())
    assert data["mcpServers"]["aprilcam"] == CLAUDE_CODE_ENTRY


def test_init_handles_empty_file(tmp_path, monkeypatch):
    """An empty .mcp.json is treated as a fresh file."""
    monkeypatch.chdir(tmp_path)

    mcp = tmp_path / ".mcp.json"
    mcp.write_text("")

    main()

    data = json.loads(mcp.read_text())
    assert data["mcpServers"]["aprilcam"] == CLAUDE_CODE_ENTRY


def test_init_handles_invalid_json(tmp_path, monkeypatch):
    """Invalid JSON in .mcp.json is overwritten with a warning."""
    monkeypatch.chdir(tmp_path)

    mcp = tmp_path / ".mcp.json"
    mcp.write_text("{not valid json!!!")

    main()

    data = json.loads(mcp.read_text())
    assert data["mcpServers"]["aprilcam"] == CLAUDE_CODE_ENTRY
