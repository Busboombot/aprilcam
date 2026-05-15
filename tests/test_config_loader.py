"""Tests for the Config dataclass and multi-source loader in config.py (T001)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from aprilcam.config import Config, _find_dotfile, _parse_dotfile


# ---------------------------------------------------------------------------
# _find_dotfile helper
# ---------------------------------------------------------------------------


def test_find_dotfile_finds_file_in_start_dir(tmp_path):
    target = tmp_path / ".aprilcam"
    target.write_text("KEY=value\n")
    found = _find_dotfile(".aprilcam", tmp_path)
    assert found == target


def test_find_dotfile_finds_file_in_parent_dir(tmp_path):
    target = tmp_path / ".aprilcam"
    target.write_text("KEY=value\n")
    subdir = tmp_path / "sub" / "dir"
    subdir.mkdir(parents=True)
    found = _find_dotfile(".aprilcam", subdir)
    assert found == target


def test_find_dotfile_returns_none_when_absent(tmp_path):
    found = _find_dotfile(".aprilcam", tmp_path)
    assert found is None


# ---------------------------------------------------------------------------
# _parse_dotfile helper
# ---------------------------------------------------------------------------


def test_parse_dotfile_basic(tmp_path):
    f = tmp_path / ".aprilcam"
    f.write_text("APRILCAM_LOG_LEVEL=DEBUG\nAPRILCAM_DATA_DIR=/data\n")
    result = _parse_dotfile(f)
    assert result == {"APRILCAM_LOG_LEVEL": "DEBUG", "APRILCAM_DATA_DIR": "/data"}


def test_parse_dotfile_strips_comments(tmp_path):
    f = tmp_path / ".aprilcam"
    f.write_text("# full-line comment\nAPRILCAM_LOG_LEVEL=INFO  # inline comment\n")
    result = _parse_dotfile(f)
    assert result == {"APRILCAM_LOG_LEVEL": "INFO"}


def test_parse_dotfile_skips_blank_lines(tmp_path):
    f = tmp_path / ".aprilcam"
    f.write_text("\n\nAPRILCAM_LOG_LEVEL=WARN\n\n")
    result = _parse_dotfile(f)
    assert result == {"APRILCAM_LOG_LEVEL": "WARN"}


def test_parse_dotfile_missing_file_returns_empty():
    result = _parse_dotfile(Path("/nonexistent/path/.aprilcam"))
    assert result == {}


# ---------------------------------------------------------------------------
# Config.load() — default values
# ---------------------------------------------------------------------------


def test_config_load_defaults(tmp_path, monkeypatch):
    """With no config files and no APRILCAM_ env vars, all defaults apply."""
    monkeypatch.chdir(tmp_path)
    # Remove any APRILCAM_ vars from the environment
    for key in list(os.environ.keys()):
        if key.startswith("APRILCAM_"):
            monkeypatch.delenv(key, raising=False)

    cfg = Config.load(start=tmp_path)

    assert cfg.data_dir == (tmp_path / "data/runtime").resolve()
    assert cfg.socket_dir == Path("/tmp/aprilcam/")
    assert cfg.calibration_source == (tmp_path / "data/calibration.json").resolve()
    assert cfg.calibration_save_path == (tmp_path / "data/calibration.json").resolve()
    assert cfg.log_level == "INFO"
    assert cfg.daemon_pidfile == Path("/tmp/aprilcam/aprilcamd.pid")


def test_config_load_does_not_raise_with_no_files(tmp_path, monkeypatch):
    """Config.load() must not raise even when no config files exist."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ.keys()):
        if key.startswith("APRILCAM_"):
            monkeypatch.delenv(key, raising=False)
    # Should complete without raising
    Config.load(start=tmp_path)


# ---------------------------------------------------------------------------
# Config.load() — env var overrides
# ---------------------------------------------------------------------------


def test_env_var_overrides_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APRILCAM_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("APRILCAM_DATA_DIR", "/custom/data")

    cfg = Config.load(start=tmp_path)

    assert cfg.log_level == "DEBUG"
    assert cfg.data_dir == Path("/custom/data")


def test_env_var_overrides_dotfile(tmp_path, monkeypatch):
    """Env vars must win over values in .aprilcam dotfile."""
    dotfile = tmp_path / ".aprilcam"
    dotfile.write_text("APRILCAM_LOG_LEVEL=WARNING\n")
    monkeypatch.setenv("APRILCAM_LOG_LEVEL", "ERROR")

    cfg = Config.load(start=tmp_path)

    assert cfg.log_level == "ERROR"


# ---------------------------------------------------------------------------
# Config.load() — dotfile overrides
# ---------------------------------------------------------------------------


def test_project_dotfile_overrides_user_dotfile(tmp_path, monkeypatch):
    """Project-local .aprilcam must win over ~/.aprilcam."""
    # Remove env interference
    monkeypatch.delenv("APRILCAM_LOG_LEVEL", raising=False)

    # Simulate ~/.aprilcam by patching Path.home()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".aprilcam").write_text("APRILCAM_LOG_LEVEL=WARNING\n")

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".aprilcam").write_text("APRILCAM_LOG_LEVEL=DEBUG\n")

    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    cfg = Config.load(start=project_dir)
    assert cfg.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Config.load() — socket_dir creation
# ---------------------------------------------------------------------------


def test_socket_dir_created_if_missing(tmp_path, monkeypatch):
    socket_dir = tmp_path / "sockets" / "aprilcam"
    assert not socket_dir.exists()

    monkeypatch.setenv("APRILCAM_SOCKET_DIR", str(socket_dir))

    Config.load(start=tmp_path)

    assert socket_dir.exists()
    assert socket_dir.is_dir()


def test_socket_dir_creation_idempotent(tmp_path, monkeypatch):
    """Config.load() must not raise if socket_dir already exists."""
    socket_dir = tmp_path / "sockets"
    socket_dir.mkdir()
    monkeypatch.setenv("APRILCAM_SOCKET_DIR", str(socket_dir))

    Config.load(start=tmp_path)  # should not raise


# ---------------------------------------------------------------------------
# AppConfig smoke test — existing class must still work
# ---------------------------------------------------------------------------


def test_appconfig_unchanged(tmp_path):
    """AppConfig.find_env raises FileNotFoundError when no .env exists."""
    from aprilcam.config import AppConfig

    with pytest.raises(FileNotFoundError):
        AppConfig.find_env(start=tmp_path)
