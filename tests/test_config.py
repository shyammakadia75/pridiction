"""Tests for centralized configuration: defaults, overrides, and validation."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from football_intelligence import config as cfg
from football_intelligence.config import Settings, load_settings
from football_intelligence.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_default_environment_is_development():
    settings = load_settings()
    assert settings.environment == "development"


def test_default_log_level_is_info():
    settings = load_settings()
    assert settings.log_level == "INFO"


def test_default_database_path_is_under_data_dir():
    settings = load_settings()
    assert settings.database_path == cfg.DATA_DIR / "football_intelligence.db"


def test_default_database_path_parent_is_data_dir():
    settings = load_settings()
    assert settings.database_path.parent == cfg.DATA_DIR


def test_module_settings_singleton_is_consistent():
    # The module-level singleton should match a freshly loaded default.
    assert cfg.settings.environment == load_settings().environment
    assert cfg.settings.database_path == load_settings().database_path


def test_project_paths_are_project_relative():
    # No absolute machine-specific paths should be baked in.
    assert cfg.PROJECT_ROOT.is_absolute()
    assert (cfg.DATA_DIR == cfg.PROJECT_ROOT / "data")
    assert (cfg.RAW_DATA_DIR == cfg.DATA_DIR / "raw")
    assert (cfg.LOGS_DIR == cfg.PROJECT_ROOT / "logs")
    assert (cfg.SRC_DIR == cfg.PROJECT_ROOT / "src")


# ---------------------------------------------------------------------------
# Environment-variable overrides
# ---------------------------------------------------------------------------
def test_env_override_log_level(monkeypatch):
    monkeypatch.setenv("FI_LOG_LEVEL", "DEBUG")
    settings = load_settings()
    assert settings.log_level == "DEBUG"


def test_env_override_environment(monkeypatch):
    monkeypatch.setenv("FI_ENVIRONMENT", "testing")
    settings = load_settings()
    assert settings.environment == "testing"


def test_env_override_database_path_relative(monkeypatch, tmp_path):
    monkeypatch.setenv("FI_DATABASE_PATH", "custom.db")
    settings = load_settings()
    # Relative paths resolve against the project root.
    assert settings.database_path == cfg.PROJECT_ROOT / "custom.db"
    assert settings.database_path.is_absolute()


def test_env_override_database_path_absolute(monkeypatch, tmp_path):
    target = tmp_path / "abs.db"
    monkeypatch.setenv("FI_DATABASE_PATH", str(target))
    settings = load_settings()
    assert settings.database_path == target


def test_env_override_log_file_and_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FI_LOG_FILE", "custom.log")
    monkeypatch.setenv("FI_LOG_DIR", str(tmp_path))
    settings = load_settings()
    assert settings.log_file == "custom.log"
    assert settings.log_dir == tmp_path


def test_real_env_var_not_overridden_by_dotenv(monkeypatch, tmp_path):
    # When a real env var is set, load_dotenv(override=False) must not clobber it.
    monkeypatch.setenv("FI_LOG_LEVEL", "ERROR")
    # Re-import of load_dotenv_file is a no-op for existing vars.
    cfg.load_dotenv_file()
    settings = load_settings()
    assert settings.log_level == "ERROR"


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_level", ["VERBOSE", "silent", "10", ""])
def test_invalid_log_level_raises(monkeypatch, bad_level):
    monkeypatch.setenv("FI_LOG_LEVEL", bad_level)
    with pytest.raises(ConfigError):
        load_settings()


@pytest.mark.parametrize("bad_env", ["staging", "prod", "", "live"])
def test_invalid_environment_raises(monkeypatch, bad_env):
    monkeypatch.setenv("FI_ENVIRONMENT", bad_env)
    with pytest.raises(ConfigError):
        load_settings()


def test_settings_is_frozen():
    settings = load_settings()
    with pytest.raises(Exception):
        settings.environment = "testing"  # frozen dataclass


def test_database_parent_that_is_a_file_raises(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("FI_DATABASE_PATH", str(blocker / "x.db"))
    with pytest.raises(ConfigError):
        load_settings()
