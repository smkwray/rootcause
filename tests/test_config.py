"""Tests for povcrime.config module."""

from __future__ import annotations

import dataclasses

from povcrime.config import get_config


def test_get_config_returns_dataclass():
    """get_config() should return a dataclass instance."""
    config = get_config()
    assert dataclasses.is_dataclass(config)
    assert not isinstance(config, type)  # instance, not the class itself


def test_config_has_required_fields():
    """Config must expose project_root, data_dir, start_year, end_year."""
    config = get_config()
    assert hasattr(config, "project_root")
    assert hasattr(config, "data_dir")
    assert hasattr(config, "start_year")
    assert hasattr(config, "end_year")


def test_config_defaults():
    """Default config should span 2000-2024."""
    config = get_config()
    assert config.start_year == 2000
    assert config.end_year == 2024
