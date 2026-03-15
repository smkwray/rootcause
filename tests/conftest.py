"""Shared pytest fixtures for the povcrime test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from povcrime.config import get_config


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory managed by pytest."""
    return tmp_path


@pytest.fixture()
def config():
    """Return the default project configuration."""
    return get_config()
