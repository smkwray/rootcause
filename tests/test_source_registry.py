"""Tests for povcrime.source_registry module."""

from __future__ import annotations

import json
from pathlib import Path

from povcrime.source_registry import export_manifest, get_sources


def test_get_sources_returns_list():
    """get_sources() should return a list."""
    sources = get_sources()
    assert isinstance(sources, list)


def test_sources_have_required_fields():
    """Every source must have name, url, and description attributes."""
    sources = get_sources()
    for src in sources:
        assert hasattr(src, "name"), f"Source missing 'name': {src}"
        assert hasattr(src, "url"), f"Source missing 'url': {src}"
        assert hasattr(src, "description"), f"Source missing 'description': {src}"


def test_minimum_source_count():
    """The registry should contain at least 7 data sources."""
    sources = get_sources()
    assert len(sources) >= 7, f"Expected >= 7 sources, got {len(sources)}"


def test_export_manifest(tmp_path: Path):
    """export_manifest() should write valid JSON to the given path."""
    manifest_path = tmp_path / "manifest.json"
    export_manifest(str(manifest_path))

    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert isinstance(data, (list, dict))
