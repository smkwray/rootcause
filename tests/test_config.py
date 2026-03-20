"""Tests for povcrime.config module."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from povcrime.config import get_config, load_project_config


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
    assert config.panel.unit == "county_fips x year"
    assert config.panel.cluster_var == "state_fips"
    assert config.analysis_lanes
    assert config.bidirectional_lanes
    assert "effective_min_wage" in config.treatments
    assert "violent_crime_rate" in config.outcomes


def test_load_project_config_reads_analysis_lanes(tmp_path: Path):
    """Custom project YAML should populate typed panel and lane metadata."""
    _write_project_yaml(
        tmp_path,
        """
study_name: Test Study
start_year: 2010
end_year: 2012
panel:
  unit: county_fips x year
  cluster_var: state_fips
analysis_lanes:
  - slug: test_lane
    title: Test Lane
    family: minimum_wage
    treatment: effective_min_wage
    outcome: violent_crime_rate
    tier: primary
    event_col: min_wage_event_year
    event_threshold: 0.01
    methods: [baseline, dml]
        """.strip(),
    )

    config = load_project_config(tmp_path)

    assert config.study_name == "Test Study"
    assert config.start_year == 2010
    assert config.end_year == 2012
    assert config.panel.unit == "county_fips x year"
    assert config.analysis_lanes[0].slug == "test_lane"
    assert config.analysis_lanes[0].event_col == "min_wage_event_year"


def test_load_project_config_rejects_duplicate_lane_slugs(tmp_path: Path):
    _write_project_yaml(
        tmp_path,
        """
panel:
  unit: county_fips x year
  cluster_var: state_fips
analysis_lanes:
  - slug: duplicate_lane
    title: First
    family: minimum_wage
    treatment: effective_min_wage
    outcome: violent_crime_rate
    tier: primary
    methods: [baseline]
  - slug: duplicate_lane
    title: Second
    family: minimum_wage
    treatment: effective_min_wage
    outcome: property_crime_rate
    tier: primary
    methods: [baseline]
        """.strip(),
    )

    with pytest.raises(ValueError, match="Duplicate analysis lane slug"):
        load_project_config(tmp_path)


def test_load_project_config_rejects_invalid_tier(tmp_path: Path):
    _write_project_yaml(
        tmp_path,
        """
panel:
  unit: county_fips x year
  cluster_var: state_fips
analysis_lanes:
  - slug: bad_lane
    title: Bad Lane
    family: minimum_wage
    treatment: effective_min_wage
    outcome: violent_crime_rate
    tier: unsupported
    methods: [baseline]
        """.strip(),
    )

    with pytest.raises(ValueError, match="must be one of"):
        load_project_config(tmp_path)


def test_load_project_config_reads_bidirectional_lanes(tmp_path: Path):
    _write_project_yaml(
        tmp_path,
        """
panel:
  unit: county_fips x year
  cluster_var: state_fips
analysis_lanes:
  - slug: policy_lane
    title: Policy Lane
    family: minimum_wage
    treatment: effective_min_wage
    outcome: violent_crime_rate
    tier: primary
    methods: [baseline]
bidirectional_lanes:
  - slug: poverty_to_violent
    title: Poverty -> Violent Crime
    family: bidirectional
    treatment: poverty_rate
    outcome: violent_crime_rate
    tier: exploratory
    methods: [baseline, dml]
        """.strip(),
    )

    config = load_project_config(tmp_path)

    assert config.bidirectional_lanes[0].slug == "poverty_to_violent"
    assert config.bidirectional_lanes[0].treatment == "poverty_rate"


def test_load_project_config_rejects_policy_bidirectional_slug_overlap(tmp_path: Path):
    _write_project_yaml(
        tmp_path,
        """
panel:
  unit: county_fips x year
  cluster_var: state_fips
analysis_lanes:
  - slug: shared_lane
    title: Policy Lane
    family: minimum_wage
    treatment: effective_min_wage
    outcome: violent_crime_rate
    tier: primary
    methods: [baseline]
bidirectional_lanes:
  - slug: shared_lane
    title: Poverty -> Violent Crime
    family: bidirectional
    treatment: poverty_rate
    outcome: violent_crime_rate
    tier: exploratory
    methods: [baseline]
        """.strip(),
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        load_project_config(tmp_path)


def test_load_project_config_requires_panel_metadata(tmp_path: Path):
    _write_project_yaml(
        tmp_path,
        """
analysis_lanes:
  - slug: test_lane
    title: Test Lane
    family: minimum_wage
    treatment: effective_min_wage
    outcome: violent_crime_rate
    tier: primary
    methods: [baseline]
        """.strip(),
    )

    with pytest.raises(ValueError, match="panel must define non-empty string field 'unit'"):
        load_project_config(tmp_path)


def test_load_project_config_requires_event_metadata_for_staggered_lanes(tmp_path: Path):
    _write_project_yaml(
        tmp_path,
        """
panel:
  unit: county_fips x year
  cluster_var: state_fips
analysis_lanes:
  - slug: staggered_lane
    title: Staggered Lane
    family: minimum_wage
    treatment: effective_min_wage
    outcome: violent_crime_rate
    tier: primary
    methods: [staggered]
        """.strip(),
    )

    with pytest.raises(ValueError, match="methods requiring event timing: staggered"):
        load_project_config(tmp_path)


def _write_project_yaml(project_root: Path, body: str) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "project.yaml").write_text(body + "\n", encoding="utf-8")
