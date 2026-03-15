"""Tests for the FBI crime adapter and panel merge safeguards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from povcrime.config import ProjectConfig
from povcrime.data.fbi_crime import FBICrimeAdapter
from povcrime.processing.panel import build_county_year_panel


def test_fbi_adapter_prefers_manual_county_file(tmp_path: Path):
    """Manual county-level fallback files should autoload when present."""
    config = ProjectConfig(project_root=tmp_path, start_year=2020, end_year=2020)
    raw_dir = config.raw_dir / "fbi_crime"
    raw_dir.mkdir(parents=True, exist_ok=True)

    county_file = raw_dir / "county_crime.csv"
    pd.DataFrame(
        {
            "county_fips": ["01001"],
            "state_fips": ["01"],
            "year": [2020],
            "violent_crime_count": [10],
            "property_crime_count": [20],
            "population_covered": [50000],
            "agencies_reporting": [3],
            "reported_month_share": [1.0],
            "coverage_pass_flag": [True],
        }
    ).to_csv(county_file, index=False)

    adapter = FBICrimeAdapter(config)
    result = adapter.load()

    assert len(result) == 1
    assert result.loc[0, "county_fips"] == "01001"
    assert result.loc[0, "violent_crime_count"] == 10


def test_build_panel_skips_state_level_fbi_merge(tmp_path: Path):
    """State-level FBI rows must not be broadcast as county-level outcomes."""
    config = ProjectConfig(project_root=tmp_path, start_year=2020, end_year=2020)
    sources = {
        "saipe": pd.DataFrame(
            {
                "county_fips": ["01001", "01003"],
                "state_fips": ["01", "01"],
                "year": [2020, 2020],
                "poverty_rate": [15.0, 12.0],
                "median_hh_income": [50000, 55000],
                "poverty_count": [8000, 9000],
                "population": [50000, 60000],
            }
        ),
        "fbi_crime": pd.DataFrame(
            {
                "county_fips": ["01000"],
                "state_fips": ["01"],
                "year": [2020],
                "violent_crime_count": [1000],
                "property_crime_count": [4000],
                "population_covered": [4900000],
                "agencies_reporting": [pd.NA],
                "reported_month_share": [pd.NA],
                "coverage_pass_flag": [True],
            }
        ),
    }

    panel = build_county_year_panel(sources, config)

    assert "violent_crime_count" not in panel.columns
    assert "property_crime_count" not in panel.columns


def test_build_panel_clips_fbi_population_covered_to_county_population(tmp_path: Path):
    """County-level FBI covered population should not exceed county population."""
    config = ProjectConfig(project_root=tmp_path, start_year=2020, end_year=2020)
    sources = {
        "saipe": pd.DataFrame(
            {
                "county_fips": ["01001"],
                "state_fips": ["01"],
                "year": [2020],
                "poverty_rate": [15.0],
                "median_hh_income": [50000],
                "poverty_count": [8000],
                "population": [50000],
            }
        ),
        "fbi_crime": pd.DataFrame(
            {
                "county_fips": ["01001"],
                "state_fips": ["01"],
                "year": [2020],
                "violent_crime_count": [10],
                "property_crime_count": [20],
                "population_covered": [65000],
                "agencies_reporting": [2],
                "reported_month_share": [pd.NA],
                "coverage_pass_flag": [False],
            }
        ),
    }

    panel = build_county_year_panel(sources, config)

    assert panel.loc[0, "population_covered"] == 50000
