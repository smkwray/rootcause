"""Tests for the FBI crime adapter and panel merge safeguards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from povcrime.config import ProjectConfig
from povcrime.data.fbi_crime import FBICrimeAdapter
from povcrime.processing.panel import build_county_year_panel
from scripts import build_panel, download_data


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
    assert adapter.county_fallback_file() == county_file
    assert adapter.has_county_fallback_file()
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


def test_download_data_builds_county_fallback_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """The downloader should materialize the county fallback automatically."""
    config = ProjectConfig(project_root=tmp_path, start_year=2020, end_year=2020)

    class FakeFBIAdapter:
        def __init__(self, config: ProjectConfig) -> None:
            self.config = config

        def download(self) -> None:
            return None

        def has_county_fallback_file(self) -> bool:
            return (self.config.raw_dir / "fbi_crime" / "county_crime.parquet").exists()

    def fake_build_county_fallback(
        *,
        start_year: int,
        end_year: int,
        raw_dir: Path,
        output_path: Path,
    ) -> pd.DataFrame:
        raw_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "county_fips": ["01001"],
                "state_fips": ["01"],
                "year": [2020],
                "violent_crime_count": [11],
                "property_crime_count": [22],
                "population_covered": [50000],
                "agencies_reporting": [3],
                "reported_month_share": [1.0],
                "coverage_pass_flag": [True],
            }
        )
        df.to_parquet(output_path, index=False)
        return df

    monkeypatch.setattr(download_data, "get_config", lambda: config)
    monkeypatch.setattr(
        "povcrime.data.fbi_crime.FBICrimeAdapter",
        FakeFBIAdapter,
    )
    monkeypatch.setattr(
        "povcrime.data.fbi_reta_master.build_county_fallback",
        fake_build_county_fallback,
    )

    download_data.main(["--sources", "fbi_crime"])

    assert (config.raw_dir / "fbi_crime" / "county_crime.parquet").exists()


def test_build_panel_requires_county_fbi_data_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Default panel builds should fail when FBI data are state-only."""
    config = ProjectConfig(project_root=tmp_path, start_year=2020, end_year=2020)

    class FakeSAIPEAdapter:
        def __init__(self, config: ProjectConfig) -> None:
            self.config = config

        def load(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "county_fips": ["01001", "01003"],
                    "state_fips": ["01", "01"],
                    "year": [2020, 2020],
                    "poverty_rate": [15.0, 12.0],
                    "median_hh_income": [50000, 55000],
                    "poverty_count": [8000, 9000],
                    "population": [50000, 60000],
                }
            )

        def validate(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    class FakeFBIAdapter:
        def __init__(self, config: ProjectConfig) -> None:
            self.config = config

        def load(self) -> pd.DataFrame:
            return pd.DataFrame(
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
            )

        def validate(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

        def has_county_fallback_file(self) -> bool:
            return False

    monkeypatch.setattr(build_panel, "get_config", lambda: config)
    monkeypatch.setattr(
        "povcrime.data.saipe.SAIPEAdapter",
        FakeSAIPEAdapter,
    )
    monkeypatch.setattr(
        "povcrime.data.fbi_crime.FBICrimeAdapter",
        FakeFBIAdapter,
    )

    with pytest.raises(SystemExit, match="1"):
        build_panel.main([])


def test_build_panel_allows_missing_county_fbi_data_with_escape_hatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """The explicit escape hatch should allow a panel without FBI crime."""
    config = ProjectConfig(project_root=tmp_path, start_year=2020, end_year=2020)

    class FakeSAIPEAdapter:
        def __init__(self, config: ProjectConfig) -> None:
            self.config = config

        def load(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "county_fips": ["01001", "01003"],
                    "state_fips": ["01", "01"],
                    "year": [2020, 2020],
                    "poverty_rate": [15.0, 12.0],
                    "median_hh_income": [50000, 55000],
                    "poverty_count": [8000, 9000],
                    "population": [50000, 60000],
                }
            )

        def validate(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    class FakeFBIAdapter:
        def __init__(self, config: ProjectConfig) -> None:
            self.config = config

        def load(self) -> pd.DataFrame:
            return pd.DataFrame(
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
            )

        def validate(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

        def has_county_fallback_file(self) -> bool:
            return False

    monkeypatch.setattr(build_panel, "get_config", lambda: config)
    monkeypatch.setattr(
        "povcrime.data.saipe.SAIPEAdapter",
        FakeSAIPEAdapter,
    )
    monkeypatch.setattr(
        "povcrime.data.fbi_crime.FBICrimeAdapter",
        FakeFBIAdapter,
    )

    build_panel.main(["--allow-missing-county-crime"])

    panel_path = config.processed_dir / "panel.parquet"
    assert panel_path.exists()
    panel = pd.read_parquet(panel_path)
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
