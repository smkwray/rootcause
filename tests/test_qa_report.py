"""Tests for QA report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from povcrime.reports.qa import build_data_quality_report


def test_build_data_quality_report(tmp_path: Path):
    """QA report builder should write the expected markdown sections."""
    panel = pd.DataFrame(
        {
            "county_fips": ["01001", "01001", "01003"],
            "state_fips": ["01", "01", "01"],
            "year": [2020, 2021, 2020],
            "source_share": [1.0, 0.75, 0.5],
            "control_completeness": [1.0, 0.9, 0.6],
            "low_coverage": [False, False, True],
            "effective_min_wage": [7.25, 8.00, 7.25],
            "broad_based_cat_elig": [1, 1, 0],
            "violent_crime_count": [10, 12, None],
            "property_crime_count": [20, 25, None],
        }
    )
    out_path = tmp_path / "qa" / "data_quality_report.md"

    result = build_data_quality_report(panel, out_path)
    text = result.read_text()

    assert result.exists()
    assert "## Panel Overview" in text
    assert "## Coverage Comparison" in text
    assert "## Treatment Availability" in text
