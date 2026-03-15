"""Tests for povcrime.processing.coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from povcrime.processing.coverage import (
    compute_coverage_metrics,
    coverage_summary,
    flag_low_coverage,
)


@pytest.fixture()
def sample_panel():
    """Create a small panel with known coverage patterns."""
    return pd.DataFrame({
        "county_fips": ["01001", "01001", "01003", "01003"],
        "state_fips": ["01", "01", "01", "01"],
        "year": [2020, 2021, 2020, 2021],
        "poverty_rate": [15.0, 16.0, None, 14.0],
        "unemployment_rate": [5.0, 5.5, 4.0, None],
        "per_capita_personal_income": [30000, 31000, None, None],
        "pct_male": [49.0, 49.0, 48.0, 48.0],
        "violent_crime_count": [100, 110, None, None],
        "population": [50000, 51000, 30000, 31000],
    })


def test_compute_coverage_metrics(sample_panel):
    result = compute_coverage_metrics(sample_panel)
    assert "source_count" in result.columns
    assert "source_share" in result.columns
    assert "control_completeness" in result.columns
    # First row has all sources present.
    assert result.loc[0, "source_count"] >= 4


def test_flag_low_coverage(sample_panel):
    with_metrics = compute_coverage_metrics(sample_panel)
    result = flag_low_coverage(with_metrics, threshold=0.75)
    assert "low_coverage" in result.columns
    assert result["low_coverage"].dtype == bool


def test_flag_low_coverage_requires_source_share():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="source_share"):
        flag_low_coverage(df)


def test_coverage_summary(sample_panel):
    with_metrics = compute_coverage_metrics(sample_panel)
    with_flags = flag_low_coverage(with_metrics)
    summary = coverage_summary(with_flags)
    assert "state_fips" in summary.columns
    assert "year" in summary.columns
    assert "mean_source_share" in summary.columns
    assert len(summary) > 0
