"""Tests for povcrime.processing.geography."""

from __future__ import annotations

import pandas as pd
import pytest

from povcrime.processing.geography import (
    VALID_STATE_FIPS,
    merge_state_to_county,
    standardize_fips_format,
    validate_county_fips,
)


def test_standardize_fips_county():
    df = pd.DataFrame({"county_fips": ["1001", "6037", "48201"]})
    result = standardize_fips_format(df, "county_fips")
    assert result["county_fips"].tolist() == ["01001", "06037", "48201"]


def test_standardize_fips_state():
    df = pd.DataFrame({"state_fips": ["1", "6", "48"]})
    result = standardize_fips_format(df, "state_fips")
    assert result["state_fips"].tolist() == ["01", "06", "48"]


def test_validate_county_fips_keeps_valid():
    df = pd.DataFrame({
        "county_fips": ["01001", "06037", "48201"],
        "year": [2020, 2020, 2020],
    })
    result = validate_county_fips(df)
    assert len(result) == 3


def test_validate_county_fips_removes_state_totals():
    df = pd.DataFrame({
        "county_fips": ["01000", "01001", "06000", "06037"],
        "year": [2020, 2020, 2020, 2020],
    })
    result = validate_county_fips(df)
    assert len(result) == 2
    assert "01000" not in result["county_fips"].values
    assert "06000" not in result["county_fips"].values


def test_validate_county_fips_removes_invalid_state():
    df = pd.DataFrame({
        "county_fips": ["01001", "99001"],  # 99 is not a valid state
        "year": [2020, 2020],
    })
    result = validate_county_fips(df)
    assert len(result) == 1
    assert result["county_fips"].iloc[0] == "01001"


def test_validate_county_fips_requires_column():
    df = pd.DataFrame({"fips": ["01001"]})
    with pytest.raises(ValueError, match="county_fips"):
        validate_county_fips(df)


def test_merge_state_to_county():
    county_df = pd.DataFrame({
        "county_fips": ["01001", "01003", "06037"],
        "state_fips": ["01", "01", "06"],
        "year": [2020, 2020, 2020],
        "population": [100, 200, 300],
    })
    state_df = pd.DataFrame({
        "state_fips": ["01", "06"],
        "year": [2020, 2020],
        "min_wage": [7.25, 15.00],
    })
    result = merge_state_to_county(state_df, county_df)
    assert len(result) == 3
    assert "min_wage" in result.columns
    assert result.loc[result["county_fips"] == "01001", "min_wage"].iloc[0] == 7.25
    assert result.loc[result["county_fips"] == "06037", "min_wage"].iloc[0] == 15.00


def test_valid_state_fips_count():
    assert len(VALID_STATE_FIPS) == 51  # 50 states + DC
