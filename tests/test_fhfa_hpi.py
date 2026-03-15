"""Tests for FHFA HPI parsing."""

from __future__ import annotations

import pandas as pd

from povcrime.data.fhfa_hpi import _reshape_fhfa_hpi


def test_reshape_fhfa_hpi_builds_expected_schema():
    raw = pd.DataFrame(
        {
            "State": ["AL", "AL"],
            "County": ["Autauga", "Autauga"],
            "FIPS code": ["01001", "01001"],
            "Year": [2000, 2001],
            "Annual Change (%)": [2.1, 3.0],
            "HPI": [123.4, 127.1],
            "HPI with 1990 base": [118.0, 121.6],
            "HPI with 2000 base": [100.0, 103.0],
        }
    )

    df = _reshape_fhfa_hpi(raw)

    assert df.columns.tolist() == [
        "county_fips",
        "state_fips",
        "year",
        "fhfa_hpi",
        "fhfa_hpi_1990_base",
        "fhfa_hpi_2000_base",
        "fhfa_annual_change_pct",
    ]
    assert df.loc[0, "county_fips"] == "01001"
    assert df.loc[0, "state_fips"] == "01"
    assert df.loc[1, "fhfa_hpi_2000_base"] == 103.0
