"""Tests for county adjacency parsing."""

from __future__ import annotations

import pandas as pd

from povcrime.data.county_adjacency import _reshape_adjacency


def test_reshape_adjacency_fills_source_county_and_extracts_states():
    raw = pd.DataFrame(
        {
            0: ['"Autauga County, AL"', "", ""],
            1: ["01001", "", ""],
            2: ['"Autauga County, AL"', '"Chilton County, AL"', '"Dallas County, AL"'],
            3: ["01001", "01021", "01047"],
        }
    )
    raw.columns = ["county_name", "county_fips", "neighbor_county_name", "neighbor_county_fips"]

    out = _reshape_adjacency(raw)

    assert out["county_fips"].tolist() == ["01001", "01001", "01001"]
    assert out["neighbor_county_fips"].tolist() == ["01001", "01021", "01047"]
    assert out["state_fips"].tolist() == ["01", "01", "01"]
