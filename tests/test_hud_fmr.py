"""Tests for HUD FMR parsing."""

from __future__ import annotations

import pandas as pd

from povcrime.data.hud_fmr import _reshape_fmr_history


def test_reshape_fmr_history_builds_county_year_rows():
    raw = pd.DataFrame(
        {
            "fips": [100199999],
            "fmr00_0": ["$395"],
            "fmr00_1": ["$422"],
            "fmr00_2": ["$499"],
            "fmr00_3": ["$679"],
            "fmr00_4": ["$818"],
            "fmr01_0": ["$402"],
            "fmr01_1": ["$429"],
            "fmr01_2": ["$507"],
            "fmr01_3": ["$691"],
            "fmr01_4": ["$832"],
        }
    )

    df = _reshape_fmr_history(raw)

    assert set(df["year"]) == {2000, 2001}
    assert (df["county_fips"] == "01001").all()
    assert df.loc[df["year"] == 2000, "fair_market_rent_2br"].iloc[0] == 499


def test_reshape_fmr_history_prefers_county_row_over_subcounty_rows():
    raw = pd.DataFrame(
        {
            "fips": [100199999, 100101111, 100102222],
            "cousub": [99999, 1111, 2222],
            "fmr00_0": ["$395", "$100", "$200"],
            "fmr00_1": ["$422", "$101", "$201"],
            "fmr00_2": ["$499", "$102", "$202"],
            "fmr00_3": ["$679", "$103", "$203"],
            "fmr00_4": ["$818", "$104", "$204"],
        }
    )

    df = _reshape_fmr_history(raw)

    assert len(df) == 1
    assert df.loc[0, "fair_market_rent_2br"] == 499


def test_reshape_fmr_history_collapses_subcounty_only_counties_to_median():
    raw = pd.DataFrame(
        {
            "fips": [2300102060, 2300119105, 2300129255],
            "cousub": [2060, 19105, 29255],
            "fmr00_0": ["$390", "$410", "$430"],
            "fmr00_1": ["$490", "$510", "$530"],
            "fmr00_2": ["$590", "$610", "$630"],
            "fmr00_3": ["$690", "$710", "$730"],
            "fmr00_4": ["$790", "$810", "$830"],
        }
    )

    df = _reshape_fmr_history(raw)

    assert len(df) == 1
    assert df.loc[0, "county_fips"] == "23001"
    assert df.loc[0, "fair_market_rent_2br"] == 610
