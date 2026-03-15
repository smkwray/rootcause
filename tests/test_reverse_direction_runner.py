"""Tests for reverse-direction runner helpers."""

from __future__ import annotations

import pandas as pd

from povcrime.models.reverse_direction import (
    lag_treatment_within_county,
    lead_treatment_within_county,
)


def test_lag_treatment_within_county_respects_entity_boundaries():
    panel = pd.DataFrame(
        {
            "county_fips": ["01001", "01001", "01003", "01003"],
            "year": [2020, 2021, 2020, 2021],
            "violent_crime_rate": [100.0, 120.0, 50.0, 60.0],
        }
    )

    result = lag_treatment_within_county(panel, treatment_col="violent_crime_rate")

    assert pd.isna(result.loc[(result["county_fips"] == "01001") & (result["year"] == 2020), "violent_crime_rate_lag1"]).all()
    assert result.loc[(result["county_fips"] == "01001") & (result["year"] == 2021), "violent_crime_rate_lag1"].iloc[0] == 100.0
    assert result.loc[(result["county_fips"] == "01003") & (result["year"] == 2021), "violent_crime_rate_lag1"].iloc[0] == 50.0


def test_lead_treatment_within_county_respects_entity_boundaries():
    panel = pd.DataFrame(
        {
            "county_fips": ["01001", "01001", "01003", "01003"],
            "year": [2020, 2021, 2020, 2021],
            "poverty_rate": [10.0, 11.0, 15.0, 16.0],
        }
    )

    result = lead_treatment_within_county(panel, treatment_col="poverty_rate")

    assert result.loc[
        (result["county_fips"] == "01001") & (result["year"] == 2020), "poverty_rate_lead1"
    ].iloc[0] == 11.0
    assert pd.isna(
        result.loc[
            (result["county_fips"] == "01003") & (result["year"] == 2021), "poverty_rate_lead1"
        ]
    ).all()
