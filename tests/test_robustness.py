"""Tests for robustness helpers."""

from __future__ import annotations

import pandas as pd

from povcrime.models.robustness import (
    build_placebo_treatment,
    detrend_variables_within_entity,
)


def test_build_placebo_treatment_shifts_within_entity():
    df = pd.DataFrame(
        {
            "county_fips": ["01001", "01001", "01001", "01003", "01003"],
            "year": [2000, 2001, 2002, 2000, 2001],
            "effective_min_wage": [5.15, 5.50, 5.75, 5.15, 5.25],
        }
    )

    result = build_placebo_treatment(
        df,
        treatment_col="effective_min_wage",
        entity_col="county_fips",
        time_col="year",
        lead_periods=1,
        output_col="placebo",
    )

    assert result.loc[result["year"] == 2000, "placebo"].tolist() == [5.50, 5.25]
    assert pd.isna(result.loc[(result["county_fips"] == "01001") & (result["year"] == 2002), "placebo"]).all()


def test_detrend_variables_within_entity_adds_prefixed_columns():
    df = pd.DataFrame(
        {
            "county_fips": ["01001", "01001", "01001", "01003", "01003", "01003"],
            "year": [2000, 2001, 2002, 2000, 2001, 2002],
            "violent_crime_rate": [10.0, 12.0, 14.0, 3.0, 4.0, 5.0],
            "effective_min_wage": [5.15, 5.15, 5.50, 5.15, 5.25, 5.25],
        }
    )

    result = detrend_variables_within_entity(
        df,
        columns=["violent_crime_rate", "effective_min_wage"],
        entity_col="county_fips",
        time_col="year",
    )

    assert "dt_violent_crime_rate" in result.columns
    assert "dt_effective_min_wage" in result.columns
    assert abs(result.groupby("county_fips")["dt_violent_crime_rate"].mean()).max() < 1e-8
