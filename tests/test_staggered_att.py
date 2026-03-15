"""Tests for stacked staggered-att helpers."""

from __future__ import annotations

import pandas as pd

from povcrime.models.staggered_att import StaggeredEventStudy, build_stacked_event_panel


def test_build_stacked_event_panel_uses_not_yet_treated_controls():
    df = pd.DataFrame(
        {
            "county_fips": ["01001", "01001", "01003", "01003", "01005", "01005"],
            "year": [2019, 2020, 2019, 2020, 2019, 2020],
            "state_fips": ["01", "01", "01", "01", "01", "01"],
            "event_year": [2020, 2020, 2021, 2021, None, None],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )

    stacked = build_stacked_event_panel(
        df,
        event_col="event_year",
        leads=1,
        lags=0,
    )

    cohort_2020 = stacked.loc[stacked["stack_id"] == 2020]
    assert set(cohort_2020["county_fips"]) == {"01001", "01003", "01005"}
    assert cohort_2020.loc[cohort_2020["county_fips"] == "01001", "is_cohort_treated"].eq(1).all()
    assert cohort_2020.loc[cohort_2020["county_fips"] == "01003", "is_cohort_treated"].eq(0).all()


def test_staggered_event_study_fit_uses_numeric_time_index():
    rows: list[dict[str, object]] = []
    for county_fips, state_fips, event_year, base in [
        ("01001", "01", 2020, 10.0),
        ("01003", "01", 2021, 12.0),
        ("02001", "02", None, 9.0),
    ]:
        for year in range(2018, 2023):
            treated = event_year is not None and year >= event_year
            rows.append(
                {
                    "county_fips": county_fips,
                    "state_fips": state_fips,
                    "year": year,
                    "event_year": event_year,
                    "y": base + (2.0 if treated else 0.0) + 0.1 * (year - 2018),
                    "control_x": float(year - 2018),
                }
            )

    df = pd.DataFrame(rows)
    model = StaggeredEventStudy(
        df,
        outcome="y",
        event_col="event_year",
        controls=["control_x"],
        leads=1,
        lags=1,
    )

    result = model.fit()

    assert result.nobs > 0
    assert not model.coef_table().empty
    pretrend = model.pretrend_test()
    assert "interpretable" in pretrend
