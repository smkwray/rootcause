"""Tests for baseline script helpers."""

from __future__ import annotations

import pandas as pd

from povcrime.models.policy_events import compute_first_treatment_event_year


def test_compute_first_treatment_event_year_handles_binary_adoption():
    panel = pd.DataFrame(
        {
            "county_fips": ["01001", "01001", "01001", "01003", "01003", "01003"],
            "year": [2018, 2019, 2020, 2018, 2019, 2020],
            "broad_based_cat_elig": [0, 1, 1, 0, 0, 0],
        }
    )

    result = compute_first_treatment_event_year(
        panel,
        treatment_col="broad_based_cat_elig",
        output_col="snap_bbce_event_year",
        change_threshold=0.5,
    )

    assert result.loc[result["county_fips"] == "01001", "snap_bbce_event_year"].iloc[0] == 2019
    assert result.loc[result["county_fips"] == "01003", "snap_bbce_event_year"].isna().all()
