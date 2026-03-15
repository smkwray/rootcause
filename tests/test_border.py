"""Tests for border-county helpers."""

from __future__ import annotations

import pandas as pd

from povcrime.models.border import (
    build_border_pair_panel,
    build_first_difference_border_panel,
    canonical_cross_state_pairs,
)


def test_canonical_cross_state_pairs_deduplicates_direction():
    adjacency = pd.DataFrame(
        {
            "county_fips": ["01003", "12033", "01001"],
            "neighbor_county_fips": ["12033", "01003", "01021"],
            "state_fips": ["01", "12", "01"],
            "neighbor_state_fips": ["12", "01", "01"],
        }
    )

    pairs = canonical_cross_state_pairs(adjacency)

    assert len(pairs) == 1
    assert pairs.loc[0, "pair_id"] == "01003__12033"
    assert pairs.loc[0, "border_state_pair"] == "01__12"


def test_build_border_pair_panel_creates_differences():
    pairs = pd.DataFrame(
        {
            "county_a": ["01003"],
            "county_b": ["12033"],
            "state_a": ["01"],
            "state_b": ["12"],
            "pair_id": ["01003__12033"],
            "border_state_pair": ["01__12"],
        }
    )
    panel = pd.DataFrame(
        {
            "county_fips": ["01003", "12033"],
            "year": [2020, 2020],
            "effective_min_wage": [7.25, 10.00],
            "violent_crime_rate": [200.0, 150.0],
            "poverty_rate": [11.0, 13.0],
        }
    )

    out = build_border_pair_panel(
        panel=panel,
        pairs=pairs,
        treatment="effective_min_wage",
        outcome="violent_crime_rate",
        controls=["poverty_rate"],
    )

    assert len(out) == 1
    assert out.loc[0, "diff_effective_min_wage"] == -2.75
    assert out.loc[0, "diff_violent_crime_rate"] == 50.0
    assert out.loc[0, "diff_poverty_rate"] == -2.0


def test_build_first_difference_border_panel_flags_one_sided_shocks():
    pair_panel = pd.DataFrame(
        {
            "pair_id": ["01003__12033", "01003__12033"],
            "border_state_pair": ["01__12", "01__12"],
            "year": [2019, 2020],
            "effective_min_wage_a": [7.25, 8.00],
            "effective_min_wage_b": [10.00, 10.00],
            "diff_effective_min_wage": [-2.75, -2.00],
            "diff_violent_crime_rate": [50.0, 40.0],
            "diff_poverty_rate": [-2.0, -1.0],
        }
    )

    out = build_first_difference_border_panel(
        pair_panel=pair_panel,
        treatment="effective_min_wage",
        outcome="violent_crime_rate",
        controls=["poverty_rate"],
    )

    assert out.loc[1, "delta_diff_effective_min_wage"] == 0.75
    assert out.loc[1, "delta_diff_violent_crime_rate"] == -10.0
    assert out.loc[1, "delta_diff_poverty_rate"] == 1.0
    assert bool(out.loc[1, "one_sided_treatment_shock"]) is True
