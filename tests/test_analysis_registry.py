"""Tests for the shared analysis-lane registry helpers."""

from __future__ import annotations

from povcrime.analysis import (
    get_analysis_lane,
    get_analysis_lanes,
    get_bidirectional_lane,
    get_bidirectional_lanes,
    get_event_definitions,
    get_falsification_treatment_specs,
)


def test_get_analysis_lanes_filters_by_method() -> None:
    baseline_slugs = {lane.slug for lane in get_analysis_lanes(method="baseline")}
    dml_slugs = {lane.slug for lane in get_analysis_lanes(method="dml")}
    staggered_slugs = {lane.slug for lane in get_analysis_lanes(method="staggered")}

    assert "snap_bbce_violent" in baseline_slugs
    assert "snap_bbce_violent" not in dml_slugs
    assert "tanf_violent" in dml_slugs
    assert staggered_slugs == {
        "min_wage_violent",
        "min_wage_property",
        "snap_bbce_violent",
        "snap_bbce_property",
    }


def test_get_analysis_lane_returns_canonical_metadata() -> None:
    lane = get_analysis_lane("eitc_violent")

    assert lane is not None
    assert lane.title == "State EITC Rate -> Violent Crime"
    assert lane.family == "state_eitc"
    assert lane.tier == "secondary"


def test_get_analysis_lanes_preserve_configured_tier_grouping() -> None:
    primary_slugs = {lane.slug for lane in get_analysis_lanes(tiers={"primary"})}
    secondary_slugs = {lane.slug for lane in get_analysis_lanes(tiers={"secondary"})}
    exploratory_slugs = {lane.slug for lane in get_analysis_lanes(tiers={"exploratory"})}

    assert primary_slugs == {"min_wage_violent", "min_wage_property"}
    assert secondary_slugs == {"eitc_violent", "eitc_property"}
    assert exploratory_slugs == {
        "snap_bbce_violent",
        "snap_bbce_property",
        "tanf_violent",
        "tanf_property",
    }


def test_get_event_definitions_deduplicates_shared_event_columns() -> None:
    baseline_defs = get_event_definitions(method="baseline")
    staggered_defs = get_event_definitions(method="staggered")

    assert len(baseline_defs) == 3
    assert len(staggered_defs) == 2
    assert {event.output_col for event in staggered_defs} == {
        "min_wage_event_year",
        "snap_bbce_event_year",
    }


def test_get_falsification_treatment_specs_are_unique() -> None:
    specs = get_falsification_treatment_specs()

    assert specs == [
        {"label": "min_wage", "treatment": "effective_min_wage"},
        {"label": "eitc", "treatment": "state_eitc_rate"},
    ]


def test_get_bidirectional_lanes_return_canonical_metadata() -> None:
    slugs = {lane.slug for lane in get_bidirectional_lanes()}

    assert slugs == {
        "poverty_to_violent",
        "poverty_to_property",
        "violent_to_poverty",
        "property_to_poverty",
    }

    lane = get_bidirectional_lane("violent_to_poverty")
    assert lane is not None
    assert lane.title == "Violent Crime -> Poverty"
    assert lane.treatment == "violent_crime_rate"
    assert lane.outcome == "poverty_rate"


def test_policy_and_bidirectional_lane_registries_are_disjoint() -> None:
    policy_slugs = {lane.slug for lane in get_analysis_lanes()}
    bidirectional_slugs = {lane.slug for lane in get_bidirectional_lanes()}

    assert policy_slugs.isdisjoint(bidirectional_slugs)
