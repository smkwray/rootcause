"""Tests for the shared report-artifact contract helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from povcrime.reports.contracts import (
    infer_crime_data_level,
    load_bidirectional_summary,
    load_credibility_summary,
    load_results_summary,
    validate_bidirectional_summary,
    validate_credibility_summary,
    validate_results_summary,
)


def test_validate_results_summary_requires_artifact_paths() -> None:
    summary = _valid_results_summary()
    del summary["artifacts"]["credibility_summary"]

    with pytest.raises(ValueError, match="credibility_summary"):
        validate_results_summary(summary)


def test_validate_results_summary_rejects_bad_panel_type() -> None:
    summary = _valid_results_summary()
    summary["panel"]["crime_data_level"] = "not-a-level"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="crime_data_level"):
        validate_results_summary(summary)


def test_load_results_summary_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "results_summary.json"
    path.write_text(json.dumps(_valid_results_summary()), encoding="utf-8")

    loaded = load_results_summary(path)

    assert loaded["panel"]["crime_data_level"] == "county_fallback"
    assert loaded["artifacts"]["credibility_summary"] == "outputs/app/credibility_summary.json"


def test_validate_results_summary_rejects_bad_nested_estimand_type() -> None:
    summary = _valid_results_summary()
    summary["estimands"][0]["dml"] = {"theta": "bad"}  # type: ignore[index]

    with pytest.raises(TypeError, match="estimands\\[0\\]\\.dml\\['theta'\\]"):
        validate_results_summary(summary)


def test_validate_credibility_summary_requires_lane_list() -> None:
    with pytest.raises(TypeError, match="lanes"):
        validate_credibility_summary({"generated_date": "2026-03-20", "lanes": {}})


def test_load_credibility_summary_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "credibility_summary.json"
    path.write_text(
        json.dumps(
            {
                "generated_date": "2026-03-20",
                "lanes": [
                    {
                        "slug": "min_wage_violent",
                        "title": "Minimum Wage -> Violent Crime",
                        "frontend_status": "primary_mixed_signal",
                        "headline_eligible": True,
                        "checks": [
                            {"name": "event_pretrend", "status": "pass", "detail": "p=0.5"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_credibility_summary(path)

    assert loaded["lanes"][0]["slug"] == "min_wage_violent"
    assert loaded["lanes"][0]["checks"][0]["name"] == "event_pretrend"


def test_load_bidirectional_summary_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "bidirectional_summary.json"
    path.write_text(json.dumps(_valid_bidirectional_summary()), encoding="utf-8")

    loaded = load_bidirectional_summary(path)

    assert loaded["estimands"][0]["label"] == "poverty_to_violent"
    assert loaded["estimands"][0]["treatment"] == "poverty_rate"


def test_validate_bidirectional_summary_rejects_missing_metadata() -> None:
    summary = _valid_bidirectional_summary()
    del summary["estimands"][0]["title"]

    with pytest.raises(ValueError, match="title"):
        validate_bidirectional_summary(summary)


def test_infer_crime_data_level_distinguishes_tiers() -> None:
    county_panel = pd.DataFrame(
        {
            "county_fips": ["01001", "01003"],
            "violent_crime_rate": [100.0, 120.0],
            "property_crime_rate": [900.0, 950.0],
        }
    )
    state_panel = pd.DataFrame(
        {
            "county_fips": ["01000", "02000"],
            "violent_crime_rate": [100.0, 120.0],
            "property_crime_rate": [900.0, 950.0],
        }
    )
    missing_panel = pd.DataFrame({"county_fips": ["01001"], "violent_crime_rate": [None]})

    assert infer_crime_data_level(county_panel) == "county_fallback"
    assert infer_crime_data_level(state_panel) == "state_estimate"
    assert infer_crime_data_level(missing_panel) == "missing"


def _valid_results_summary() -> dict[str, object]:
    return {
        "generated_date": "2026-03-20",
        "panel": {
            "rows": 2,
            "counties": 2,
            "year_min": 2020,
            "year_max": 2020,
            "violent_rows": 2,
            "property_rows": 2,
            "low_coverage_rows": 0,
            "crime_data_level": "county_fallback",
            "available_sources": [
                {
                    "name": "saipe",
                    "columns": ["poverty_rate", "population"],
                    "non_null_rows": 2,
                    "share": 1.0,
                    "min_year_share": 1.0,
                    "max_year_share": 1.0,
                    "missing_rows": 0,
                }
            ],
        },
        "artifacts": {
            "qa_report": "outputs/qa/data_quality_report.md",
            "final_report": None,
            "robustness_summary": None,
            "border_summary": None,
            "min_wage_identification_summary": None,
            "min_wage_event_study_summary": None,
            "min_wage_dose_bucket_summary": None,
            "min_wage_negative_control_treatment_summary": None,
            "support_trim_summary": None,
            "negative_control_summary": None,
            "staggered_summary": None,
            "crime_measurement_validation": None,
            "credibility_summary": "outputs/app/credibility_summary.json",
            "bidirectional_poverty_crime_summary": None,
        },
        "research_lanes": {
            "primary": ["min_wage_violent"],
            "secondary": [],
            "exploratory": [],
        },
        "estimands": [
            {
                "slug": "min_wage_violent",
                "title": "Minimum Wage -> Violent Crime",
                "dml": {
                    "theta": 0.2,
                    "se": 0.1,
                    "p_value": 0.04,
                    "panel_mode": "two_way_within",
                },
                "frontend": {
                    "display_priority": "primary",
                    "headline_eligible": True,
                    "status": "primary_mixed_signal",
                },
            }
        ],
        "exploratory": {},
    }


def _valid_bidirectional_summary() -> dict[str, object]:
    return {
        "generated_date": "2026-03-20",
        "design": {
            "baseline_timing": "one-year lagged treatment",
            "dml_panel_mode": "two_way_within",
        },
        "estimands": [
            {
                "label": "poverty_to_violent",
                "title": "Poverty -> Violent Crime",
                "treatment": "poverty_rate",
                "outcome": "violent_crime_rate",
                "baseline_fe": {"coefficient": 0.5, "p_value": 0.2},
                "dml": {"theta": 0.7, "p_value": 0.03, "panel_mode": "two_way_within"},
                "overlap": {"max_abs_smd": 0.8, "panel_mode": "two_way_within"},
                "robustness": [{"spec": "placebo_lead1_high_coverage", "p_value": 0.4}],
                "headline": "Exploratory summary.",
            }
        ],
    }
