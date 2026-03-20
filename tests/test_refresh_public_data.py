"""Tests for the public-data refresh script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refresh_public_data.py"
SPEC = importlib.util.spec_from_file_location("refresh_public_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
refresh_public_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh_public_data)


def test_build_site_data_includes_crime_data_provenance(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    outputs = root / "outputs" / "app"
    outputs.mkdir(parents=True)
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "index.html").write_text(
        '<html><body><script id="site-data" type="application/json">%%SITE_DATA%%</script></body></html>',
        encoding="utf-8",
    )

    (outputs / "results_summary.json").write_text(
        json.dumps(
            {
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
                    "qa_report": None,
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
                    }
                ],
                "exploratory": {
                    "bidirectional_poverty_crime": {
                        "estimands": [
                            {
                                "label": "poverty_to_violent",
                                "title": "Wrong Title",
                                "treatment": "wrong_treatment",
                                "outcome": "wrong_outcome",
                                "baseline_fe": {"coefficient": 0.2, "p_value": 0.4},
                                "dml": {"theta": 0.5, "p_value": 0.03, "panel_mode": "two_way_within"},
                                "overlap": {"max_abs_smd": 0.8, "panel_mode": "two_way_within"},
                                "headline": "Exploratory summary.",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "credibility_summary.json").write_text(
        json.dumps({"generated_date": "2026-03-20", "lanes": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(refresh_public_data, "ROOT", root)
    monkeypatch.setattr(refresh_public_data, "OUTPUTS", outputs)
    monkeypatch.setattr(refresh_public_data, "DEST", root / "docs" / "assets" / "data")

    refresh_public_data.build_site_data()

    site_data = json.loads((root / "docs" / "assets" / "data" / "site_data.json").read_text(encoding="utf-8"))
    assert site_data["panel"]["crime_data_level"] == "county_fallback"
    assert site_data["lanes"][0]["slug"] == "min_wage_violent"
    assert site_data["bidirectional"][0]["title"] == "Poverty -> Violent Crime"
    assert site_data["bidirectional"][0]["treatment"] == "poverty_rate"
