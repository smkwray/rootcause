"""Tests for frontend-facing app artifact outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from povcrime.reports.app_artifacts import build_app_artifacts


def test_build_app_artifacts_writes_manifest_and_summary(tmp_path: Path):
    project_root = tmp_path
    output_dir = project_root / "outputs"
    (output_dir / "qa").mkdir(parents=True)
    (output_dir / "report").mkdir(parents=True)
    (output_dir / "baseline" / "min_wage_violent").mkdir(parents=True)
    (output_dir / "dml" / "min_wage_violent").mkdir(parents=True)
    (output_dir / "robustness").mkdir(parents=True)
    (output_dir / "overlap" / "min_wage_violent").mkdir(parents=True)

    (output_dir / "qa" / "data_quality_report.md").write_text("# QA\n", encoding="utf-8")
    (output_dir / "report" / "final_report.md").write_text("# Report\n", encoding="utf-8")
    pd.DataFrame(
        {
            "variable": ["effective_min_wage"],
            "coefficient": [1.2],
            "std_error": [0.3],
            "t_stat": [4.0],
            "p_value": [0.01],
            "ci_lower": [0.6],
            "ci_upper": [1.8],
        }
    ).to_csv(output_dir / "baseline" / "min_wage_violent" / "baseline_fe_summary.csv", index=False)
    (output_dir / "baseline" / "min_wage_violent" / "event_study_pretrend.json").write_text(
        json.dumps({"p_value": 0.4}),
        encoding="utf-8",
    )
    (output_dir / "baseline" / "min_wage_violent" / "event_study_coefs.csv").write_text(
        "relative_time,coefficient\n0,0.1\n",
        encoding="utf-8",
    )
    (output_dir / "baseline" / "min_wage_violent" / "event_study_plot.png").write_bytes(b"png")
    (output_dir / "dml" / "min_wage_violent" / "dml_effective_min_wage__violent_crime_rate.json").write_text(
        json.dumps({"theta": 0.5, "se": 0.1, "p_value": 0.01}),
        encoding="utf-8",
    )
    (output_dir / "dml" / "min_wage_violent" / "causal_forest_ate_effective_min_wage__violent_crime_rate.json").write_text(
        json.dumps({"ate": 0.4, "ci_lower": 0.1, "ci_upper": 0.7}),
        encoding="utf-8",
    )
    pd.DataFrame({"feature": ["poverty_rate"], "importance": [0.6]}).to_csv(
        output_dir / "dml" / "min_wage_violent" / "causal_forest_importance_effective_min_wage__violent_crime_rate.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "label": ["min_wage_violent"],
            "spec": ["all_rows"],
            "coefficient": [1.0],
            "std_error": [0.2],
            "p_value": [0.1],
            "n_obs_used": [100],
        }
    ).to_csv(output_dir / "robustness" / "robustness_summary.csv", index=False)
    (output_dir / "overlap" / "min_wage_violent" / "support_summary.json").write_text(
        json.dumps({"oof_r2": 0.2, "max_abs_smd": 0.3}),
        encoding="utf-8",
    )
    (output_dir / "overlap" / "min_wage_violent" / "control_balance.csv").write_text(
        "control,abs_smd\npoverty_rate,0.3\n",
        encoding="utf-8",
    )
    (output_dir / "overlap" / "min_wage_violent" / "support_bins.csv").write_text(
        "predicted_bin,n_obs\n0,10\n",
        encoding="utf-8",
    )
    (output_dir / "exploratory" / "bidirectional_poverty_crime").mkdir(parents=True)
    (output_dir / "exploratory" / "bidirectional_poverty_crime" / "bidirectional_summary.json").write_text(
        json.dumps({"estimands": [{"label": "poverty_to_violent"}]}),
        encoding="utf-8",
    )
    (output_dir / "border").mkdir(parents=True)
    (output_dir / "border" / "border_summary.csv").write_text("label,spec\nx,baseline\n", encoding="utf-8")
    (output_dir / "min_wage_identification").mkdir(parents=True)
    (output_dir / "min_wage_identification" / "min_wage_identification_summary.csv").write_text(
        "outcome_label,spec\nviolent,border_pair_first_difference\n",
        encoding="utf-8",
    )
    (output_dir / "min_wage_identification" / "min_wage_event_study_summary.csv").write_text(
        "outcome_label,spec\nviolent,border_pair_first_difference_event_study\n",
        encoding="utf-8",
    )
    (output_dir / "min_wage_identification" / "min_wage_dose_bucket_summary.csv").write_text(
        "outcome_label,dose_bucket\nviolent,small\n",
        encoding="utf-8",
    )
    (output_dir / "min_wage_identification" / "min_wage_negative_control_treatment_summary.csv").write_text(
        "outcome_label,spec,p_value\nviolent,border_pair_first_difference_negative_control_treatment,0.4\n",
        encoding="utf-8",
    )
    (output_dir / "support_trim").mkdir(parents=True)
    (output_dir / "support_trim" / "support_trim_summary.csv").write_text("label,theta_base\nx,1.0\n", encoding="utf-8")
    (output_dir / "falsification").mkdir(parents=True)
    (output_dir / "falsification" / "negative_control_summary.csv").write_text("label,p_value\nx,0.1\n", encoding="utf-8")
    (output_dir / "staggered").mkdir(parents=True)
    (output_dir / "staggered" / "staggered_summary.csv").write_text("label,interpretable\nx,True\n", encoding="utf-8")
    (output_dir / "crime_validation").mkdir(parents=True)
    (output_dir / "crime_validation" / "crime_measurement_validation.json").write_text(
        json.dumps({"robustness_sensitivity": []}),
        encoding="utf-8",
    )

    panel = pd.DataFrame(
        {
            "county_fips": ["01001"],
            "year": [2020],
            "poverty_rate": [10.0],
            "violent_crime_rate": [100.0],
            "property_crime_rate": [1000.0],
            "low_coverage": [False],
            "effective_min_wage": [7.25],
        }
    )

    manifest_path, summary_path = build_app_artifacts(
        project_root=project_root,
        panel=panel,
        output_dir=output_dir,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    credibility = json.loads((output_dir / "app" / "credibility_summary.json").read_text(encoding="utf-8"))

    assert manifest["files"]
    assert "project_root" not in manifest
    assert summary["artifacts"]["final_report"] == "outputs/report/final_report.md"
    assert summary["artifacts"]["border_summary"] == "outputs/border/border_summary.csv"
    assert summary["artifacts"]["min_wage_identification_summary"] == (
        "outputs/min_wage_identification/min_wage_identification_summary.csv"
    )
    assert summary["artifacts"]["min_wage_event_study_summary"] == (
        "outputs/min_wage_identification/min_wage_event_study_summary.csv"
    )
    assert summary["artifacts"]["min_wage_dose_bucket_summary"] == (
        "outputs/min_wage_identification/min_wage_dose_bucket_summary.csv"
    )
    assert summary["artifacts"]["min_wage_negative_control_treatment_summary"] == (
        "outputs/min_wage_identification/min_wage_negative_control_treatment_summary.csv"
    )
    assert summary["artifacts"]["support_trim_summary"] == "outputs/support_trim/support_trim_summary.csv"
    assert summary["artifacts"]["negative_control_summary"] == "outputs/falsification/negative_control_summary.csv"
    assert summary["artifacts"]["staggered_summary"] == "outputs/staggered/staggered_summary.csv"
    assert summary["artifacts"]["crime_measurement_validation"] == (
        "outputs/crime_validation/crime_measurement_validation.json"
    )
    assert summary["artifacts"]["credibility_summary"] == "outputs/app/credibility_summary.json"
    assert summary["artifacts"]["bidirectional_poverty_crime_summary"] == (
        "outputs/exploratory/bidirectional_poverty_crime/bidirectional_summary.json"
    )
    assert summary["exploratory"]["bidirectional_poverty_crime"]["estimands"][0]["label"] == "poverty_to_violent"
    assert summary["estimands"][0]["slug"] == "min_wage_violent"
    assert summary["research_lanes"]["primary"] == ["min_wage_violent"]
    assert summary["estimands"][0]["frontend"]["display_priority"] == "primary"
    assert summary["estimands"][0]["frontend"]["headline_eligible"] is True
    assert credibility["lanes"][0]["slug"] == "min_wage_violent"
    assert any(source["name"] == "saipe" for source in summary["panel"]["available_sources"])
