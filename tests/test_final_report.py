"""Tests for final report assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from povcrime.reports.final_report import build_final_report


def test_build_final_report_writes_markdown(tmp_path: Path):
    panel = pd.DataFrame(
        {
            "county_fips": ["01001", "01003"],
            "year": [2020, 2020],
            "violent_crime_rate": [100.0, 200.0],
            "property_crime_rate": [1000.0, 1200.0],
            "source_share": [0.8, 1.0],
            "low_coverage": [False, False],
        }
    )

    panel_path = tmp_path / "data" / "processed" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    panel.to_parquet(panel_path, index=False)

    qa_path = tmp_path / "outputs" / "qa" / "data_quality_report.md"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text("# QA\n", encoding="utf-8")

    for slug in ("min_wage_violent", "min_wage_property"):
        base_dir = tmp_path / "outputs" / "baseline" / slug
        base_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "variable": ["effective_min_wage"],
                "coefficient": [1.23],
                "std_error": [0.45],
                "t_stat": [2.7],
                "p_value": [0.01],
                "ci_lower": [0.3],
                "ci_upper": [2.1],
            }
        ).to_csv(base_dir / "baseline_fe_summary.csv", index=False)
        pd.DataFrame(
            {
                "relative_time": [-1, 0, 1],
                "coefficient": [0.0, 0.2, 0.3],
                "std_error": [0.0, 0.1, 0.1],
                "ci_lower": [0.0, 0.0, 0.1],
                "ci_upper": [0.0, 0.4, 0.5],
                "p_value": [1.0, 0.1, 0.05],
            }
        ).to_csv(base_dir / "event_study_coefs.csv", index=False)
        (base_dir / "event_study_plot.png").write_bytes(b"png")
        (base_dir / "event_study_pretrend.json").write_text(
            json.dumps({"f_stat": 1.0, "p_value": 0.2, "n_pre_coefs": 2, "pass": True}),
            encoding="utf-8",
        )

        dml_dir = tmp_path / "outputs" / "dml" / slug
        dml_dir.mkdir(parents=True)
        (dml_dir / f"dml_effective_min_wage__{slug}.json").write_text(
            json.dumps(
                {
                    "theta": 0.5,
                    "se": 0.2,
                    "p_value": 0.03,
                    "ci_lower": 0.1,
                    "ci_upper": 0.9,
                }
            ),
            encoding="utf-8",
        )
        (dml_dir / f"causal_forest_ate_effective_min_wage__{slug}.json").write_text(
            json.dumps({"ate": 0.4, "ci_lower": -0.2, "ci_upper": 1.0}),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "feature": ["pct_black", "log_population", "poverty_rate"],
                "importance": [0.4, 0.3, 0.2],
            }
        ).to_csv(
            dml_dir / f"causal_forest_importance_effective_min_wage__{slug}.csv",
            index=False,
        )

    robustness_path = tmp_path / "outputs" / "robustness" / "robustness_summary.csv"
    robustness_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "label": ["min_wage_violent"],
            "spec": ["baseline_high_coverage"],
            "coefficient": [1.0],
            "std_error": [0.2],
            "p_value": [0.01],
            "n_obs_used": [100],
        }
    ).to_csv(robustness_path, index=False)

    overlap_dir = tmp_path / "outputs" / "overlap"
    for slug in ("min_wage_violent", "min_wage_property", "eitc_violent", "tanf_violent"):
        spec_dir = overlap_dir / slug
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "support_summary.json").write_text(
            json.dumps(
                {
                    "oof_r2": 0.2,
                    "residual_to_treatment_std": 0.8,
                    "max_abs_smd": 0.3,
                }
            ),
            encoding="utf-8",
        )

    app_dir = tmp_path / "outputs" / "app"
    app_dir.mkdir(parents=True)
    (app_dir / "results_summary.json").write_text(
        json.dumps(
            {
                "panel": {
                    "available_sources": [
                        {
                            "name": "fhfa_hpi",
                            "columns": ["fhfa_hpi"],
                            "non_null_rows": 1,
                            "share": 0.5,
                        },
                        {
                            "name": "ukcpr_welfare",
                            "columns": ["state_eitc_rate"],
                            "non_null_rows": 2,
                            "share": 1.0,
                        },
                    ]
                },
                "estimands": [
                    {
                        "slug": "eitc_violent",
                        "baseline": {
                            "treatment_row": {"coefficient": 2.0, "p_value": 0.4},
                            "pretrend": {"p_value": 0.2, "pass": True},
                        },
                        "dml": {"theta": 3.0, "p_value": 0.01},
                        "overlap": {"max_abs_smd": 0.7},
                        "frontend": {"status": "secondary_method_sensitive"},
                    },
                    {
                        "slug": "tanf_violent",
                        "baseline": {
                            "treatment_row": {"coefficient": 0.1, "p_value": 0.8},
                            "pretrend": None,
                        },
                        "dml": {"theta": 0.0, "p_value": 0.6},
                        "overlap": {"max_abs_smd": 1.2},
                        "frontend": {"status": "exploratory_low_signal"},
                    },
                    {
                        "slug": "snap_bbce_violent",
                        "baseline": {
                            "treatment_row": {"coefficient": 1.0, "p_value": 0.7},
                            "pretrend": {"p_value": 0.001, "pass": False},
                        },
                        "dml": None,
                        "overlap": None,
                        "frontend": {"status": "exploratory_failed_pretrends"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    bidirectional_dir = tmp_path / "outputs" / "exploratory" / "bidirectional_poverty_crime"
    bidirectional_dir.mkdir(parents=True)
    (bidirectional_dir / "bidirectional_summary.json").write_text(
        json.dumps(
            {
                "estimands": [
                    {
                        "title": "Poverty -> Violent Crime",
                        "baseline_fe": {"coefficient": 1.2, "p_value": 0.2},
                        "dml": {"theta": 1.8, "p_value": 0.04},
                        "overlap": {"max_abs_smd": 0.6},
                        "headline": "lag1 FE does not clearly detect an association; DML is statistically strong; placebo lead is clean; longer lags are weak.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    border_dir = tmp_path / "outputs" / "border"
    border_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "label": ["border_min_wage_violent"],
            "spec": ["baseline"],
            "coefficient": [1.0],
            "std_error": [0.5],
            "p_value": [0.2],
            "n_obs_used": [100],
            "n_entities": [20],
        }
    ).to_csv(border_dir / "border_summary.csv", index=False)
    support_trim_dir = tmp_path / "outputs" / "support_trim"
    support_trim_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "label": ["min_wage_violent"],
            "theta_base": [1.0],
            "theta_trimmed": [0.8],
            "p_value_base": [0.04],
            "p_value_trimmed": [0.08],
        }
    ).to_csv(support_trim_dir / "support_trim_summary.csv", index=False)
    min_wage_id_dir = tmp_path / "outputs" / "min_wage_identification"
    min_wage_id_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "outcome_label": ["violent"],
            "spec": ["border_pair_first_difference"],
            "coefficient": [0.9],
            "p_value": [0.04],
            "n_entities": [20],
        }
    ).to_csv(min_wage_id_dir / "min_wage_identification_summary.csv", index=False)
    pd.DataFrame(
        {
            "outcome_label": ["violent"],
            "spec": ["border_pair_first_difference_event_study"],
            "pretrend_p_value": [0.2],
            "pretrend_pass": [True],
        }
    ).to_csv(min_wage_id_dir / "min_wage_event_study_summary.csv", index=False)
    pd.DataFrame(
        {
            "outcome_label": ["violent"],
            "dose_bucket": ["small"],
            "coefficient": [0.3],
            "p_value": [0.6],
        }
    ).to_csv(min_wage_id_dir / "min_wage_dose_bucket_summary.csv", index=False)
    pd.DataFrame(
        {
            "outcome_label": ["violent"],
            "spec": ["border_pair_first_difference_negative_control_treatment"],
            "coefficient": [0.1],
            "p_value": [0.7],
        }
    ).to_csv(min_wage_id_dir / "min_wage_negative_control_treatment_summary.csv", index=False)
    falsification_dir = tmp_path / "outputs" / "falsification"
    falsification_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "label": ["min_wage__median_age"],
            "coefficient": [0.1],
            "p_value": [0.03],
        }
    ).to_csv(falsification_dir / "negative_control_summary.csv", index=False)
    staggered_dir = tmp_path / "outputs" / "staggered"
    staggered_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "label": ["min_wage_violent", "snap_bbce_violent"],
            "pretrend_p_value": [None, 0.38],
            "pretrend_pass": [False, True],
            "n_pre_coefs": [0, 3],
            "interpretable": [False, True],
        }
    ).to_csv(staggered_dir / "staggered_summary.csv", index=False)
    crime_validation_dir = tmp_path / "outputs" / "crime_validation"
    crime_validation_dir.mkdir(parents=True)
    (crime_validation_dir / "crime_measurement_validation.json").write_text(
        json.dumps(
            {
                "external_benchmark_available": False,
                "note": "Coverage-only validation.",
                "robustness_sensitivity": [
                    {"label": "min_wage_violent", "coef_range": 1.1, "sign_flip": False}
                ],
            }
        ),
        encoding="utf-8",
    )
    (crime_validation_dir / "crime_measurement_validation.md").write_text(
        "# Crime validation\n",
        encoding="utf-8",
    )

    out_path = build_final_report(
        panel=panel,
        panel_path=panel_path,
        qa_report_path=qa_path,
        baseline_dir=tmp_path / "outputs" / "baseline",
        dml_dir=tmp_path / "outputs" / "dml",
        robustness_summary_path=robustness_path,
        output_dir=tmp_path / "outputs" / "report",
        overlap_dir=overlap_dir,
        app_dir=app_dir,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "Final Backend Report" in text
    assert "Minimum Wage -> Violent Crime" in text
    assert "Public Data Expansion" in text
    assert "Additional Policy Lanes" in text
    assert "Border-County Design" in text
    assert "Minimum Wage Identification Redesign" in text
    assert "Crime Measurement Validation" in text
    assert "Support-Trimmed DML" in text
    assert "Negative-Control Outcomes" in text
    assert "Staggered-Adoption ATT" in text
    assert "Bidirectional Poverty-Crime Check" in text
    assert "Poverty -> Violent Crime" in text
    assert "State EITC Rate -> Violent Crime" in text
    assert "Robustness" in text
    assert "data_quality_report.md" in text
