"""Smoke test for the bidirectional exploratory runner."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from povcrime.config import load_project_config
from povcrime.reports.contracts import load_bidirectional_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_bidirectional_poverty_crime.py"
SPEC = importlib.util.spec_from_file_location("run_bidirectional_poverty_crime", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
run_bidirectional_poverty_crime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_bidirectional_poverty_crime)


def test_bidirectional_runner_writes_canonical_summary(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "bidirectional_project"
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "configs" / "project.yaml", project_root / "configs" / "project.yaml")

    config = load_project_config(project_root)
    panel_path = config.processed_dir / "panel.parquet"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    _make_bidirectional_panel().to_parquet(panel_path, index=False)

    monkeypatch.setattr(run_bidirectional_poverty_crime, "get_config", lambda: config)

    run_bidirectional_poverty_crime.main(["--panel", str(panel_path), "--n-folds", "3"])

    summary_path = (
        config.output_dir
        / "exploratory"
        / "bidirectional_poverty_crime"
        / "bidirectional_summary.json"
    )
    summary = load_bidirectional_summary(summary_path)

    assert summary["design"]["dml_panel_mode"] == "two_way_within"
    assert summary["design"]["cross_fitting"] == "county_grouped"
    assert {row["label"] for row in summary["estimands"]} == {
        "poverty_to_violent",
        "poverty_to_property",
        "violent_to_poverty",
        "property_to_poverty",
    }
    violent_to_poverty = next(
        row for row in summary["estimands"] if row["label"] == "violent_to_poverty"
    )
    assert violent_to_poverty["title"] == "Violent Crime -> Poverty"
    assert violent_to_poverty["treatment"] == "violent_crime_rate"
    assert violent_to_poverty["outcome"] == "poverty_rate"


def _make_bidirectional_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int | str | bool]] = []
    years = list(range(2010, 2016))

    for county_idx in range(50):
        county_fips = f"{1 + county_idx // 10:02d}{county_idx + 1:03d}"
        state_fips = f"{1 + county_idx // 10:02d}"
        county_effect = rng.normal(scale=1.5)
        for year in years:
            year_offset = year - years[0]
            poverty_rate = 12.0 + county_effect + 0.15 * year_offset + rng.normal(scale=0.6)
            unemployment_rate = 4.5 + 0.08 * year_offset + rng.normal(scale=0.3)
            income = 42000 + 350 * year_offset + rng.normal(scale=1200)
            log_population = 10.2 + 0.01 * year_offset + rng.normal(scale=0.05)
            pct_black = 0.18 + 0.01 * county_effect + rng.normal(scale=0.01)
            pct_white = 0.64 - 0.01 * county_effect + rng.normal(scale=0.01)
            pct_hispanic = 0.12 + rng.normal(scale=0.01)
            pct_under_18 = 0.23 + rng.normal(scale=0.01)
            pct_over_65 = 0.15 + rng.normal(scale=0.01)
            pct_hs_or_higher = 0.88 + rng.normal(scale=0.01)
            median_age = 38 + 0.12 * year_offset + rng.normal(scale=0.4)
            effective_min_wage = 7.25 + 0.05 * year_offset + 0.1 * (county_idx % 3)
            broad_based_cat_elig = float((county_idx + year) % 2 == 0)
            state_eitc_rate = 0.08 + 0.005 * (county_idx // 10) + 0.002 * year_offset
            tanf_benefit = 350 + 8 * (county_idx // 10) + 5 * year_offset
            cbp_emp_pc = 0.24 + rng.normal(scale=0.015)
            cbp_est_1k = 19 + rng.normal(scale=0.8)
            rent_to_income = 0.24 + rng.normal(scale=0.015)
            log_hpi = 4.7 + 0.02 * year_offset + rng.normal(scale=0.03)

            violent_crime_rate = (
                180
                + 4.0 * poverty_rate
                + 1.5 * unemployment_rate
                - 0.0012 * income
                + rng.normal(scale=8.0)
            )
            property_crime_rate = (
                1400
                + 12.0 * poverty_rate
                + 3.0 * unemployment_rate
                - 0.006 * income
                + rng.normal(scale=30.0)
            )

            rows.append(
                {
                    "county_fips": county_fips,
                    "state_fips": state_fips,
                    "year": year,
                    "poverty_rate": float(poverty_rate),
                    "violent_crime_rate": float(violent_crime_rate),
                    "property_crime_rate": float(property_crime_rate),
                    "unemployment_rate": float(unemployment_rate),
                    "per_capita_personal_income": float(income),
                    "cbp_employment_per_capita": float(cbp_emp_pc),
                    "cbp_establishments_per_1k": float(cbp_est_1k),
                    "rent_to_income_ratio_2br": float(rent_to_income),
                    "log_fhfa_hpi_2000_base": float(log_hpi),
                    "effective_min_wage": float(effective_min_wage),
                    "broad_based_cat_elig": float(broad_based_cat_elig),
                    "state_eitc_rate": float(state_eitc_rate),
                    "tanf_benefit_3_person": float(tanf_benefit),
                    "log_population": float(log_population),
                    "pct_white": float(pct_white),
                    "pct_black": float(pct_black),
                    "pct_hispanic": float(pct_hispanic),
                    "pct_under_18": float(pct_under_18),
                    "pct_over_65": float(pct_over_65),
                    "pct_hs_or_higher": float(pct_hs_or_higher),
                    "median_age": float(median_age),
                    "source_share": 1.0,
                    "low_coverage": False,
                }
            )

    return pd.DataFrame(rows)
