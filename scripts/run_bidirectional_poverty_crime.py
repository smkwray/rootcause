"""Run symmetric exploratory poverty<->crime specifications."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from povcrime.analysis import get_bidirectional_lanes
from povcrime.config import get_config
from povcrime.models.baseline_fe import BaselineFE
from povcrime.models.dml import DMLEstimator
from povcrime.models.overlap import build_continuous_treatment_support_diagnostics
from povcrime.models.reverse_direction import (
    lag_treatment_within_county,
    lead_treatment_within_county,
)
from povcrime.models.robustness import detrend_variables_within_entity, extract_treatment_row
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CONTROLS = [
    "unemployment_rate",
    "per_capita_personal_income",
    "cbp_employment_per_capita",
    "cbp_establishments_per_1k",
    "rent_to_income_ratio_2br",
    "log_fhfa_hpi_2000_base",
    "effective_min_wage",
    "broad_based_cat_elig",
    "state_eitc_rate",
    "tanf_benefit_3_person",
    "log_population",
    "pct_white",
    "pct_black",
    "pct_hispanic",
    "pct_under_18",
    "pct_over_65",
    "pct_hs_or_higher",
    "median_age",
]

_ROBUSTNESS_SPECS = [
    {"name": "lag1_high_coverage", "lag": 1, "coverage": "high"},
    {"name": "lag2_high_coverage", "lag": 2, "coverage": "high"},
    {"name": "lag3_high_coverage", "lag": 3, "coverage": "high"},
    {"name": "all_rows_lag1", "lag": 1, "coverage": "all"},
    {"name": "strict_coverage_lag1", "lag": 1, "coverage": "strict"},
    {"name": "placebo_lead1_high_coverage", "lead": 1, "coverage": "high"},
    {"name": "county_detrended_lag1", "lag": 1, "coverage": "high", "detrend": True},
]


def _available_controls(panel: pd.DataFrame, *, treatment: str, outcome: str) -> list[str]:
    excluded = {treatment, outcome}
    return [col for col in _CONTROLS if col in panel.columns and col not in excluded]


def _coverage_filter(panel: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    if mode == "all":
        return panel.copy()
    if mode == "strict" and "source_share" in panel.columns:
        return panel.loc[panel["source_share"] >= 0.90].copy()
    if "low_coverage" in panel.columns:
        return panel.loc[~panel["low_coverage"]].copy()
    return panel.copy()


def _build_shifted_sample(
    panel: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    controls: list[str],
    lag: int | None = None,
    lead: int | None = None,
    detrend: bool = False,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    working = panel.copy()
    treatment_col = treatment
    outcome_col = outcome
    controls_used = list(controls)

    if lag is not None:
        working = lag_treatment_within_county(working, treatment_col=treatment, periods=lag)
        treatment_col = f"{treatment}_lag{lag}"
    elif lead is not None:
        working = lead_treatment_within_county(working, treatment_col=treatment, periods=lead)
        treatment_col = f"{treatment}_lead{lead}"

    if detrend:
        working = detrend_variables_within_entity(
            working,
            columns=[outcome_col, treatment_col, *controls_used],
            entity_col="county_fips",
            time_col="year",
        )
        outcome_col = f"dt_{outcome_col}"
        treatment_col = f"dt_{treatment_col}"
        controls_used = [f"dt_{col}" for col in controls_used]

    cols_needed = ["county_fips", "year", "state_fips", outcome_col, treatment_col, *controls_used]
    sub = working[cols_needed].dropna(subset=[outcome_col, treatment_col]).copy()
    return sub, treatment_col, outcome_col, controls_used


def _run_baseline_fe(
    *,
    panel: pd.DataFrame,
    lane,
    controls: list[str],
    output_dir: Path,
) -> dict[str, object]:
    sub, treatment_col, outcome_col, controls_used = _build_shifted_sample(
        panel,
        treatment=lane.treatment,
        outcome=lane.outcome,
        controls=controls,
        lag=1,
    )
    if len(sub) < 200:
        raise ValueError(f"Only {len(sub)} usable rows for baseline {lane.slug}.")

    model = BaselineFE(
        df=sub,
        outcome=outcome_col,
        treatment=treatment_col,
        controls=controls_used,
    )
    result = model.fit()
    spec_dir = output_dir / "baseline" / lane.slug
    model.save_results(spec_dir)
    coef = extract_treatment_row(model.summary_table(), treatment=treatment_col)
    return {
        "treatment_variable": treatment_col,
        "sample_rows": int(len(sub)),
        "n_obs_used": int(result.nobs),
        "n_entities": int(result.entity_info["total"]),
        "coefficient": float(coef["coefficient"]),
        "std_error": float(coef["std_error"]),
        "p_value": float(coef["p_value"]),
        "ci_lower": float(coef["ci_lower"]),
        "ci_upper": float(coef["ci_upper"]),
    }


def _run_robustness(
    *,
    panel: pd.DataFrame,
    lane,
    controls: list[str],
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for robustness in _ROBUSTNESS_SPECS:
        filtered = _coverage_filter(panel, mode=str(robustness["coverage"]))
        sub, treatment_col, outcome_col, controls_used = _build_shifted_sample(
            filtered,
            treatment=lane.treatment,
            outcome=lane.outcome,
            controls=controls,
            lag=robustness.get("lag"),
            lead=robustness.get("lead"),
            detrend=bool(robustness.get("detrend", False)),
        )
        if len(sub) < 150:
            logger.warning(
                "Skipping bidirectional robustness %s for %s: only %d rows.",
                robustness["name"],
                lane.slug,
                len(sub),
            )
            continue
        model = BaselineFE(
            df=sub,
            outcome=outcome_col,
            treatment=treatment_col,
            controls=controls_used,
        )
        result = model.fit()
        spec_dir = output_dir / "robustness" / lane.slug / str(robustness["name"])
        model.save_results(spec_dir)
        coef = extract_treatment_row(model.summary_table(), treatment=treatment_col)
        rows.append(
            {
                "label": lane.slug,
                "title": lane.title,
                "spec": robustness["name"],
                "treatment_variable": treatment_col,
                "sample_rows": int(len(sub)),
                "n_obs_used": int(result.nobs),
                "n_entities": int(result.entity_info["total"]),
                "coefficient": float(coef["coefficient"]),
                "std_error": float(coef["std_error"]),
                "p_value": float(coef["p_value"]),
                "ci_lower": float(coef["ci_lower"]),
                "ci_upper": float(coef["ci_upper"]),
            }
        )
    return rows


def _run_dml_and_overlap(
    *,
    panel: pd.DataFrame,
    lane,
    controls: list[str],
    output_dir: Path,
    n_folds: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    sub, treatment_col, outcome_col, controls_used = _build_shifted_sample(
        panel,
        treatment=lane.treatment,
        outcome=lane.outcome,
        controls=controls,
        lag=1,
    )
    dml_summary: dict[str, object] | None = None
    overlap_summary: dict[str, object] | None = None
    if len(sub) < 200:
        logger.warning("Skipping bidirectional ML for %s: only %d rows.", lane.slug, len(sub))
        return dml_summary, overlap_summary

    dml_dir = output_dir / "dml" / lane.slug
    dml_dir.mkdir(parents=True, exist_ok=True)
    dml = DMLEstimator(
        df=sub,
        outcome=outcome_col,
        treatment=treatment_col,
        controls=controls_used,
        group_col="county_fips",
        panel_mode="two_way_within",
        entity_col="county_fips",
        time_col="year",
        n_folds=n_folds,
    )
    dml.fit()
    dml.save_results(dml_dir)
    dml_summary = dml.summary()

    overlap_dir = output_dir / "overlap" / lane.slug
    overlap_summary = build_continuous_treatment_support_diagnostics(
        df=sub,
        treatment=treatment_col,
        controls=controls_used,
        output_dir=overlap_dir,
        group_col="county_fips",
        panel_mode="two_way_within",
        entity_col="county_fips",
        time_col="year",
    )
    return dml_summary, overlap_summary


def _headline(fe: dict[str, object], dml: dict[str, object] | None, robustness: list[dict[str, object]]) -> str:
    placebo = next((row for row in robustness if row["spec"] == "placebo_lead1_high_coverage"), None)
    lag2 = next((row for row in robustness if row["spec"] == "lag2_high_coverage"), None)
    lag3 = next((row for row in robustness if row["spec"] == "lag3_high_coverage"), None)
    parts: list[str] = []
    parts.append(
        f"lag1 FE {'detects' if fe['p_value'] < 0.05 else 'does not clearly detect'} an association"
    )
    if dml is not None:
        parts.append(
            f"DML is {'statistically strong' if dml['p_value'] < 0.05 else 'not statistically strong'}"
        )
    if placebo is not None:
        parts.append(
            f"placebo lead is {'clean' if placebo['p_value'] >= 0.05 else 'concerning'}"
        )
    lag_support = [
        row for row in (lag2, lag3) if row is not None and row["p_value"] < 0.05
    ]
    parts.append(
        "longer lags persist" if lag_support else "longer lags are weak"
    )
    return "; ".join(parts) + "."


def _write_outputs(
    *,
    rows: list[dict[str, object]],
    summary_json: dict[str, object],
    output_dir: Path,
) -> None:
    flat_rows: list[dict[str, object]] = []
    for row in rows:
        flat_rows.append(
            {
                "label": row["label"],
                "title": row["title"],
                "fe_coefficient": row["baseline_fe"]["coefficient"],
                "fe_p_value": row["baseline_fe"]["p_value"],
                "dml_theta": None if row["dml"] is None else row["dml"]["theta"],
                "dml_p_value": None if row["dml"] is None else row["dml"]["p_value"],
                "max_abs_smd": None if row["overlap"] is None else row["overlap"]["max_abs_smd"],
                "headline": row["headline"],
            }
        )
    pd.DataFrame(flat_rows).to_csv(output_dir / "bidirectional_summary.csv", index=False)
    (output_dir / "bidirectional_summary.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Bidirectional Poverty-Crime Summary",
        "",
        "- Scope: exploratory but designed to compare `poverty -> crime` and `crime -> poverty` on more equal footing.",
        "- Design: county and year fixed effects, one-year lagged treatments, shared control set, FE robustness checks, county-grouped cross-fitting, two-way within residualization for ML steps, and continuous-treatment support diagnostics.",
        "",
        "| Lane | FE Coef | FE p-value | DML Theta | DML p-value | Max Abs SMD | Headline |",
        "|------|---------|------------|-----------|-------------|-------------|----------|",
    ]
    for row in rows:
        dml = row["dml"] or {}
        overlap = row["overlap"] or {}
        lines.append(
            f"| {row['title']} | {row['baseline_fe']['coefficient']:.4f} | {row['baseline_fe']['p_value']:.4f} | "
            f"{_fmt(dml.get('theta'))} | {_fmt(dml.get('p_value'))} | {_fmt(overlap.get('max_abs_smd'))} | {row['headline']} |"
        )
    lines.extend(
        [
            "",
            "- Robustness details live under the per-lane directories in this folder.",
            "",
        ]
    )
    (output_dir / "bidirectional_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run symmetric poverty<->crime exploratory specifications.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    parser.add_argument("--n-folds", type=int, default=3, help="DML cross-fitting folds.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    output_dir = config.output_dir / "exploratory" / "bidirectional_poverty_crime"
    output_dir.mkdir(parents=True, exist_ok=True)

    high_coverage_panel = _coverage_filter(panel, mode="high")
    rows: list[dict[str, object]] = []
    for lane in get_bidirectional_lanes(config=config):
        if lane.treatment not in panel.columns or lane.outcome not in panel.columns:
            logger.warning("Skipping %s: missing columns.", lane.slug)
            continue
        controls = _available_controls(panel, treatment=lane.treatment, outcome=lane.outcome)
        baseline_fe = _run_baseline_fe(
            panel=high_coverage_panel,
            lane=lane,
            controls=controls,
            output_dir=output_dir,
        )
        robustness = _run_robustness(
            panel=panel,
            lane=lane,
            controls=controls,
            output_dir=output_dir,
        )
        dml_summary, overlap_summary = _run_dml_and_overlap(
            panel=high_coverage_panel,
            lane=lane,
            controls=controls,
            output_dir=output_dir,
            n_folds=args.n_folds,
        )
        rows.append(
            {
                "label": lane.slug,
                "title": lane.title,
                "treatment": lane.treatment,
                "outcome": lane.outcome,
                "baseline_fe": baseline_fe,
                "dml": dml_summary,
                "overlap": overlap_summary,
                "robustness": robustness,
                "headline": _headline(baseline_fe, dml_summary, robustness),
            }
        )
        logger.info(
            "Bidirectional %s: FE coef=%.4f p=%.4f; DML p=%s",
            lane.slug,
            baseline_fe["coefficient"],
            baseline_fe["p_value"],
            "n/a" if dml_summary is None else f"{dml_summary['p_value']:.4f}",
        )

    if not rows:
        raise SystemExit("No bidirectional poverty-crime specifications were estimated.")

    summary_json = {
        "generated_date": pd.Timestamp.today().date().isoformat(),
        "design": {
            "baseline_timing": "one-year lagged treatment",
            "robustness_specs": [spec["name"] for spec in _ROBUSTNESS_SPECS],
            "coverage_rule": "baseline FE, DML, and overlap use high-coverage rows; robustness also includes all-rows and strict-coverage samples",
            "dml_panel_mode": "two_way_within",
            "cross_fitting": "county_grouped",
        },
        "estimands": rows,
    }
    _write_outputs(rows=rows, summary_json=summary_json, output_dir=output_dir)


if __name__ == "__main__":
    main()
