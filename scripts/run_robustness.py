"""Run robustness checks for the baseline FE estimands."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.analysis import get_analysis_lanes
from povcrime.config import get_config
from povcrime.models.baseline_fe import BaselineFE
from povcrime.models.robustness import (
    build_placebo_treatment,
    detrend_variables_within_entity,
    extract_treatment_row,
)
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CONTROLS = [
    "unemployment_rate",
    "poverty_rate",
    "per_capita_personal_income",
    "cbp_employment_per_capita",
    "cbp_establishments_per_1k",
    "rent_to_income_ratio_2br",
    "log_fhfa_hpi_2000_base",
    "log_population",
    "pct_white",
    "pct_black",
    "pct_hispanic",
    "pct_under_18",
    "pct_over_65",
    "pct_hs_or_higher",
    "median_age",
]


def _build_specs(
    *,
    baseline_threshold: float,
    strict_threshold: float,
    placebo_lead: int,
    include_county_trends: bool,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "name": "baseline_high_coverage",
            "coverage_mode": "low_flag",
            "note": f"Rows with low_coverage=False (roughly source_share >= {baseline_threshold:.2f}).",
        },
        {
            "name": "all_rows",
            "coverage_mode": "none",
            "note": "No coverage filter applied.",
        },
        {
            "name": "strict_coverage",
            "coverage_mode": "threshold",
            "coverage_threshold": strict_threshold,
            "note": f"Rows restricted to source_share >= {strict_threshold:.2f}.",
        },
        {
            "name": "population_weighted",
            "coverage_mode": "low_flag",
            "weight_col": "population_weight",
            "note": "High-coverage sample weighted by county population.",
        },
        {
            "name": "placebo_lead",
            "coverage_mode": "low_flag",
            "placebo_lead": placebo_lead,
            "note": f"Treatment replaced with its {placebo_lead}-year lead.",
        },
    ]
    if include_county_trends:
        specs.append(
            {
                "name": "county_detrended",
                "coverage_mode": "low_flag",
                "detrend": True,
                "note": (
                    "Approximate county-specific linear trends via within-county "
                    "linear detrending of outcome, treatment, and controls."
                ),
            }
        )
    return specs


def _apply_coverage_filter(
    panel: pd.DataFrame,
    *,
    mode: str,
    threshold: float | None = None,
) -> pd.DataFrame:
    if mode == "none":
        return panel.copy()
    if mode == "low_flag" and "low_coverage" in panel.columns:
        return panel.loc[~panel["low_coverage"]].copy()
    if mode == "threshold" and threshold is not None and "source_share" in panel.columns:
        return panel.loc[panel["source_share"] >= threshold].copy()
    return panel.copy()


def _run_spec(
    *,
    panel: pd.DataFrame,
    spec: dict[str, object],
    treatment: str,
    outcome: str,
    label: str,
    controls: list[str],
    output_dir: Path,
    placebo_lead: int,
) -> dict[str, object]:
    working = _apply_coverage_filter(
        panel,
        mode=str(spec.get("coverage_mode", "none")),
        threshold=(
            float(spec["coverage_threshold"])
            if "coverage_threshold" in spec
            else None
        ),
    )

    controls_used = [c for c in controls if c in working.columns]
    treatment_col = treatment
    outcome_col = outcome
    weight_col = None

    if spec.get("weight_col") == "population_weight":
        working["population_weight"] = (
            pd.to_numeric(working["population"], errors="coerce")
            .clip(lower=1.0)
        )
        weight_col = "population_weight"

    if spec.get("placebo_lead"):
        treatment_col = f"{treatment}_placebo_lead_{placebo_lead}"
        working = build_placebo_treatment(
            working,
            treatment_col=treatment,
            entity_col="county_fips",
            time_col="year",
            lead_periods=placebo_lead,
            output_col=treatment_col,
        )

    if spec.get("detrend"):
        cols_to_detrend = [outcome_col, treatment_col, *controls_used]
        working = detrend_variables_within_entity(
            working,
            columns=cols_to_detrend,
            entity_col="county_fips",
            time_col="year",
        )
        outcome_col = f"dt_{outcome_col}"
        treatment_col = f"dt_{treatment_col}"
        controls_used = [f"dt_{col}" for col in controls_used]

    cols_needed = {
        "county_fips",
        "year",
        "state_fips",
        outcome_col,
        treatment_col,
        *controls_used,
    }
    if weight_col is not None:
        cols_needed.add(weight_col)
    sub = working[list(cols_needed)].copy()
    sub = sub.dropna(subset=[outcome_col, treatment_col])

    spec_dir = output_dir / label / str(spec["name"])
    spec_dir.mkdir(parents=True, exist_ok=True)

    if len(sub) < 100:
        raise ValueError(f"Only {len(sub)} usable rows remain for spec '{spec['name']}'.")

    model = BaselineFE(
        df=sub,
        outcome=outcome_col,
        treatment=treatment_col,
        controls=controls_used,
        weight_col=weight_col,
    )
    result = model.fit()
    model.save_results(spec_dir)
    coef = extract_treatment_row(model.summary_table(), treatment=treatment_col)

    return {
        "label": label,
        "outcome": outcome,
        "spec": spec["name"],
        "treatment_variable": treatment_col,
        "sample_rows": int(len(sub)),
        "n_obs_used": int(result.nobs),
        "n_entities": int(result.entity_info["total"]),
        "coefficient": float(coef["coefficient"]),
        "std_error": float(coef["std_error"]),
        "t_stat": float(coef["t_stat"]),
        "p_value": float(coef["p_value"]),
        "ci_lower": float(coef["ci_lower"]),
        "ci_upper": float(coef["ci_upper"]),
        "note": str(spec.get("note", "")),
    }


def _write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Robustness Summary",
        "",
        "| Outcome | Spec | Coef | SE | p-value | N used | Note |",
        "|---------|------|------|----|---------|--------|------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['spec']} | {row['coefficient']:.4f} | "
            f"{row['std_error']:.4f} | {row['p_value']:.4f} | "
            f"{int(row['n_obs_used']):,} | {row['note']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run robustness FE specifications.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=0.75,
        help="Baseline coverage threshold used to define the high-coverage sample.",
    )
    parser.add_argument(
        "--strict-threshold",
        type=float,
        default=0.90,
        help="Strict coverage threshold for the sensitivity sample.",
    )
    parser.add_argument(
        "--placebo-lead",
        type=int,
        default=2,
        help="Number of years to lead treatment for the placebo test.",
    )
    parser.add_argument(
        "--skip-county-trends",
        action="store_true",
        help="Skip the approximate county-trend robustness spec.",
    )
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = (
        Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    )
    panel = pd.read_parquet(panel_path)
    logger.info("Loaded panel: %d rows, %d columns.", len(panel), len(panel.columns))

    output_dir = config.output_dir / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(
        baseline_threshold=args.baseline_threshold,
        strict_threshold=args.strict_threshold,
        placebo_lead=args.placebo_lead,
        include_county_trends=not args.skip_county_trends,
    )

    rows: list[dict[str, object]] = []
    for lane in get_analysis_lanes(config=config, method="robustness"):
        for spec in specs:
            logger.info(
                "Running robustness spec %s for %s.",
                spec["name"],
                lane.slug,
            )
            try:
                row = _run_spec(
                    panel=panel,
                    spec=spec,
                    treatment=lane.treatment,
                    outcome=lane.outcome,
                    label=lane.slug,
                    controls=_CONTROLS,
                    output_dir=output_dir,
                    placebo_lead=args.placebo_lead,
                )
                rows.append(row)
            except Exception as exc:
                logger.exception(
                    "Robustness spec %s failed for %s.",
                    spec["name"],
                    lane.slug,
                )
                rows.append(
                    {
                        "label": lane.slug,
                        "outcome": lane.outcome,
                        "spec": spec["name"],
                        "treatment_variable": lane.treatment,
                        "sample_rows": 0,
                        "n_obs_used": 0,
                        "n_entities": 0,
                        "coefficient": float("nan"),
                        "std_error": float("nan"),
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "ci_lower": float("nan"),
                        "ci_upper": float("nan"),
                        "note": f"FAILED: {exc}",
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["label", "spec"]).reset_index(drop=True)
    summary_path = output_dir / "robustness_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Robustness summary saved to %s.", summary_path)

    markdown_path = output_dir / "robustness_summary.md"
    _write_markdown_summary(summary, markdown_path)
    logger.info("Robustness markdown summary saved to %s.", markdown_path)


if __name__ == "__main__":
    main()
