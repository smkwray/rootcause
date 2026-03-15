"""Run cross-state border-county differenced FE designs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.data.county_adjacency import CountyAdjacencyAdapter
from povcrime.models.baseline_fe import BaselineFE
from povcrime.models.border import build_border_pair_panel, canonical_cross_state_pairs
from povcrime.models.robustness import build_placebo_treatment, extract_treatment_row
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_ESTIMANDS = [
    {"label": "border_min_wage_violent", "treatment": "effective_min_wage", "outcome": "violent_crime_rate"},
    {"label": "border_min_wage_property", "treatment": "effective_min_wage", "outcome": "property_crime_rate"},
    {"label": "border_eitc_violent", "treatment": "state_eitc_rate", "outcome": "violent_crime_rate"},
    {"label": "border_eitc_property", "treatment": "state_eitc_rate", "outcome": "property_crime_rate"},
]

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


def _fit_spec(
    *,
    pair_panel: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    controls: list[str],
    output_dir: Path,
    spec_name: str,
) -> dict[str, object]:
    cols_needed = ["pair_id", "year", "border_state_pair", outcome_col, treatment_col, *controls]
    sub = pair_panel[cols_needed].dropna(subset=[outcome_col, treatment_col]).copy()
    if len(sub) < 150:
        raise ValueError(f"Only {len(sub)} usable rows remain for {spec_name}.")

    model = BaselineFE(
        df=sub.rename(columns={"border_state_pair": "state_fips"}),
        outcome=outcome_col,
        treatment=treatment_col,
        controls=controls,
        entity_col="pair_id",
        cluster_col="state_fips",
    )
    result = model.fit()
    spec_dir = output_dir / spec_name
    model.save_results(spec_dir)
    coef = extract_treatment_row(model.summary_table(), treatment=treatment_col)
    return {
        "spec": spec_name,
        "sample_rows": int(len(sub)),
        "n_obs_used": int(result.nobs),
        "n_entities": int(result.entity_info["total"]),
        "coefficient": float(coef["coefficient"]),
        "std_error": float(coef["std_error"]),
        "p_value": float(coef["p_value"]),
        "ci_lower": float(coef["ci_lower"]),
        "ci_upper": float(coef["ci_upper"]),
    }


def _write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    summary = pd.DataFrame(rows).sort_values(["label", "spec"]).reset_index(drop=True)
    summary.to_csv(output_dir / "border_summary.csv", index=False)

    lines = [
        "# Border-County Design Summary",
        "",
        "- Scope: cross-state adjacent county pairs only.",
        "- Design: within-pair outcome differences with pair and year fixed effects.",
        "- Treatments: minimum wage and state EITC rate.",
        "",
        "| Label | Spec | Coef | SE | p-value | N used | Pairs |",
        "|-------|------|------|----|---------|--------|-------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['spec']} | {row['coefficient']:.4f} | {row['std_error']:.4f} | "
            f"{row['p_value']:.4f} | {int(row['n_obs_used']):,} | {int(row['n_entities']):,} |"
        )
    (output_dir / "border_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run border-county FE designs.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    parser.add_argument(
        "--placebo-lead",
        type=int,
        default=2,
        help="Years to lead treatment for the placebo border spec.",
    )
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "low_coverage" in panel.columns:
        panel = panel.loc[~panel["low_coverage"]].copy()

    adjacency_adapter = CountyAdjacencyAdapter(config)
    if not (config.raw_dir / "county_adjacency" / "county_adjacency.txt").exists():
        adjacency_adapter.download()
    adjacency = adjacency_adapter.validate(adjacency_adapter.load())
    pairs = canonical_cross_state_pairs(adjacency)

    output_dir = config.output_dir / "border"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for estimand in _ESTIMANDS:
        controls = [col for col in _CONTROLS if col in panel.columns and col not in {estimand["treatment"], estimand["outcome"]}]
        pair_panel = build_border_pair_panel(
            panel=panel,
            pairs=pairs,
            treatment=estimand["treatment"],
            outcome=estimand["outcome"],
            controls=controls,
        )
        if pair_panel.empty:
            logger.warning("Skipping %s: no usable border-pair rows.", estimand["label"])
            continue

        label_dir = output_dir / estimand["label"]
        label_dir.mkdir(parents=True, exist_ok=True)
        treatment_col = f"diff_{estimand['treatment']}"
        outcome_col = f"diff_{estimand['outcome']}"
        control_cols = [f"diff_{col}" for col in controls]

        baseline = _fit_spec(
            pair_panel=pair_panel,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            controls=control_cols,
            output_dir=label_dir,
            spec_name="baseline",
        )
        baseline["label"] = estimand["label"]
        rows.append(baseline)

        placebo_col = f"{treatment_col}_placebo_lead_{args.placebo_lead}"
        placebo_panel = build_placebo_treatment(
            pair_panel,
            treatment_col=treatment_col,
            entity_col="pair_id",
            time_col="year",
            lead_periods=args.placebo_lead,
            output_col=placebo_col,
        )
        placebo = _fit_spec(
            pair_panel=placebo_panel,
            outcome_col=outcome_col,
            treatment_col=placebo_col,
            controls=control_cols,
            output_dir=label_dir,
            spec_name="placebo_lead",
        )
        placebo["label"] = estimand["label"]
        rows.append(placebo)

        logger.info(
            "Border %s baseline coef=%.4f p=%.4f; placebo p=%.4f",
            estimand["label"],
            baseline["coefficient"],
            baseline["p_value"],
            placebo["p_value"],
        )

    if not rows:
        raise SystemExit("No border-county specifications could be estimated.")

    _write_summary(rows, output_dir)


if __name__ == "__main__":
    main()
