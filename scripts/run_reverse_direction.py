"""Run exploratory reverse-direction fixed-effects specifications."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.models.baseline_fe import BaselineFE
from povcrime.models.reverse_direction import lag_treatment_within_county
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_SPECS = [
    {
        "label": "violent_to_poverty",
        "treatment": "violent_crime_rate",
        "outcome": "poverty_rate",
    },
    {
        "label": "property_to_poverty",
        "treatment": "property_crime_rate",
        "outcome": "poverty_rate",
    },
    {
        "label": "violent_to_unemployment",
        "treatment": "violent_crime_rate",
        "outcome": "unemployment_rate",
    },
    {
        "label": "property_to_unemployment",
        "treatment": "property_crime_rate",
        "outcome": "unemployment_rate",
    },
]

_REDUCED_CONTROLS = [
    "log_population",
    "pct_white",
    "pct_black",
    "pct_hispanic",
    "pct_under_18",
    "pct_over_65",
    "pct_hs_or_higher",
    "median_age",
    "effective_min_wage",
    "broad_based_cat_elig",
]

def _write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Reverse-Direction Exploratory Results",
        "",
        "- Scope: exploratory only.",
        "- Design: county and year fixed effects using one-year lagged crime rates as predictors.",
        "- Warning: these are not headline causal estimates and should not be presented as symmetric evidence against the forward design.",
        "",
        "| Label | Outcome | Lagged Treatment | Coef | SE | p-value | N used |",
        "|-------|---------|------------------|------|----|---------|--------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['outcome']} | {row['treatment_variable']} | "
            f"{row['coefficient']:.4f} | {row['std_error']:.4f} | "
            f"{row['p_value']:.4f} | {int(row['n_obs_used']):,} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run exploratory reverse-direction FE specs.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "low_coverage" in panel.columns:
        panel = panel.loc[~panel["low_coverage"]].copy()

    output_dir = config.output_dir / "exploratory" / "reverse_direction"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for spec in _SPECS:
        treatment = spec["treatment"]
        outcome = spec["outcome"]
        label = spec["label"]

        if treatment not in panel.columns or outcome not in panel.columns:
            logger.warning("Skipping %s: missing columns.", label)
            continue

        working = lag_treatment_within_county(panel, treatment_col=treatment)
        lagged_treatment = f"{treatment}_lag1"
        controls = [col for col in _REDUCED_CONTROLS if col in working.columns and col != outcome]
        cols_needed = ["county_fips", "year", "state_fips", outcome, lagged_treatment, *controls]
        sub = working[cols_needed].dropna(subset=[outcome, lagged_treatment]).copy()
        if len(sub) < 200:
            logger.warning("Skipping %s: only %d usable rows.", label, len(sub))
            continue

        spec_dir = output_dir / label
        model = BaselineFE(
            df=sub,
            outcome=outcome,
            treatment=lagged_treatment,
            controls=controls,
        )
        result = model.fit()
        model.save_results(spec_dir)
        coef = model.summary_table().loc[
            lambda df: df["variable"] == lagged_treatment
        ].iloc[0]
        results.append(
            {
                "label": label,
                "outcome": outcome,
                "treatment_variable": lagged_treatment,
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
        logger.info(
            "Reverse-direction %s: coef=%.4f p=%.4f n=%d",
            label,
            results[-1]["coefficient"],
            results[-1]["p_value"],
            results[-1]["n_obs_used"],
        )

    if not results:
        raise SystemExit("No reverse-direction specifications could be estimated.")

    summary = pd.DataFrame(results).sort_values("label").reset_index(drop=True)
    summary.to_csv(output_dir / "reverse_direction_summary.csv", index=False)
    _write_markdown_summary(summary, output_dir / "reverse_direction_summary.md")


if __name__ == "__main__":
    main()
