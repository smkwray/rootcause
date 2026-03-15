"""Run negative-control outcome falsification regressions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.models.baseline_fe import BaselineFE
from povcrime.models.robustness import extract_treatment_row
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_TREATMENTS = [
    {"label": "min_wage", "treatment": "effective_min_wage"},
    {"label": "eitc", "treatment": "state_eitc_rate"},
]

_NEGATIVE_OUTCOMES = [
    "pct_over_65",
    "median_age",
    "pct_hs_or_higher",
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run negative-control outcome falsification FE regressions.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)
    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "low_coverage" in panel.columns:
        panel = panel.loc[~panel["low_coverage"]].copy()

    output_dir = config.output_dir / "falsification"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for treatment_spec in _TREATMENTS:
        for outcome in _NEGATIVE_OUTCOMES:
            if treatment_spec["treatment"] not in panel.columns or outcome not in panel.columns:
                continue
            controls = [
                col for col in _CONTROLS
                if col in panel.columns and col not in {treatment_spec["treatment"], outcome}
            ]
            cols_needed = ["county_fips", "year", "state_fips", treatment_spec["treatment"], outcome, *controls]
            sub = panel[cols_needed].dropna(subset=[treatment_spec["treatment"], outcome]).copy()
            if len(sub) < 200:
                continue
            label = f"{treatment_spec['label']}__{outcome}"
            model = BaselineFE(
                df=sub,
                outcome=outcome,
                treatment=treatment_spec["treatment"],
                controls=controls,
            )
            result = model.fit()
            spec_dir = output_dir / label
            model.save_results(spec_dir)
            coef = extract_treatment_row(model.summary_table(), treatment=treatment_spec["treatment"])
            rows.append(
                {
                    "label": label,
                    "treatment": treatment_spec["treatment"],
                    "outcome": outcome,
                    "n_obs_used": int(result.nobs),
                    "n_entities": int(result.entity_info["total"]),
                    "coefficient": float(coef["coefficient"]),
                    "std_error": float(coef["std_error"]),
                    "p_value": float(coef["p_value"]),
                }
            )
            logger.info("Falsification %s p=%.4f", label, coef["p_value"])

    if not rows:
        raise SystemExit("No falsification regressions were estimated.")

    summary = pd.DataFrame(rows).sort_values(["treatment", "outcome"]).reset_index(drop=True)
    summary.to_csv(output_dir / "negative_control_summary.csv", index=False)
    lines = [
        "# Negative-Control Outcome Summary",
        "",
        "- Scope: demographic outcomes that should move slowly and are not the main crime endpoints.",
        "",
        "| Label | Coef | SE | p-value | N used |",
        "|-------|------|----|---------|--------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['coefficient']:.4f} | {row['std_error']:.4f} | {row['p_value']:.4f} | {int(row['n_obs_used']):,} |"
        )
    (output_dir / "negative_control_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
