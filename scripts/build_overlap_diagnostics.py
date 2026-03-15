"""Build overlap/support diagnostics for the ML stage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.models.overlap import build_continuous_treatment_support_diagnostics
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_ESTIMANDS = [
    {
        "treatment": "effective_min_wage",
        "outcome": "violent_crime_rate",
        "label": "min_wage_violent",
    },
    {
        "treatment": "effective_min_wage",
        "outcome": "property_crime_rate",
        "label": "min_wage_property",
    },
    {
        "treatment": "state_eitc_rate",
        "outcome": "violent_crime_rate",
        "label": "eitc_violent",
    },
    {
        "treatment": "state_eitc_rate",
        "outcome": "property_crime_rate",
        "label": "eitc_property",
    },
    {
        "treatment": "tanf_benefit_3_person",
        "outcome": "violent_crime_rate",
        "label": "tanf_violent",
    },
    {
        "treatment": "tanf_benefit_3_person",
        "outcome": "property_crime_rate",
        "label": "tanf_property",
    },
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
    parser = argparse.ArgumentParser(description="Build overlap diagnostics for DML.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "low_coverage" in panel.columns:
        panel = panel.loc[~panel["low_coverage"]].copy()

    output_dir = config.output_dir / "overlap"
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in _ESTIMANDS:
        controls = [col for col in _CONTROLS if col in panel.columns]
        cols_needed = [spec["treatment"], spec["outcome"], *controls]
        sub = panel[cols_needed].dropna().reset_index(drop=True)
        summary = build_continuous_treatment_support_diagnostics(
            df=sub,
            treatment=spec["treatment"],
            controls=controls,
            output_dir=output_dir / spec["label"],
        )
        logger.info(
            "Support diagnostics %s: OOF R2=%.3f, max_abs_smd=%.3f",
            spec["label"],
            summary["oof_r2"],
            summary["max_abs_smd"],
        )


if __name__ == "__main__":
    main()
