"""Build overlap/support diagnostics for the ML stage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.analysis import get_analysis_lanes
from povcrime.config import get_config
from povcrime.models.overlap import build_continuous_treatment_support_diagnostics
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

    for lane in get_analysis_lanes(config=config, method="overlap"):
        controls = [col for col in _CONTROLS if col in panel.columns]
        cols_needed = ["county_fips", "year", lane.treatment, lane.outcome, *controls]
        sub = panel[cols_needed].dropna().reset_index(drop=True)
        summary = build_continuous_treatment_support_diagnostics(
            df=sub,
            treatment=lane.treatment,
            controls=controls,
            output_dir=output_dir / lane.slug,
            group_col="county_fips",
            panel_mode="two_way_within",
            entity_col="county_fips",
            time_col="year",
        )
        logger.info(
            "Support diagnostics %s: OOF R2=%.3f, max_abs_smd=%.3f",
            lane.slug,
            summary["oof_r2"],
            summary["max_abs_smd"],
        )


if __name__ == "__main__":
    main()
