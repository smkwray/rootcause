"""Run stacked not-yet-treated event studies for staggered policy adoption."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.analysis import get_analysis_lanes, get_event_definitions
from povcrime.config import get_config
from povcrime.models.policy_events import compute_first_treatment_event_year
from povcrime.models.staggered_att import StaggeredEventStudy
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
    parser = argparse.ArgumentParser(description="Run stacked staggered ATT event studies.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "low_coverage" in panel.columns:
        panel = panel.loc[~panel["low_coverage"]].copy()

    event_defs = get_event_definitions(config=config, method="staggered")
    for event_def in event_defs:
        if event_def.treatment_col in panel.columns and event_def.output_col not in panel.columns:
            panel = compute_first_treatment_event_year(
                panel,
                treatment_col=event_def.treatment_col,
                output_col=event_def.output_col,
                change_threshold=event_def.change_threshold,
            )

    output_dir = config.output_dir / "staggered"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for lane in get_analysis_lanes(config=config, method="staggered"):
        if any(col not in panel.columns for col in [lane.treatment, lane.outcome, lane.event_col]):
            logger.warning("Skipping %s: missing required columns.", lane.slug)
            continue
        controls = [col for col in _CONTROLS if col in panel.columns]
        cols_needed = [
            "county_fips",
            "year",
            "state_fips",
            lane.outcome,
            lane.event_col,
            *controls,
        ]
        sub = panel[cols_needed].dropna(subset=[lane.outcome]).copy()
        spec_dir = output_dir / lane.slug
        try:
            model = StaggeredEventStudy(
                df=sub,
                outcome=lane.outcome,
                event_col=str(lane.event_col),
                controls=controls,
                leads=4,
                lags=6,
            )
            model.fit()
            model.save_results(spec_dir)
            pretrend = model.pretrend_test()
            rows.append(
                {
                    "label": lane.slug,
                    "outcome": lane.outcome,
                    "event_col": lane.event_col,
                    "n_coefs": int(len(model.coef_table())),
                    "n_pre_coefs": int(pretrend["n_pre_coefs"]),
                    "pretrend_p_value": float(pretrend["p_value"]),
                    "pretrend_pass": bool(pretrend["pass"]),
                    "interpretable": bool(pretrend.get("interpretable", False)),
                }
            )
            logger.info(
                "Staggered ATT %s pretrend p=%.4f pass=%s",
                lane.slug,
                pretrend["p_value"],
                pretrend["pass"],
            )
        except Exception:
            logger.exception("Staggered ATT failed for %s.", lane.slug)

    if not rows:
        raise SystemExit("No staggered ATT models were estimated.")

    summary = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    summary.to_csv(output_dir / "staggered_summary.csv", index=False)


if __name__ == "__main__":
    main()
