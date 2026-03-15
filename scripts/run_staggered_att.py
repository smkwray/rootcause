"""Run stacked not-yet-treated event studies for staggered policy adoption."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.models.policy_events import compute_first_treatment_event_year
from povcrime.models.staggered_att import StaggeredEventStudy
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_ESTIMANDS = [
    {
        "label": "min_wage_violent",
        "treatment": "effective_min_wage",
        "outcome": "violent_crime_rate",
        "event_col": "min_wage_event_year",
    },
    {
        "label": "min_wage_property",
        "treatment": "effective_min_wage",
        "outcome": "property_crime_rate",
        "event_col": "min_wage_event_year",
    },
    {
        "label": "snap_bbce_violent",
        "treatment": "broad_based_cat_elig",
        "outcome": "violent_crime_rate",
        "event_col": "snap_bbce_event_year",
    },
    {
        "label": "snap_bbce_property",
        "treatment": "broad_based_cat_elig",
        "outcome": "property_crime_rate",
        "event_col": "snap_bbce_event_year",
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
    parser = argparse.ArgumentParser(description="Run stacked staggered ATT event studies.")
    parser.add_argument("--panel", type=str, default=None, help="Path to panel parquet.")
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    panel_path = Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    if "low_coverage" in panel.columns:
        panel = panel.loc[~panel["low_coverage"]].copy()

    event_defs = [
        {"treatment_col": "effective_min_wage", "output_col": "min_wage_event_year", "change_threshold": 0.01},
        {"treatment_col": "broad_based_cat_elig", "output_col": "snap_bbce_event_year", "change_threshold": 0.5},
    ]
    for event_def in event_defs:
        if event_def["treatment_col"] in panel.columns and event_def["output_col"] not in panel.columns:
            panel = compute_first_treatment_event_year(panel, **event_def)

    output_dir = config.output_dir / "staggered"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for estimand in _ESTIMANDS:
        if any(col not in panel.columns for col in [estimand["treatment"], estimand["outcome"], estimand["event_col"]]):
            logger.warning("Skipping %s: missing required columns.", estimand["label"])
            continue
        controls = [col for col in _CONTROLS if col in panel.columns]
        cols_needed = [
            "county_fips",
            "year",
            "state_fips",
            estimand["outcome"],
            estimand["event_col"],
            *controls,
        ]
        sub = panel[cols_needed].dropna(subset=[estimand["outcome"]]).copy()
        spec_dir = output_dir / estimand["label"]
        try:
            model = StaggeredEventStudy(
                df=sub,
                outcome=estimand["outcome"],
                event_col=estimand["event_col"],
                controls=controls,
                leads=4,
                lags=6,
            )
            model.fit()
            model.save_results(spec_dir)
            pretrend = model.pretrend_test()
            rows.append(
                {
                    "label": estimand["label"],
                    "outcome": estimand["outcome"],
                    "event_col": estimand["event_col"],
                    "n_coefs": int(len(model.coef_table())),
                    "n_pre_coefs": int(pretrend["n_pre_coefs"]),
                    "pretrend_p_value": float(pretrend["p_value"]),
                    "pretrend_pass": bool(pretrend["pass"]),
                    "interpretable": bool(pretrend.get("interpretable", False)),
                }
            )
            logger.info(
                "Staggered ATT %s pretrend p=%.4f pass=%s",
                estimand["label"],
                pretrend["p_value"],
                pretrend["pass"],
            )
        except Exception:
            logger.exception("Staggered ATT failed for %s.", estimand["label"])

    if not rows:
        raise SystemExit("No staggered ATT models were estimated.")

    summary = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    summary.to_csv(output_dir / "staggered_summary.csv", index=False)


if __name__ == "__main__":
    main()
