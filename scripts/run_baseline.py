"""Run baseline (TWFE / event-study) models on the analysis panel.

Loads the county-year panel, filters to high-coverage rows, and runs:
1. Two-way fixed effects (BaselineFE) for each treatment-outcome pair.
2. Event-study regressions if an event-timing column is available.

Results are saved to ``outputs/baseline/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.config import get_config
from povcrime.models.baseline_fe import BaselineFE
from povcrime.models.event_study import EventStudy
from povcrime.models.policy_events import compute_first_treatment_event_year
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Treatment-outcome pairs to estimate.
_ESTIMANDS = [
    {
        "treatment": "effective_min_wage",
        "outcome": "violent_crime_rate",
        "label": "min_wage_violent",
        "event_col": "min_wage_event_year",
    },
    {
        "treatment": "effective_min_wage",
        "outcome": "property_crime_rate",
        "label": "min_wage_property",
        "event_col": "min_wage_event_year",
    },
    {
        "treatment": "broad_based_cat_elig",
        "outcome": "violent_crime_rate",
        "label": "snap_bbce_violent",
        "event_col": "snap_bbce_event_year",
    },
    {
        "treatment": "broad_based_cat_elig",
        "outcome": "property_crime_rate",
        "label": "snap_bbce_property",
        "event_col": "snap_bbce_event_year",
    },
    {
        "treatment": "state_eitc_rate",
        "outcome": "violent_crime_rate",
        "label": "eitc_violent",
        "event_col": "eitc_event_year",
    },
    {
        "treatment": "state_eitc_rate",
        "outcome": "property_crime_rate",
        "label": "eitc_property",
        "event_col": "eitc_event_year",
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

# Core control variables.
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
    """Entry point for the baseline-models script."""
    parser = argparse.ArgumentParser(
        description="Run baseline TWFE and event-study models.",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help="Path to the panel parquet file (default: data/processed/panel.parquet).",
    )
    parser.add_argument(
        "--high-coverage-only",
        action="store_true",
        default=True,
        help="Filter to high-coverage rows only (default: True).",
    )
    parser.add_argument(
        "--skip-event-study",
        action="store_true",
        default=False,
        help="Skip event-study estimation.",
    )
    args = parser.parse_args(argv)

    config = get_config()
    ensure_dirs(config)

    # ---- Load panel ------------------------------------------------- #
    panel_path = (
        Path(args.panel) if args.panel else config.processed_dir / "panel.parquet"
    )
    if not panel_path.exists():
        logger.error(
            "Panel not found at %s. Run build_panel.py first.", panel_path
        )
        raise SystemExit(1)

    panel = pd.read_parquet(panel_path)
    logger.info(
        "Loaded panel: %d rows, %d columns.", len(panel), len(panel.columns)
    )

    # ---- Filter to high-coverage rows ------------------------------- #
    if args.high_coverage_only and "low_coverage" in panel.columns:
        n_before = len(panel)
        panel = panel[~panel["low_coverage"]].copy()
        logger.info(
            "Filtered to high-coverage: %d -> %d rows.", n_before, len(panel)
        )

    # ---- Compute event timing if not already present ---------------- #
    event_defs = [
        {
            "treatment_col": "effective_min_wage",
            "output_col": "min_wage_event_year",
            "change_threshold": 0.01,
        },
        {
            "treatment_col": "broad_based_cat_elig",
            "output_col": "snap_bbce_event_year",
            "change_threshold": 0.5,
        },
        {
            "treatment_col": "state_eitc_rate",
            "output_col": "eitc_event_year",
            "change_threshold": 0.01,
        },
    ]
    for event_def in event_defs:
        if (
            event_def["treatment_col"] in panel.columns
            and event_def["output_col"] not in panel.columns
        ):
            panel = compute_first_treatment_event_year(panel, **event_def)

    # ---- Output directory ------------------------------------------- #
    output_dir = config.output_dir / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    controls_avail = [c for c in _CONTROLS if c in panel.columns]

    # ---- TWFE estimation -------------------------------------------- #
    for spec in _ESTIMANDS:
        treatment = spec["treatment"]
        outcome = spec["outcome"]
        label = spec["label"]
        event_col = spec.get("event_col")

        if treatment not in panel.columns or outcome not in panel.columns:
            logger.warning(
                "Skipping %s: missing %s or %s in panel.",
                label,
                treatment,
                outcome,
            )
            continue

        # Subset to needed columns with non-null treatment and outcome
        cols_needed = list(
            {treatment, outcome, "county_fips", "year", "state_fips"}
            | set(controls_avail)
        )
        if event_col and event_col in panel.columns:
            cols_needed.append(event_col)
        cols_needed = [c for c in cols_needed if c in panel.columns]
        sub = panel[cols_needed].dropna(subset=[treatment, outcome]).copy()

        if len(sub) < 100:
            logger.warning(
                "Skipping %s: only %d usable rows.", label, len(sub)
            )
            continue

        # --- BaselineFE --- #
        logger.info(
            "Running BaselineFE: %s -> %s (%d rows, %d controls).",
            treatment,
            outcome,
            len(sub),
            len(controls_avail),
        )

        spec_dir = output_dir / label
        try:
            fe_model = BaselineFE(
                df=sub,
                outcome=outcome,
                treatment=treatment,
                controls=controls_avail,
            )
            fe_model.fit()
            fe_model.save_results(spec_dir)
            logger.info("BaselineFE results saved to %s.", spec_dir)
        except Exception:
            logger.exception("BaselineFE failed for %s.", label)

        # --- EventStudy --- #
        if (
            not args.skip_event_study
            and event_col
            and event_col in sub.columns
            and sub[event_col].notna().sum() > 0
        ):
            logger.info(
                "Running EventStudy: %s -> %s (event_col=%s).",
                treatment,
                outcome,
                event_col,
            )
            try:
                es_model = EventStudy(
                    df=sub,
                    outcome=outcome,
                    event_col=event_col,
                    controls=controls_avail,
                    leads=4,
                    lags=6,
                )
                es_model.fit()
                es_model.save_results(spec_dir)
                logger.info(
                    "EventStudy results saved to %s.", spec_dir
                )

                # Log pre-trend test result
                pretrend = es_model.pretrend_test()
                status = "PASS" if pretrend["pass"] else "FAIL"
                logger.info(
                    "Pre-trend test for %s: F=%.3f, p=%.4f -> %s.",
                    label,
                    pretrend["f_stat"],
                    pretrend["p_value"],
                    status,
                )
            except Exception:
                logger.exception("EventStudy failed for %s.", label)
        else:
            logger.info(
                "Skipping EventStudy for %s (no event timing column or "
                "--skip-event-study).",
                label,
            )

    logger.info("Baseline models complete. Results in %s.", output_dir)


if __name__ == "__main__":
    main()
