"""Run double/debiased ML and causal forest models on the analysis panel.

Loads the county-year panel, filters to high-coverage rows, and runs:
1. DMLEstimator (Partially Linear Regression) for each treatment-outcome pair.
2. CausalForestEstimator for heterogeneous treatment effects.

Results are saved to ``outputs/dml/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from povcrime.analysis import get_analysis_lanes
from povcrime.config import get_config
from povcrime.models.causal_forest import CausalForestEstimator
from povcrime.models.dml import DMLEstimator
from povcrime.utils import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Controls for DML nuisance models.
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

# Variables that may drive treatment effect heterogeneity.
_EFFECT_MODIFIERS = [
    "poverty_rate",
    "unemployment_rate",
    "rent_to_income_ratio_2br",
    "log_fhfa_hpi_2000_base",
    "log_population",
    "pct_black",
    "pct_hispanic",
    "median_age",
]


def main(argv: list[str] | None = None) -> None:
    """Entry point for the DML-models script."""
    parser = argparse.ArgumentParser(
        description="Run DML and causal forest models.",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help="Path to the panel parquet file.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of cross-fitting folds for DML (default: 3).",
    )
    parser.add_argument(
        "--skip-causal-forest",
        action="store_true",
        help="Skip causal forest estimation (faster).",
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
    if "low_coverage" in panel.columns:
        n_before = len(panel)
        panel = panel[~panel["low_coverage"]].copy()
        logger.info(
            "Filtered to high-coverage: %d -> %d rows.", n_before, len(panel)
        )

    # ---- Output directory ------------------------------------------- #
    output_dir = config.output_dir / "dml"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run models for each treatment-outcome pair ----------------- #
    for lane in get_analysis_lanes(config=config, method="dml"):
        treatment = lane.treatment
        outcome = lane.outcome
        label = lane.slug

        if treatment not in panel.columns or outcome not in panel.columns:
            logger.warning("Skipping %s: missing columns.", label)
            continue

        controls_avail = [c for c in _CONTROLS if c in panel.columns]
        cols_needed = ["county_fips", "year", outcome, treatment, *controls_avail]
        sub = panel[cols_needed].dropna().reset_index(drop=True)

        if len(sub) < 200:
            logger.warning(
                "Skipping %s: only %d usable rows.", label, len(sub)
            )
            continue

        spec_dir = output_dir / label
        spec_dir.mkdir(parents=True, exist_ok=True)

        # --- DML (Partially Linear Regression) --- #
        logger.info(
            "Running DML: %s -> %s (%d rows, %d controls).",
            treatment,
            outcome,
            len(sub),
            len(controls_avail),
        )
        try:
            dml = DMLEstimator(
                df=sub,
                outcome=outcome,
                treatment=treatment,
                controls=controls_avail,
                group_col="county_fips",
                panel_mode="two_way_within",
                entity_col="county_fips",
                time_col="year",
                n_folds=args.n_folds,
            )
            dml.fit()
            dml.save_results(spec_dir)
            summary = dml.summary()
            logger.info(
                "DML %s: theta=%.6f (SE=%.6f, p=%.4f).",
                label,
                summary["theta"],
                summary["se"],
                summary["p_value"],
            )
        except Exception:
            logger.exception("DML failed for %s.", label)

        # --- Causal Forest (heterogeneous effects) --- #
        if not args.skip_causal_forest:
            effect_mods = [c for c in _EFFECT_MODIFIERS if c in sub.columns]
            if not effect_mods:
                logger.warning(
                    "Skipping CausalForest for %s: no effect modifiers available.",
                    label,
                )
                continue

            logger.info(
                "Running CausalForest: %s -> %s (%d effect modifiers).",
                treatment,
                outcome,
                len(effect_mods),
            )
            try:
                cf = CausalForestEstimator(
                    df=sub,
                    outcome=outcome,
                    treatment=treatment,
                    controls=controls_avail,
                    effect_modifiers=effect_mods,
                )
                cf.fit()
                cf.save_results(spec_dir)
                ate = cf.ate_summary()
                logger.info(
                    "CausalForest %s: ATE=%.6f [%.6f, %.6f].",
                    label,
                    ate["ate"],
                    ate["ci_lower"],
                    ate["ci_upper"],
                )

                # Log top-3 most important features
                fi = cf.feature_importance()
                top3 = fi.head(3)
                for _, row in top3.iterrows():
                    logger.info(
                        "  Feature importance: %s = %.4f",
                        row["feature"],
                        row["importance"],
                    )
            except Exception:
                logger.exception("CausalForest failed for %s.", label)

    logger.info("DML models complete. Results in %s.", output_dir)


if __name__ == "__main__":
    main()
